import os
import pdb
from typing import List, Dict
from dataclasses import dataclass, field
import logging

import imageio
import trimesh
import numpy as np
import mujoco
import mujoco.viewer
import transforms3d.quaternions as tq

from dexonomy.util.np_rot_util import (
    np_interp_slide,
    np_interp_qpos,
    np_array32,
    np_transform_points,
    np_inv_transform_points,
    np_get_delta_pose,
)
from dexonomy.sim.basic import HandCfg, SimCfg


class MuJoCo_BaseEnv:
    def __init__(
        self,
        hand_cfg: HandCfg,
        scene_cfg: Dict | None = None,
        sim_cfg: SimCfg = SimCfg(),
        debug_render: bool = False,
        debug_viewer: bool = False,
    ):
        self.sim_cfg = sim_cfg
        self.spec = mujoco.MjSpec()
        self.spec.meshdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.spec.option.timestep = self.sim_cfg.timestep
        self.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_GRAVITY

        if debug_render or debug_viewer:
            self._add_for_visualization()

        self._add_hand(
            hand_cfg.xml_path,
            hand_cfg.freejoint,
            hand_cfg.arm_flag,
            hand_cfg.exclude_table_contact,
        )

        self._obj_init_qpos = []
        self._obj_nv = 0
        if scene_cfg is not None:
            for obj_name, obj_cfg in scene_cfg["scene"].items():
                obj_type = obj_cfg["type"]
                if obj_type in ["plane", "rigid_object", "articulated_object"]:
                    eval(f"self._add_{obj_type}")(obj_name, **obj_cfg)
                else:
                    raise NotImplementedError(
                        f"Unsupported object type: {obj_type}. Available choices: 'plane', 'rigid_object', 'articulated_object'."
                    )
            self._obj_init_qpos = np_array32(self._obj_init_qpos)

        self._set_friction(self.sim_cfg.friction_coef)
        self.spec.add_key()

        # Get ready for simulation
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.reset_qpos(self.get_hand_qpos(), set_ctrl=False)

        # Post-processing for object
        if scene_cfg is not None:
            obj_interest_name = scene_cfg["task"]["obj_name"]
            if scene_cfg["scene"][obj_interest_name]["type"] == "articulated_object":
                obj_interest_name += scene_cfg["task"]["part_name"]
            elif scene_cfg["scene"][obj_interest_name]["type"] == "rigid_object":
                obj_interest_name += "world"
            else:
                raise NotImplementedError
            self._obj_interest_bodyid = self.model.body(
                self.sim_cfg.obj_prefix + obj_interest_name
            ).id

        # For general ctrl
        self._hand_qpos2ctrl_mat = np.zeros((self.model.nu, self.model.nv))
        mujoco.mju_sparse2dense(
            self._hand_qpos2ctrl_mat,
            self.data.actuator_moment,
            self.data.moment_rownnz,
            self.data.moment_rowadr,
            self.data.moment_colind,
        )
        self._hand_qpos2ctrl_mat = self._hand_qpos2ctrl_mat[
            ..., : self.model.nv - self._obj_nv
        ]

        # For IK-based ctrl
        if hand_cfg.arm_flag and hand_cfg.ee_name is not None:
            self._ik_body_id = self.model.body(
                self.sim_cfg.hand_prefix + hand_cfg.ee_name
            ).id
            self._ik_jac = np.zeros((6, self.model.nv))
            self._ik_damping = 1e-4 * np.eye(6)
            self._ik_error = np.zeros(6)
            self._ik_quat_conj = np.zeros(4)
            self._ik_error_quat = np.zeros(4)
            self._ik_real_pose = np.zeros(7)

        self.debug_viewer = None
        self.debug_render = None
        if debug_render or debug_viewer:
            with open("debug.xml", "w") as f:
                f.write(self.spec.to_xml())

        if debug_viewer:
            self.debug_viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.debug_viewer.sync()
            pdb.set_trace()

        if debug_render:
            self.debug_render = mujoco.Renderer(self.model, 480, 640)
            self.debug_options = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self.debug_options)
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
            self.debug_images = []
        return

    def _add_hand(
        self,
        xml_path: str,
        freejoint: bool,
        arm_flag: bool,
        exclude_table_contact: List[str] | None,
    ):
        # Read hand xml
        child_spec = mujoco.MjSpec.from_file(xml_path)
        for m in child_spec.meshes:
            m.file = os.path.join(os.path.dirname(xml_path), child_spec.meshdir, m.file)
        child_spec.meshdir = self.spec.meshdir
        self.hand_body_num = len(child_spec.bodies)
        for g in child_spec.geoms:
            # This solimp and solref comes from the Shadow Hand xml
            # The body will be more "rigid" and less "soft"
            g.solimp[:3] = [0.5, 0.99, 0.0001]
            g.solref[:2] = [0.005, 1]
            g.margin = self.sim_cfg.hand_margin

        attach_frame = self.spec.worldbody.add_frame()
        child_world = attach_frame.attach_body(
            child_spec.worldbody, self.sim_cfg.hand_prefix, ""
        )
        # Add freejoint and mocap of hand root
        if not arm_flag and freejoint:
            child_world.add_freejoint(name="hand_freejoint")
            self.spec.worldbody.add_body(name="mocap_body", mocap=True)
            self.spec.add_equality(
                type=mujoco.mjtEq.mjEQ_WELD,
                name1="mocap_body",
                name2=f"{self.sim_cfg.hand_prefix}world",
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                solimp=self.sim_cfg.hand_mocap_solimp,
                data=self.sim_cfg.hand_mocap_data,
            )
        else:
            if exclude_table_contact is not None:
                for body_name in exclude_table_contact:
                    self.spec.add_exclude(
                        bodyname1="world",
                        bodyname2=f"{self.sim_cfg.hand_prefix}{body_name}",
                    )
        return

    def _add_rigid_object(self, name, xml_path, pose, scale, density=1000, **kwargs):
        child_spec = mujoco.MjSpec.from_file(xml_path)
        for m in child_spec.meshes:
            m.file = os.path.join(os.path.dirname(xml_path), child_spec.meshdir, m.file)
        child_spec.meshdir = self.spec.meshdir
        for g in child_spec.geoms:
            if g.contype != 0:
                g.density = density
                g.margin = self.sim_cfg.obj_margin
        for m in child_spec.meshes:
            m.scale *= scale
        attach_frame = self.spec.worldbody.add_frame()
        child_world = attach_frame.attach_body(
            child_spec.worldbody, self.sim_cfg.obj_prefix + name, ""
        )
        child_world.pos = pose[:3]
        child_world.quat = pose[3:]
        if self.sim_cfg.obj_freejoint:
            child_world.add_freejoint(name=f"{name}_freejoint")
            self._obj_init_qpos.extend(pose)
            self._obj_nv += 6
        return

    def _add_articulated_object(self, name, xml_path, pose, scale, fix_root, **kwargs):
        child_spec = mujoco.MjSpec.from_file(xml_path)
        for m in child_spec.meshes:
            m.file = os.path.join(os.path.dirname(xml_path), child_spec.meshdir, m.file)
        child_spec.meshdir = self.spec.meshdir
        for g in child_spec.geoms:
            if g.contype != 0:
                g.margin = self.sim_cfg.obj_margin
        for m in child_spec.meshes:
            m.scale *= scale

        if not self.sim_cfg.obj_freejoint:
            for j in child_spec.joints:
                j.delete()
        for a in child_spec.actuators:
            a.delete()

        attach_frame = self.spec.worldbody.add_frame()
        child_world = attach_frame.attach_body(
            child_spec.worldbody, self.sim_cfg.obj_prefix + name, ""
        )
        child_world.pos = pose[:3]
        child_world.quat = pose[3:]
        if self.sim_cfg.obj_freejoint:
            if not fix_root:
                child_world.add_freejoint(name=f"{name}_freejoint")
                self._obj_init_qpos.extend(pose)
                self._obj_nv += 6
            self._obj_init_qpos.extend([0] * len(child_spec.joints))
            self._obj_nv += len(child_spec.joints)
        return

    def _add_plane(
        self, name, pose=[0.0, 0, 0, 1, 0, 0, 0], size=[0, 0, 1.0], **kwargs
    ):
        self.spec.worldbody.add_geom(
            name=self.sim_cfg.plane_prefix + name + "_visual",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=pose[:3],
            quat=pose[3:],
            conaffinity=0,
            contype=0,
            size=size,
        )
        self.spec.worldbody.add_geom(
            name=self.sim_cfg.plane_prefix + name + "_collision",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=pose[:3],
            quat=pose[3:],
            size=size,
            margin=self.sim_cfg.plane_margin,
        )
        self.plane_num = 1
        return

    def _add_for_visualization(self):
        self.spec.add_texture(
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[0.3, 0.5, 0.7],
            rgb2=[0.3, 0.5, 0.7],
            width=512,
            height=512,
        )
        self.spec.worldbody.add_light(
            name="spotlight",
            pos=[0, -1, 2],
            castshadow=False,
        )
        self.spec.worldbody.add_camera(
            name="closeup", pos=[0.5, -1.0, 1.0], xyaxes=[1, 0, 0, 0, 1, 1]
        )

    def _set_friction(self, friction_coef: tuple[float, float]):
        raise NotImplementedError

    def _get_qpos_with_ik(self, hand_qpos) -> np.ndarray:
        # Solve system of equations: J @ dq = error.
        mujoco.mj_jacBody(
            self.model, self.data, self._ik_jac[:3], self._ik_jac[3:], self._ik_body_id
        )
        mujoco.mju_mulPose(
            self._ik_real_pose[:3],
            self._ik_real_pose[3:],
            hand_qpos[:3],
            hand_qpos[3:7],
            self.model.body_pos[self._ik_body_id],
            self.model.body_quat[self._ik_body_id],
        )

        self._ik_error[:3] = self._ik_real_pose[:3] - self.data.xpos[self._ik_body_id]
        mujoco.mju_negQuat(self._ik_quat_conj, self.data.xquat[self._ik_body_id])
        mujoco.mju_mulQuat(
            self._ik_error_quat, self._ik_real_pose[3:7], self._ik_quat_conj
        )
        mujoco.mju_quat2Vel(self._ik_error[3:], self._ik_error_quat, 1.0)
        dq = self._ik_jac.T @ np.linalg.solve(
            self._ik_jac @ self._ik_jac.T + self._ik_damping, self._ik_error
        )
        # Integrate joint velocities to obtain joint positions.
        new_qpos = self.data.qpos.copy()
        obj_nq = len(self._obj_init_qpos)
        new_qpos[len(new_qpos) - len(hand_qpos) - obj_nq + 7 : -obj_nq] = hand_qpos[7:]
        mujoco.mj_integratePos(self.model, new_qpos, dq, 1.0)
        return new_qpos[:-obj_nq]

    def set_ctrl(self, hand_qpos, ctype="ee_pose", skip_mocap=False) -> None:
        if ctype == "joint_angle":
            self.data.ctrl[:] = self._hand_qpos2ctrl_mat @ hand_qpos
        elif ctype == "ee_pose":
            if len(self.data.mocap_pos) != 0:
                if not skip_mocap:
                    self.data.mocap_pos[0] = hand_qpos[:3]
                    self.data.mocap_quat[0] = hand_qpos[3:7]
                self.data.ctrl[:] = self._hand_qpos2ctrl_mat[:, 6:] @ hand_qpos[7:]
            else:
                self.data.ctrl[:] = self._hand_qpos2ctrl_mat @ self._get_qpos_with_ik(
                    hand_qpos
                )
        else:
            raise NotImplementedError(
                f"Unsupported control type: {ctype}. Available choices: ['ee_pose', 'joint_angle']"
            )

    def get_interest_object_pose(self) -> np.ndarray:
        return np.concatenate(
            [
                self.data.xpos[self._obj_interest_bodyid],
                self.data.xquat[self._obj_interest_bodyid],
            ]
        )

    def set_interest_object_extdir(self, extdir) -> None:
        obj_gravity = self.model.body_subtreemass[self._obj_interest_bodyid] * 9.8
        if obj_gravity < 0.001:
            logging.error(
                f"Too small object gravity: {obj_gravity}. Please check the object density."
            )
        extforce = extdir * obj_gravity
        if np.linalg.norm(extforce) < 0.001:
            logging.error(
                f"Too small external force: {extforce}. Please check the axis in task of scene cfg."
            )
        self.data.xfrc_applied[self._obj_interest_bodyid][:3] = extforce

    def get_hand_qpos(self) -> np.ndarray:
        return self.data.qpos[: self.model.nq - len(self._obj_init_qpos)]

    def get_contact_info(self, obj_margin=None) -> tuple[list[dict], list[dict]]:
        if obj_margin is not None:
            self.set_obj_margin(obj_margin, temporal=True)
            mujoco.mj_forward(self.model, self.data)

        # object body id > object_id. hand_id >= hand body id > world_id.
        object_id = self.hand_body_num + len(self.data.mocap_pos)
        hand_id = self.hand_body_num
        world_id = -1 if self.plane_num == 0 else 0

        # Processing all contact information
        ho_contact = []
        hh_contact = []
        for contact in self.data.contact:
            body1_id = self.model.geom(contact.geom1).bodyid[0]
            body2_id = self.model.geom(contact.geom2).bodyid[0]
            body1_name = self.model.body(self.model.geom(contact.geom1).bodyid).name
            body2_name = self.model.body(self.model.geom(contact.geom2).bodyid).name
            # hand and object
            if (
                body1_id > world_id and body1_id <= hand_id and body2_id > object_id
            ) or (body2_id > world_id and body2_id <= hand_id and body1_id > object_id):
                if body2_id > object_id:
                    contact_normal = contact.frame[0:3]
                    hand_body_name = body1_name.removeprefix(self.sim_cfg.hand_prefix)
                    obj_body_name = body2_name
                else:
                    contact_normal = -contact.frame[0:3]
                    hand_body_name = body2_name.removeprefix(self.sim_cfg.hand_prefix)
                    obj_body_name = body1_name
                ho_contact.append(
                    {
                        "contact_dist": contact.dist,
                        "contact_pos": contact.pos
                        - contact.dist * contact_normal,  # point on hand
                        "contact_normal": contact_normal,  # from body1 to body2
                        "body1_name": hand_body_name,
                        "body2_name": obj_body_name,
                    }
                )
            # hand and hand
            elif (
                body1_id > world_id
                and body1_id <= hand_id
                and body2_id > world_id
                and body2_id <= hand_id
            ):
                hh_contact.append(
                    {
                        "contact_dist": contact.dist,
                        "contact_pos": contact.pos,
                        "contact_normal": contact.frame[0:3],
                        "body1_name": body1_name,
                        "body2_name": body2_name,
                    }
                )
            # else:
            #     print(body1_name, body2_name, body1_id, body2_id)

        # Set margin and gap back
        if obj_margin is not None:
            self.set_obj_margin(self.sim_cfg.obj_margin, temporal=True)
        return ho_contact, hh_contact

    def set_obj_margin(self, obj_margin, temporal=False) -> None:
        if not temporal:
            self.sim_cfg.obj_margin = obj_margin
        for i in range(self.model.ngeom):
            if self.model.geom(i).name.startswith(self.sim_cfg.obj_prefix):
                self.model.geom_margin[i] = obj_margin
        return

    def reset_qpos(self, hand_qpos, set_ctrl=True) -> None:
        # set key frame
        self.model.key_qvel[0] = 0
        self.model.key_act[0] = 0
        self.model.key_qpos[0] = np.concatenate(
            [hand_qpos, self._obj_init_qpos], axis=0
        )
        if len(self.data.mocap_pos) > 0:
            self.model.key_mpos[0] = hand_qpos[:3]
            self.model.key_mquat[0] = hand_qpos[3:7]
        if set_ctrl:
            self.set_ctrl(hand_qpos)
            self.model.key_ctrl[0] = np.copy(self.data.ctrl)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        return

    def control_with_interp(
        self, qpos_lst, ctype_lst, extdir_lst, interp_lst, step_inner=10
    ) -> np.ndarray:
        real_qpos_lst = []
        for i in range(len(qpos_lst) - 1):
            if len(qpos_lst[i]) == len(qpos_lst[i + 1]):
                interp_qpos_lst = np_interp_qpos(
                    qpos_lst[i], qpos_lst[i + 1], interp_lst[i]
                )
                if ctype_lst[i] == "ee_pose":
                    interp_qpos_lst[:, :7] = np_interp_slide(
                        qpos_lst[i][:7], qpos_lst[i + 1][:7], interp_lst[i]
                    )
            else:
                interp_qpos_lst = [qpos_lst[i + 1]] * interp_lst[i]
            if extdir_lst[i] is not None:
                self.set_interest_object_extdir(extdir_lst[i])
            for j in range(interp_lst[i]):
                self.set_ctrl(interp_qpos_lst[j], ctype_lst[i])
                self.simulation_step(step_inner)
            real_qpos_lst.append(self.data.qpos.copy())
        return np.stack(real_qpos_lst, axis=0)

    def simulation_step(self, step_inner) -> None:
        mujoco.mj_forward(self.model, self.data)
        for _ in range(step_inner):
            mujoco.mj_step(self.model, self.data)

        if self.debug_render is not None:
            self.debug_render.update_scene(self.data, "closeup", self.debug_options)
            pixels = self.debug_render.render()
            self.debug_images.append(pixels)

        if self.debug_viewer is not None:
            self.debug_viewer.sync()
            pdb.set_trace()
        return

    def debug_postprocess(self, save_path=None) -> None:
        if self.debug_render is not None:
            assert save_path is not None
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.mimsave(save_path, self.debug_images)
            logging.info(f"Save GIF to {save_path}")

        if self.debug_viewer is not None:
            self.debug_viewer.close()


@dataclass
class MuJoCo_OptCfg(SimCfg):
    obj_freejoint: bool = False
    hand_mocap_solimp: List[float] = field(
        default_factory=lambda: [0.01, 0.095, 1, 0.5, 2]
    )  # looser constraints
    hand_mocap_data: List[float] = field(
        default_factory=lambda: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10]
    )  # looser constraints
    obj_margin: float = 0.001
    hand_margin: float = 0.001
    plane_margin: float = 0.02


class MuJoCo_OptEnv(MuJoCo_BaseEnv):
    def __init__(
        self,
        hand_cfg: HandCfg,
        scene_cfg: Dict | None = None,
        sim_cfg: SimCfg = MuJoCo_OptCfg(),
        debug_render: bool = False,
        debug_viewer: bool = False,
    ):
        super().__init__(hand_cfg, scene_cfg, sim_cfg, debug_render, debug_viewer)
        return

    def _set_friction(self, friction_coef):
        self.spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        for g in self.spec.geoms:
            g.condim = 1
        return

    def get_hand_worldframe_contact(
        self, hand_body_name, hand_bodyframe_contact
    ) -> np.ndarray:
        hand_worldframe_contact = []
        for body_name, hf_c in zip(hand_body_name, hand_bodyframe_contact):
            body_id = self.model.body(self.sim_cfg.hand_prefix + body_name).id
            br = self.data.xmat[body_id].reshape(3, 3)
            bp = self.data.xpos[body_id]
            hand_worldframe_contact.append(np_transform_points(hf_c, br, bp))
        return np_array32(hand_worldframe_contact)

    def get_hand_bodyframe_contact(
        self, hand_body_name, hand_worldframe_contact
    ) -> np.ndarray:
        hand_bodyframe_contact = []
        for body_name, wf_c in zip(hand_body_name, hand_worldframe_contact):
            body_id = self.model.body(self.sim_cfg.hand_prefix + body_name).id
            br = self.data.xmat[body_id].reshape(3, 3)
            bp = self.data.xpos[body_id]
            hand_bodyframe_contact.append(np_inv_transform_points(wf_c, br, bp))
        return np_array32(hand_bodyframe_contact)

    def apply_force_on_hand(
        self, hand_body_name, hand_bodyframe_contact, obj_worldframe_contact
    ) -> np.ndarray:
        self.data.qfrc_applied[:] = 0
        self.data.xfrc_applied[:] = 0
        total_loss = []
        for i, body_name in enumerate(hand_body_name):
            body_id = self.model.body(self.sim_cfg.hand_prefix + body_name).id
            br = self.data.xmat[body_id].reshape(3, 3)
            bp = self.data.xpos[body_id]
            hcp_world = br @ hand_bodyframe_contact[i, :3] + bp
            ocp_world = obj_worldframe_contact[i, :3]
            total_loss.append(np.linalg.norm(ocp_world - hcp_world))
            delta_normal = (
                np.sum(
                    (ocp_world - hcp_world) * obj_worldframe_contact[i, 3:],
                    axis=-1,
                    keepdims=True,
                )
                * obj_worldframe_contact[i, 3:]
            )
            delta_tangent = ocp_world - hcp_world - delta_normal
            point_force = delta_normal * 500 + delta_tangent * 100
            mujoco.mj_applyFT(
                self.model,
                self.data,
                point_force,
                0 * point_force,
                hcp_world,
                body_id,
                self.data.qfrc_applied,
            )
        self.set_ctrl(self.data.qpos, skip_mocap=True)
        self.data.qvel[:] = 0
        return np_array32(total_loss)

    def keep_hand_stable(self) -> None:
        self.data.qfrc_applied[:] = 0
        self.data.xfrc_applied[:] = 0
        self.set_ctrl(self.data.qpos, skip_mocap=True)
        self.data.qvel[:] = 0
        return

    def get_squeeze_qpos(
        self, grasp_qpos, hand_body_name, hand_worldframe_contact, contact_wrench
    ) -> np.ndarray:
        self.data.qfrc_applied[:] = 0
        for hbn, hcw, hwc in zip(
            hand_body_name, contact_wrench, hand_worldframe_contact
        ):
            mujoco.mj_applyFT(
                self.model,
                self.data,
                hcw[:3],
                0 * hcw[3:],
                hwc[:3],
                self.model.body(self.sim_cfg.hand_prefix + hbn).id,
                self.data.qfrc_applied,
            )
        delta_qpos = np.copy(self.data.qfrc_applied)
        self.data.qfrc_applied[:] = 0
        actuator_gainprm = self.model.actuator_gainprm[:, 0]
        for i in range(len(delta_qpos)):
            actuator_id = np.where(self._hand_qpos2ctrl_mat[:, i] != 0)[0]
            if len(actuator_id) > 0:
                delta_qpos[i] /= (
                    actuator_gainprm[actuator_id[0]]
                    * self._hand_qpos2ctrl_mat[actuator_id[0]].sum()
                )
        squeeze_qpos = np.concatenate([grasp_qpos[:7], grasp_qpos[7:] + delta_qpos[6:]])
        return squeeze_qpos


@dataclass
class MuJoCo_TestCfg(SimCfg):
    timestep: float = 0.004
    hand_mocap_solimp: List[float] = field(
        default_factory=lambda: [0.9, 0.95, 0.001, 0.5, 2]
    )
    hand_mocap_data: List[float] = field(
        default_factory=lambda: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    )


class MuJoCo_TestEnv(MuJoCo_BaseEnv):
    def __init__(
        self,
        hand_cfg: HandCfg,
        scene_cfg: Dict | None = None,
        sim_cfg: SimCfg = MuJoCo_TestCfg(),
        debug_render: bool = False,
        debug_viewer: bool = True,
    ):
        super().__init__(hand_cfg, scene_cfg, sim_cfg, debug_render, debug_viewer)
        return

    def _set_friction(self, friction_coef):
        self.spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.spec.option.noslip_iterations = 2
        self.spec.option.impratio = 10
        for g in self.spec.geoms:
            g.friction[:2] = friction_coef
            g.condim = 4
        return

    def test_fc(
        self,
        qpos_lst,
        ctype_lst,
        extdir_lst,
        interp_lst,
        target_obj_pose,
        trans_thre,
        angle_thre,
    ) -> tuple[bool, None]:
        external_force_direction = np.array(
            [[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        )

        for extforce_dir in external_force_direction:
            self.reset_qpos(qpos_lst[0])
            self.control_with_interp(qpos_lst, ctype_lst, extdir_lst, interp_lst)
            self.set_interest_object_extdir(extforce_dir)
            for _ in range(10):
                self.simulation_step(step_inner=50)
                latter_obj_pose = self.get_interest_object_pose()
                delta_pos, delta_angle = np_get_delta_pose(
                    target_obj_pose, latter_obj_pose
                )
                succ_flag = (delta_pos < trans_thre) & (delta_angle < angle_thre)
                if not succ_flag:
                    break
            if not succ_flag:
                break
        return succ_flag, None

    def test_move(
        self,
        qpos_lst,
        ctype_lst,
        extdir_lst,
        interp_lst,
        target_obj_pose,
        trans_thre,
        angle_thre,
    ) -> tuple[bool, np.ndarray]:
        self.reset_qpos(qpos_lst[0])
        real_qpos_lst = self.control_with_interp(
            qpos_lst, ctype_lst, extdir_lst, interp_lst
        )
        self.simulation_step(step_inner=50)
        latter_obj_pose = self.get_interest_object_pose()
        delta_pos, delta_angle = np_get_delta_pose(target_obj_pose, latter_obj_pose)
        succ_flag = (delta_pos < trans_thre) & (delta_angle < angle_thre)
        return succ_flag, real_qpos_lst


class MuJoCo_VisEnv(MuJoCo_BaseEnv):
    def __init__(
        self,
        hand_cfg: HandCfg,
        scene_cfg: Dict | None = None,
        sim_cfg: SimCfg = MuJoCo_TestCfg(),
        vis_mesh_mode: str | None = None,
        debug_render: bool = False,
        debug_viewer: bool = False,
    ):
        super().__init__(hand_cfg, scene_cfg, sim_cfg, debug_render, debug_viewer)
        assert (
            vis_mesh_mode == "visual"
            or vis_mesh_mode == "collision"
            or vis_mesh_mode is None
        )

        self.body_mesh_dict = {}
        self.body_id_dict = {}
        for i in range(self.model.ngeom):
            geom = self.model.geom(i)
            mesh_id = geom.dataid
            body_id = geom.bodyid[0]
            body_name = self.model.body(body_id).name
            self.body_id_dict[body_name] = body_id

            if geom.contype == 0 and vis_mesh_mode != "visual":
                continue
            if geom.contype != 0 and vis_mesh_mode != "collision":
                continue

            if mesh_id == -1:  # Primitives
                if geom.type == 6:
                    tm = trimesh.creation.box(extents=2 * geom.size)
                elif geom.type == 5:
                    tm = trimesh.creation.cylinder(
                        radius=geom.size[0], height=2 * geom.size[1]
                    )
                elif geom.type == 2:
                    tm = trimesh.creation.icosphere(radius=geom.size[0])
                elif geom.type == 3:
                    tm = trimesh.creation.capsule(
                        radius=geom.size[0], height=2 * geom.size[1]
                    )
                elif geom.type == 0:
                    tm = trimesh.creation.box(extents=[2*geom.size[-1], 2*geom.size[-1], 0.001])
                else:
                    raise NotImplementedError(
                        f"Unsupported mujoco primitive type: {geom.type}. Available choices: 2(icosphere), 3(capsule), 5(cylinder), 6(box)."
                    )
            else:  # Meshes
                mjm = self.model.mesh(mesh_id)
                vert = self.model.mesh_vert[
                    mjm.vertadr[0] : mjm.vertadr[0] + mjm.vertnum[0]
                ]
                face = self.model.mesh_face[
                    mjm.faceadr[0] : mjm.faceadr[0] + mjm.facenum[0]
                ]
                tm = trimesh.Trimesh(vertices=vert, faces=face)

            if vis_mesh_mode == "collision":
                tm = tm.convex_hull

            geom_rot = self.data.geom_xmat[i].reshape(3, 3)
            geom_trans = self.data.geom_xpos[i]
            body_rot = self.data.xmat[body_id].reshape(3, 3)
            body_trans = self.data.xpos[body_id]
            tm.vertices = (
                tm.vertices @ geom_rot.T + geom_trans - body_trans
            ) @ body_rot

            if body_name not in self.body_mesh_dict:
                self.body_mesh_dict[body_name] = [tm]
            else:
                self.body_mesh_dict[body_name].append(tm)

        for body_name, body_mesh in self.body_mesh_dict.items():
            self.body_mesh_dict[body_name] = trimesh.util.concatenate(body_mesh)
        return

    def _set_friction(self, friction_coef):
        self.spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        for g in self.spec.geoms:
            g.condim = 1
        return

    def forward_kinematics(self, qpos) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos = qpos
        mujoco.mj_kinematics(self.model, self.data)
        xmat = np_array32(self.data.xmat).reshape(-1, 3, 3)
        xpos = np_array32(self.data.xpos)
        return xmat, xpos

    def get_init_body_meshes(self) -> tuple[list[str], list[trimesh.Trimesh]]:
        return list(self.body_mesh_dict.keys()), list(self.body_mesh_dict.values())

    def get_posed_meshes(self, xmat, xpos, vis_prefix=[""]) -> trimesh.Trimesh:
        full_tm = []
        for k, v in self.body_mesh_dict.items():
            if not any(k.startswith(p) for p in vis_prefix):
                continue
            body_rot = xmat[self.body_id_dict[k]].reshape(3, 3)
            body_trans = xpos[self.body_id_dict[k]]
            posed_vert = v.vertices @ body_rot.T + body_trans
            posed_tm = trimesh.Trimesh(vertices=posed_vert, faces=v.faces)
            full_tm.append(posed_tm)
        full_tm = trimesh.util.concatenate(full_tm)
        return full_tm


if __name__ == "__main__":
    xml_path = os.path.join(
        os.path.dirname(__file__), "../../assets/hand/shadow/right.xml"
    )
    kinematic = MuJoCo_VisEnv(
        hand_cfg=HandCfg(xml_path=xml_path), vis_mesh_mode="visual"
    )
    hand_qpos = np.zeros((22))
    xmat, xpos = kinematic.forward_kinematics(hand_qpos)
    visual_mesh = kinematic.get_posed_meshes(xmat, xpos)
    visual_mesh.export(f"debug_hand.obj")
