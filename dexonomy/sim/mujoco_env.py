import os
from dataclasses import dataclass, field
import logging
import time

import imageio
import trimesh
import numpy as np
import mujoco
import mujoco.viewer

from dexonomy.util.np_util import (
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
        scene_cfg: dict | None = None,
        sim_cfg: SimCfg = SimCfg(),
        debug_render: bool = False,
        debug_view: bool = False,
    ):
        self._cfg = sim_cfg
        self._spec = mujoco.MjSpec()
        self._spec.meshdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self._spec.option.timestep = self._cfg.timestep
        self._spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self._spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_GRAVITY

        if debug_render or debug_view:
            self._add_for_visualization()

        self._add_hand(
            hand_cfg.xml_path,
            hand_cfg.freejoint,
            hand_cfg.arm_flag,
            hand_cfg.skip_table_cbody,
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

        self._set_friction(self._cfg.miu_coef)
        self._spec.add_key()

        # Get ready for simulation
        self._model = self._spec.compile()
        self._data = mujoco.MjData(self._model)
        self.reset_qpos(self.get_hand_qpos(), set_ctrl=False)

        # Post-processing for object
        if scene_cfg is not None:
            obj_active_name = scene_cfg["task"]["obj_name"]
            if scene_cfg["scene"][obj_active_name]["type"] == "articulated_object":
                obj_active_name += scene_cfg["task"]["part_name"]
            elif scene_cfg["scene"][obj_active_name]["type"] == "rigid_object":
                obj_active_name += "world"
            else:
                raise NotImplementedError
            self._obj_active_bodyid = self._model.body(
                self._cfg.obj_prefix + obj_active_name
            ).id

        # For general ctrl
        self._hand_qpos2ctrl_mat = np.zeros((self._model.nu, self._model.nv))
        mujoco.mju_sparse2dense(
            self._hand_qpos2ctrl_mat,
            self._data.actuator_moment,
            self._data.moment_rownnz,
            self._data.moment_rowadr,
            self._data.moment_colind,
        )
        self._hand_qpos2ctrl_mat = self._hand_qpos2ctrl_mat[
            ..., : self._model.nv - self._obj_nv
        ]

        # For IK-based ctrl
        if hand_cfg.arm_flag and hand_cfg.ee_name is not None:
            self._ik_body_id = self._model.body(
                self._cfg.hand_prefix + hand_cfg.ee_name
            ).id
            self._ik_jac = np.zeros((6, self._model.nv))
            self._ik_damping = 1e-4 * np.eye(6)
            self._ik_error = np.zeros(6)
            self._ik_quat_conj = np.zeros(4)
            self._ik_error_quat = np.zeros(4)
            self._ik_real_pose = np.zeros(7)

        self._debug_view = None
        self._debug_render = None
        if debug_render or debug_view:
            with open("debug.xml", "w") as f:
                f.write(self._spec.to_xml())

        if debug_view:
            self._debug_view = mujoco.viewer.launch_passive(self._model, self._data)
            self._debug_view.sync()
            time.sleep(5.0)

        if debug_render:
            self._debug_render = mujoco.Renderer(self._model, 480, 640)
            self._debug_option = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self._debug_option)
            self._debug_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self._debug_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            self._debug_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
            self._debug_img = []
        return

    def _add_hand(
        self,
        xml_path: str,
        freejoint: bool,
        arm_flag: bool,
        skip_table_cbody: list[str] | None,
    ):
        # Read hand xml
        child_spec = mujoco.MjSpec.from_file(xml_path)
        for m in child_spec.meshes:
            m.file = os.path.join(os.path.dirname(xml_path), child_spec.meshdir, m.file)
        child_spec.meshdir = self._spec.meshdir
        self.n_hand_body = len(child_spec.bodies)
        for j in child_spec.joints:
            j.margin = 0.1
        for g in child_spec.geoms:
            # This solimp and solref comes from the Shadow Hand xml
            # The body will be more "rigid" and less "soft"
            g.solimp[:3] = [0.5, 0.99, 0.0001]
            g.solref[:2] = [0.005, 1]
            g.margin = self._cfg.hand_margin

        attach_frame = self._spec.worldbody.add_frame()
        child_world = attach_frame.attach_body(
            child_spec.worldbody, self._cfg.hand_prefix, ""
        )
        # Add freejoint and mocap of hand root
        if not arm_flag and freejoint:
            child_world.add_freejoint(name="hand_freejoint")
            self._spec.worldbody.add_body(name="mocap_body", mocap=True)
            self._spec.add_equality(
                type=mujoco.mjtEq.mjEQ_WELD,
                name1="mocap_body",
                name2=f"{self._cfg.hand_prefix}world",
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                solimp=self._cfg.hand_mocap_solimp,
                data=self._cfg.hand_mocap_data,
            )
        else:
            if skip_table_cbody is not None:
                for body_name in skip_table_cbody:
                    self._spec.add_exclude(
                        bodyname1="world",
                        bodyname2=f"{self._cfg.hand_prefix}{body_name}",
                    )
        return

    def _add_rigid_object(self, name, xml_path, pose, scale, density=1000, **kwargs):
        child_spec = mujoco.MjSpec.from_file(xml_path)
        for m in child_spec.meshes:
            m.file = os.path.join(os.path.dirname(xml_path), child_spec.meshdir, m.file)
        child_spec.meshdir = self._spec.meshdir
        for g in child_spec.geoms:
            if g.contype != 0:
                g.density = density
                g.margin = self._cfg.obj_margin
        for m in child_spec.meshes:
            m.scale *= scale
        attach_frame = self._spec.worldbody.add_frame()
        child_world = attach_frame.attach_body(
            child_spec.worldbody, self._cfg.obj_prefix + name, ""
        )
        child_world.pos = pose[:3]
        child_world.quat = pose[3:]
        if self._cfg.obj_freejoint:
            child_world.add_freejoint(name=f"{name}_freejoint")
            self._obj_init_qpos.extend(pose)
            self._obj_nv += 6
        return

    def _add_articulated_object(self, name, xml_path, pose, scale, fix_root, **kwargs):
        child_spec = mujoco.MjSpec.from_file(xml_path)
        for m in child_spec.meshes:
            m.file = os.path.join(os.path.dirname(xml_path), child_spec.meshdir, m.file)
        child_spec.meshdir = self._spec.meshdir
        for g in child_spec.geoms:
            if g.contype != 0:
                g.margin = self._cfg.obj_margin
        for m in child_spec.meshes:
            m.scale *= scale

        if not self._cfg.obj_freejoint:
            for j in child_spec.joints:
                j.delete()
        for a in child_spec.actuators:
            a.delete()

        attach_frame = self._spec.worldbody.add_frame()
        child_world = attach_frame.attach_body(
            child_spec.worldbody, self._cfg.obj_prefix + name, ""
        )
        child_world.pos = pose[:3]
        child_world.quat = pose[3:]
        if self._cfg.obj_freejoint:
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
        self._spec.worldbody.add_geom(
            name=self._cfg.plane_prefix + name + "_visual",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=pose[:3],
            quat=pose[3:],
            conaffinity=0,
            contype=0,
            size=size,
        )
        self._spec.worldbody.add_geom(
            name=self._cfg.plane_prefix + name + "_collision",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=pose[:3],
            quat=pose[3:],
            size=size,
            margin=self._cfg.plane_margin,
        )
        return

    def _add_for_visualization(self):
        self._spec.add_texture(
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[0.3, 0.5, 0.7],
            rgb2=[0.3, 0.5, 0.7],
            width=512,
            height=512,
        )
        self._spec.worldbody.add_light(
            name="spotlight",
            pos=[0, -1, 2],
            castshadow=False,
        )
        self._spec.worldbody.add_camera(
            name="closeup", pos=[0.5, -1.0, 1.0], xyaxes=[1, 0, 0, 0, 1, 1]
        )

    def _set_friction(self, miu_coef: tuple[float, float]):
        raise NotImplementedError

    def _get_qpos_with_ik(self, hand_qpos) -> np.ndarray:
        # Solve system of equations: J @ dq = error.
        mujoco.mj_jacBody(
            self._model,
            self._data,
            self._ik_jac[:3],
            self._ik_jac[3:],
            self._ik_body_id,
        )
        mujoco.mju_mulPose(
            self._ik_real_pose[:3],
            self._ik_real_pose[3:],
            hand_qpos[:3],
            hand_qpos[3:7],
            self._model.body_pos[self._ik_body_id],
            self._model.body_quat[self._ik_body_id],
        )

        self._ik_error[:3] = self._ik_real_pose[:3] - self._data.xpos[self._ik_body_id]
        mujoco.mju_negQuat(self._ik_quat_conj, self._data.xquat[self._ik_body_id])
        mujoco.mju_mulQuat(
            self._ik_error_quat, self._ik_real_pose[3:7], self._ik_quat_conj
        )
        mujoco.mju_quat2Vel(self._ik_error[3:], self._ik_error_quat, 1.0)
        dq = self._ik_jac.T @ np.linalg.solve(
            self._ik_jac @ self._ik_jac.T + self._ik_damping, self._ik_error
        )
        # Integrate joint velocities to obtain joint positions.
        new_qpos = self._data.qpos.copy()
        obj_nq = len(self._obj_init_qpos)
        new_qpos[len(new_qpos) - len(hand_qpos) - obj_nq + 7 : -obj_nq] = hand_qpos[7:]
        mujoco.mj_integratePos(self._model, new_qpos, dq, 1.0)
        return new_qpos[:-obj_nq]

    def set_ctrl(self, hand_qpos, ctrl_type="ee_pose", skip_mocap=False) -> np.ndarray:
        if ctrl_type == "joint_angle":
            self._data.ctrl[:] = self._hand_qpos2ctrl_mat @ hand_qpos
        elif ctrl_type == "ee_pose":
            if len(self._data.mocap_pos) != 0:
                if not skip_mocap:
                    self._data.mocap_pos[0] = hand_qpos[:3]
                    self._data.mocap_quat[0] = hand_qpos[3:7]
                self._data.ctrl[:] = self._hand_qpos2ctrl_mat[:, 6:] @ hand_qpos[7:]
            else:
                hand_qpos = self._get_qpos_with_ik(hand_qpos)
                self._data.ctrl[:] = self._hand_qpos2ctrl_mat @ hand_qpos
        else:
            raise NotImplementedError(
                f"Unsupported control type: {ctrl_type}. Available choices: ['ee_pose', 'joint_angle']"
            )
        return hand_qpos

    def get_active_obj_pose(self) -> np.ndarray:
        return np.concatenate(
            [
                self._data.xpos[self._obj_active_bodyid],
                self._data.xquat[self._obj_active_bodyid],
            ]
        )

    def set_active_obj_ext_fdir(self, ext_fdir) -> None:
        obj_gravity = self._model.body_subtreemass[self._obj_active_bodyid] * 9.8
        if obj_gravity < 0.001:
            logging.error(
                f"Too small object gravity: {obj_gravity}. Please check the object density."
            )
        extforce = ext_fdir * obj_gravity
        if np.linalg.norm(extforce) < 0.001:
            logging.error(
                f"Too small external force: {extforce}. Please check the axis in task of scene cfg."
            )
        self._data.qfrc_applied[:] = 0
        mujoco.mj_applyFT(
            self._model,
            self._data,
            extforce,
            extforce * 0,
            self._data.subtree_com[self._obj_active_bodyid],
            self._obj_active_bodyid,
            self._data.qfrc_applied,
        )

    def get_hand_qpos(self) -> np.ndarray:
        return self._data.qpos[: self._model.nq - len(self._obj_init_qpos)]

    def check_qpos_limit(self, thre: float = 0.0) -> bool:
        for j in range(self._model.njnt):
            if self._model.jnt_limited[j]:
                hand_qpos_j = self._data.qpos[self._model.jnt_qposadr[j]]
                limit_lower = self._model.jnt_range[j, 0]
                limit_upper = self._model.jnt_range[j, 1]
                if hand_qpos_j < limit_lower + thre or hand_qpos_j > limit_upper - thre:
                    return False
        return True

    def get_contacts(self, obj_margin=None) -> tuple[dict, dict]:
        if obj_margin is not None:
            self.set_obj_margin(obj_margin, temporal=True)
            mujoco.mj_forward(self._model, self._data)

        # object body id > object_id. hand_id >= hand body id > world_id.
        object_id = self.n_hand_body + len(self._data.mocap_pos)
        hand_id, world_id = self.n_hand_body, 0

        # Processing all contact information
        ho_contact = {"dist": [], "pos": [], "normal": [], "bn1": [], "bn2": []}
        hh_contact = {"dist": [], "pos": [], "normal": [], "bn1": [], "bn2": []}
        for contact in self._data.contact:
            body1_id = self._model.geom(contact.geom1).bodyid[0]
            body2_id = self._model.geom(contact.geom2).bodyid[0]
            body1_name = self._model.body(self._model.geom(contact.geom1).bodyid).name
            body2_name = self._model.body(self._model.geom(contact.geom2).bodyid).name
            # hand and object
            if (
                body1_id > world_id and body1_id <= hand_id and body2_id > object_id
            ) or (body2_id > world_id and body2_id <= hand_id and body1_id > object_id):
                if body2_id > object_id:
                    contact_normal = contact.frame[0:3]
                    hand_cb = body1_name.removeprefix(self._cfg.hand_prefix)
                    obj_cb = body2_name
                else:
                    contact_normal = -contact.frame[0:3]
                    hand_cb = body2_name.removeprefix(self._cfg.hand_prefix)
                    obj_cb = body1_name
                ho_contact["dist"].append(contact.dist)
                ho_contact["pos"].append(contact.pos - contact.dist * contact_normal)
                ho_contact["normal"].append(contact_normal)  # from body1 to body2
                ho_contact["bn1"].append(hand_cb)
                ho_contact["bn2"].append(obj_cb)
            # hand and hand
            elif (
                body1_id > world_id
                and body1_id <= hand_id
                and body2_id > world_id
                and body2_id <= hand_id
            ):
                hh_contact["dist"].append(contact.dist)
                hh_contact["pos"].append(contact.pos)
                hh_contact["normal"].append(contact.frame[0:3])
                hh_contact["bn1"].append(body1_name)
                hh_contact["bn2"].append(body2_name)

        ho_contact = {
            k: np_array32(v) if "bn" not in k else v for k, v in ho_contact.items()
        }
        hh_contact = {
            k: np_array32(v) if "bn" not in k else v for k, v in hh_contact.items()
        }

        # Set margin and gap back
        if obj_margin is not None:
            self.set_obj_margin(self._cfg.obj_margin, temporal=True)
        return ho_contact, hh_contact

    def set_obj_margin(self, obj_margin, temporal=False) -> None:
        if not temporal:
            self._cfg.obj_margin = obj_margin
        for i in range(self._model.ngeom):
            if self._model.geom(i).name.startswith(self._cfg.obj_prefix):
                self._model.geom_margin[i] = obj_margin
        return

    def reset_qpos(self, hand_qpos, set_ctrl=True) -> None:
        # set key frame
        self._model.key_qvel[0] = 0
        self._model.key_act[0] = 0
        self._model.key_qpos[0] = np.concatenate(
            [hand_qpos, self._obj_init_qpos], axis=0
        )
        if len(self._data.mocap_pos) > 0:
            self._model.key_mpos[0] = hand_qpos[:3]
            self._model.key_mquat[0] = hand_qpos[3:7]
        if set_ctrl:
            self.set_ctrl(hand_qpos)
            self._model.key_ctrl[0] = np.copy(self._data.ctrl)

        mujoco.mj_resetDataKeyframe(self._model, self._data, 0)
        mujoco.mj_forward(self._model, self._data)
        return

    def run_trajectory(
        self, ctrl_qpos, ctrl_type, ext_fdir, step=5, substep=5
    ) -> tuple[np.ndarray, np.ndarray]:
        real_state_qpos = []
        real_ctrl_qpos = []
        for i in range(len(ctrl_qpos) - 1):
            if len(ctrl_qpos[i]) == len(ctrl_qpos[i + 1]):
                interp_qpos = np_interp_qpos(ctrl_qpos[i], ctrl_qpos[i + 1], step)
                if ctrl_type[i] == "ee_pose":
                    interp_qpos[:, :7] = np_interp_slide(
                        ctrl_qpos[i][:7], ctrl_qpos[i + 1][:7], step
                    )
            else:
                interp_qpos = [ctrl_qpos[i + 1]] * step
            if ext_fdir[i] is not None:
                self.set_active_obj_ext_fdir(ext_fdir[i])
            for j in range(step):
                real_ctrl = self.set_ctrl(interp_qpos[j], ctrl_type[i])
                self.step_sim(substep)
            real_state_qpos.append(self._data.qpos.copy())
            real_ctrl_qpos.append(real_ctrl.copy())
        return np.stack(real_state_qpos, axis=0), np.stack(real_ctrl_qpos, axis=0)

    def step_sim(self, substep) -> None:
        for _ in range(substep):
            mujoco.mj_step(self._model, self._data)

        if self._debug_render is not None:
            self._debug_render.update_scene(self._data, "closeup", self._debug_option)
            pixels = self._debug_render.render()
            self._debug_img.append(pixels)

        if self._debug_view is not None:
            self._debug_view.sync()
            time.sleep(1.0)
        return

    def save_debug(self, save_path=None) -> None:
        if self._debug_render is not None:
            assert save_path is not None
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.mimsave(save_path.replace(".npy", ".gif"), self._debug_img)
            logging.info(f"Save GIF to {save_path}")

        if self._debug_view is not None:
            self._debug_view.close()


@dataclass
class MuJoCo_OptCfg(SimCfg):
    obj_freejoint: bool = False
    hand_mocap_solimp: list[float] = field(
        default_factory=lambda: [0.01, 0.095, 1, 0.5, 2]
    )  # looser constraints
    hand_mocap_data: list[float] = field(
        default_factory=lambda: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10]
    )  # looser constraints
    obj_margin: float = 0.001
    hand_margin: float = 0.001
    plane_margin: float = 0.02


class MuJoCo_OptEnv(MuJoCo_BaseEnv):
    def __init__(
        self,
        hand_cfg: HandCfg,
        scene_cfg: dict | None = None,
        sim_cfg: SimCfg = MuJoCo_OptCfg(),
        debug_render: bool = False,
        debug_view: bool = False,
    ):
        super().__init__(hand_cfg, scene_cfg, sim_cfg, debug_render, debug_view)
        return

    def _set_friction(self, miu_coef):
        self._spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        for g in self._spec.geoms:
            g.condim = 1
        return

    def transform_cpn_b2w(
        self, hand_cbody: list[str], hand_cpn_b: np.ndarray
    ) -> np.ndarray:
        """
        Args:
            hand_cbody: list of hand contact bodies
            hand_cpn_b: (N, 6) hand contact points and normals in hand body frame
        Returns:
            hand_cpn_w: (N, 6) handcontact points and normals in world frame
        """
        hand_cpn_w = []
        for h_cb, h_cpn in zip(hand_cbody, hand_cpn_b):
            b_id = self._model.body(self._cfg.hand_prefix + h_cb).id
            b_r = self._data.xmat[b_id].reshape(3, 3)
            b_t = self._data.xpos[b_id]
            hand_cpn_w.append(np_transform_points(h_cpn, b_r, b_t))
        return np_array32(hand_cpn_w)

    def transform_cpn_w2b(
        self, hand_cbody: list[str], hand_cpn_w: np.ndarray
    ) -> np.ndarray:
        """
        Args:
            hand_cbody: list of hand contact bodies
            hand_cpn_w: (N, 6) hand contact points and normals in world frame
        Returns:
            hand_cpn_b: (N, 6) hand contact points and normals in hand body frame
        """
        hand_cpn_b = []
        for h_cb, h_cpn in zip(hand_cbody, hand_cpn_w):
            b_id = self._model.body(self._cfg.hand_prefix + h_cb).id
            b_r = self._data.xmat[b_id].reshape(3, 3)
            b_t = self._data.xpos[b_id]
            hand_cpn_b.append(np_inv_transform_points(h_cpn, b_r, b_t))
        return np_array32(hand_cpn_b)

    def apply_contact_forces(
        self,
        hand_cbody: list[str],
        hand_cpn_b: np.ndarray,
        obj_cpn_w: np.ndarray,
    ) -> np.ndarray:
        """
        Args:
            hand_cbody: list of hand contact bodies
            hand_cpn_b: (N, 6) hand contact points and normals in hand body frame
            obj_cpn_w: (N, 6) object contact points and normals in world frame
        Returns:
            contact_diffs: (N,) difference between object contact point and hand contact point
        """
        self._data.qfrc_applied[:] = 0
        self._data.xfrc_applied[:] = 0
        contact_diffs = []
        for i, h_cb in enumerate(hand_cbody):
            b_id = self._model.body(self._cfg.hand_prefix + h_cb).id
            b_r = self._data.xmat[b_id].reshape(3, 3)
            b_t = self._data.xpos[b_id]
            h_cp_w = b_r @ hand_cpn_b[i, :3] + b_t
            o_cp_w = obj_cpn_w[i, :3]
            contact_diffs.append(np.linalg.norm(o_cp_w - h_cp_w))
            delta_normal = (
                np.sum((o_cp_w - h_cp_w) * obj_cpn_w[i, 3:], axis=-1, keepdims=True)
                * obj_cpn_w[i, 3:]
            )
            delta_tangent = o_cp_w - h_cp_w - delta_normal
            spring_force = delta_normal * 500 + delta_tangent * 100
            mujoco.mj_applyFT(
                self._model,
                self._data,
                spring_force,
                0 * spring_force,
                h_cp_w,
                b_id,
                self._data.qfrc_applied,
            )
        self.set_ctrl(self._data.qpos, skip_mocap=True)
        self._data.qvel[:] = 0
        return np_array32(contact_diffs)

    def keep_hand_stable(self) -> None:
        self._data.qfrc_applied[:] = 0
        self._data.xfrc_applied[:] = 0
        self.set_ctrl(self._data.qpos, skip_mocap=True)
        self._data.qvel[:] = 0
        return

    def get_squeeze_qpos(
        self,
        grasp_qpos: np.ndarray,
        hand_cbody: list[str],
        hand_cp_w: np.ndarray,
        cw: np.ndarray,
    ) -> np.ndarray:
        """
        Args:
            grasp_qpos: (7,) grasp qpos
            hand_cbody: list of hand contact bodies
            hand_cp_w: (N, 3) hand contact points in world frame
            cw: (N, 6) contact wrenches (force, torque) in world frame
        Returns:
            squeeze_qpos: (7,) squeeze qpos
        """
        self._data.qfrc_applied[:] = 0
        for h_cb, h_cp_w, h_cw in zip(hand_cbody, hand_cp_w, cw):
            mujoco.mj_applyFT(
                self._model,
                self._data,
                h_cw[:3],
                0 * h_cw[3:],
                h_cp_w[:3],
                self._model.body(self._cfg.hand_prefix + h_cb).id,
                self._data.qfrc_applied,
            )
        delta_qpos = np.copy(self._data.qfrc_applied)
        self._data.qfrc_applied[:] = 0
        actuator_gainprm = self._model.actuator_gainprm[:, 0]
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
class MuJoCo_EvalCfg(SimCfg):
    timestep: float = 0.004
    hand_mocap_solimp: list[float] = field(
        default_factory=lambda: [0.9, 0.95, 0.001, 0.5, 2]
    )
    hand_mocap_data: list[float] = field(
        default_factory=lambda: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    )


class MuJoCo_EvalEnv(MuJoCo_BaseEnv):
    def __init__(
        self,
        hand_cfg: HandCfg,
        scene_cfg: dict | None = None,
        sim_cfg: SimCfg = MuJoCo_EvalCfg(),
        debug_render: bool = False,
        debug_view: bool = True,
    ):
        super().__init__(hand_cfg, scene_cfg, sim_cfg, debug_render, debug_view)
        return

    def _set_friction(self, miu_coef):
        self._spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self._spec.option.noslip_iterations = 2
        self._spec.option.impratio = 10
        for g in self._spec.geoms:
            g.friction[:2] = miu_coef
            g.condim = 4
        return

    def eval_fc(
        self,
        ctrl_qpos,
        ctrl_type,
        ext_fdir,
        target_obj_pose,
        trans_thre,
        rot_thre,
    ) -> tuple[bool, None, None]:
        ext_fdir_lst = np.array(
            [[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        )

        for real_ext_fdir in ext_fdir_lst:
            self.reset_qpos(ctrl_qpos[0])
            self.run_trajectory(ctrl_qpos, ctrl_type, ext_fdir)
            self.set_active_obj_ext_fdir(real_ext_fdir)
            for _ in range(10):
                self.step_sim(substep=50)
                curr_obj_pose = self.get_active_obj_pose()
                delta_trans, delta_rot = np_get_delta_pose(
                    target_obj_pose, curr_obj_pose
                )
                succ_flag = (delta_trans < trans_thre) & (delta_rot < rot_thre)
                if not succ_flag:
                    break
            if not succ_flag:
                break
        return succ_flag, None, None

    def eval_move(
        self,
        ctrl_qpos,
        ctrl_type,
        ext_fdir,
        target_obj_pose,
        trans_thre,
        rot_thre,
    ) -> tuple[bool, np.ndarray, np.ndarray]:
        self.reset_qpos(ctrl_qpos[0])
        state_qpos, ctrl_qpos = self.run_trajectory(ctrl_qpos, ctrl_type, ext_fdir)
        self.step_sim(substep=50)
        curr_obj_pose = self.get_active_obj_pose()
        delta_trans, delta_rot = np_get_delta_pose(target_obj_pose, curr_obj_pose)
        succ_flag = (delta_trans < trans_thre) & (delta_rot < rot_thre)
        return succ_flag, state_qpos, ctrl_qpos


class MuJoCo_VisEnv(MuJoCo_BaseEnv):
    def __init__(
        self,
        hand_cfg: HandCfg,
        scene_cfg: dict | None = None,
        sim_cfg: SimCfg = MuJoCo_EvalCfg(),
        vis_mode: str | None = None,
        debug_render: bool = False,
        debug_view: bool = False,
    ):
        super().__init__(hand_cfg, scene_cfg, sim_cfg, debug_render, debug_view)
        assert vis_mode == "visual" or vis_mode == "collision" or vis_mode is None

        self._body_mesh_dict = {}
        self._body_id_dict = {}
        for i in range(self._model.ngeom):
            geom = self._model.geom(i)
            mesh_id = geom.dataid
            body_id = geom.bodyid[0]
            body_name = self._model.body(body_id).name
            self._body_id_dict[body_name] = body_id

            if geom.contype == 0 and vis_mode != "visual":
                continue
            if geom.contype != 0 and vis_mode != "collision":
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
                    tm = trimesh.creation.box(
                        extents=[2 * geom.size[-1], 2 * geom.size[-1], 0.001]
                    )
                else:
                    raise NotImplementedError(
                        f"Unsupported mujoco primitive type: {geom.type}. Available choices: 2(icosphere), 3(capsule), 5(cylinder), 6(box)."
                    )
            else:  # Meshes
                mjm = self._model.mesh(mesh_id)
                vert = self._model.mesh_vert[
                    mjm.vertadr[0] : mjm.vertadr[0] + mjm.vertnum[0]
                ]
                face = self._model.mesh_face[
                    mjm.faceadr[0] : mjm.faceadr[0] + mjm.facenum[0]
                ]
                tm = trimesh.Trimesh(vertices=vert, faces=face)

            if vis_mode == "collision":
                tm = tm.convex_hull

            geom_rot = self._data.geom_xmat[i].reshape(3, 3)
            geom_trans = self._data.geom_xpos[i]
            body_rot = self._data.xmat[body_id].reshape(3, 3)
            body_trans = self._data.xpos[body_id]
            tm.vertices = (
                tm.vertices @ geom_rot.T + geom_trans - body_trans
            ) @ body_rot

            if body_name not in self._body_mesh_dict:
                self._body_mesh_dict[body_name] = [tm]
            else:
                self._body_mesh_dict[body_name].append(tm)

        for body_name, body_mesh in self._body_mesh_dict.items():
            self._body_mesh_dict[body_name] = trimesh.util.concatenate(body_mesh)
        return

    def _set_friction(self, miu_coef):
        self._spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        for g in self._spec.geoms:
            g.condim = 1
        return

    def forward_kinematics(self, qpos) -> tuple[np.ndarray, np.ndarray]:
        self._data.qpos = qpos
        mujoco.mj_kinematics(self._model, self._data)
        xmat = np_array32(self._data.xmat).reshape(-1, 3, 3)
        xpos = np_array32(self._data.xpos)
        return xmat, xpos

    def get_prefix(self, body_type="all") -> list[str]:
        if body_type == "all":
            return [
                self._cfg.hand_prefix,
                self._cfg.obj_prefix,
                self._cfg.plane_prefix,
            ]
        elif body_type == "hand":
            return [self._cfg.hand_prefix]
        elif body_type == "obj":
            return [self._cfg.obj_prefix, self._cfg.plane_prefix]
        else:
            raise NotImplementedError(
                f"Unsupported mesh type: {body_type}. Available choices: ['all', 'hand', 'obj']"
            )

    def get_all_body_names(self) -> list[str]:
        return list(self._body_mesh_dict.keys())

    def get_body_id(self, body_name: str) -> int:
        if body_name in self._body_id_dict:
            return self._body_id_dict[body_name]
        for p in self.get_prefix("all"):
            if p + body_name in self._body_id_dict:
                return self._body_id_dict[p + body_name]
        raise ValueError(f"Body name {body_name} not found")

    def get_body_mesh(self, body_name: str) -> trimesh.Trimesh:
        if body_name in self._body_mesh_dict:
            return self._body_mesh_dict[body_name]
        for p in self.get_prefix("all"):
            if p + body_name in self._body_mesh_dict:
                return self._body_mesh_dict[p + body_name]
        raise ValueError(f"Body name {body_name} not found")

    def get_posed_mesh(self, xmat, xpos, body_type="all") -> trimesh.Trimesh:
        vis_prefix = self.get_prefix(body_type)
        full_tm = []
        for k, v in self._body_mesh_dict.items():
            if not any(k.startswith(p) for p in vis_prefix):
                continue
            body_rot = xmat[self._body_id_dict[k]].reshape(3, 3)
            body_trans = xpos[self._body_id_dict[k]]
            posed_vert = v.vertices @ body_rot.T + body_trans
            posed_tm = trimesh.Trimesh(vertices=posed_vert, faces=v.faces)
            full_tm.append(posed_tm)
        full_tm = trimesh.util.concatenate(full_tm)
        return full_tm


if __name__ == "__main__":
    xml_path = os.path.join(
        os.path.dirname(__file__), "../../assets/hand/shadow/right.xml"
    )
    kinematic = MuJoCo_VisEnv(hand_cfg=HandCfg(xml_path=xml_path), vis_mode="visual")
    hand_qpos = np.zeros((22))
    xmat, xpos = kinematic.forward_kinematics(hand_qpos)
    visual_mesh = kinematic.get_posed_mesh(xmat, xpos)
    visual_mesh.export(f"debug_hand.obj")
