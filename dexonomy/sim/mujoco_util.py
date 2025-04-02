import os
import pdb
from typing import List, Dict
import logging

os.environ["MUJOCO_GL"] = "glfw"

import imageio
import trimesh
import numpy as np
import mujoco
import mujoco.viewer
import transforms3d.quaternions as tq

from dexonomy.util.np_rot_util import (
    np_interplote_pose,
    np_interplote_qpos,
    np_array32,
    np_transform_points,
    np_inv_transform_points,
    np_get_delta_pose,
)


class MuJoCo_BaseEnv:

    timestep: float
    obj_freejoint: bool
    hand_mocap_solimp: List[float]
    hand_mocap_data: List[float]
    obj_margin: float
    hand_margin: float
    plane_margin: float
    hand_prefix: str = "child-"
    rigid_obj_num: int = 0
    plane_num: int = 0

    def __init__(
        self,
        hand_xml_path: str,
        hand_add_mocap: bool,
        hand_exclude_table_contact: List[str],
        friction_coef: tuple[float, float] = None,
        obj_cfg_lst: List[Dict] = None,
        rigid_obj_interest_id: int = 0,
        debug_render: bool = False,
        debug_viewer: bool = False,
    ):
        self.hand_add_mocap = hand_add_mocap
        self.spec = mujoco.MjSpec()
        self.spec.meshdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.spec.option.timestep = self.timestep
        self.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_GRAVITY
        self.spec.option.enableflags = mujoco.mjtEnableBit.mjENBL_NATIVECCD

        if debug_render or debug_viewer:
            self._add_for_visualization()

        self._add_hand(hand_xml_path, hand_add_mocap, hand_exclude_table_contact)

        self.rigid_obj_init_pose = []
        self.rigid_obj_interest_id = rigid_obj_interest_id
        for obj_cfg in obj_cfg_lst:
            obj_type = obj_cfg.pop("type")
            if obj_type == "plane":
                self._add_plane(**obj_cfg)
            elif obj_type == "rigid":
                self._add_rigid_object(**obj_cfg)
            else:
                raise NotImplementedError
        self.rigid_obj_init_pose = np_array32(self.rigid_obj_init_pose).reshape(-1)

        self._set_friction(friction_coef)
        self.spec.add_key()

        # Get ready for simulation
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        # For ctrl
        qpos2ctrl_matrix = np.zeros((self.model.nu, self.model.nv))
        mujoco.mju_sparse2dense(
            qpos2ctrl_matrix,
            self.data.actuator_moment,
            self.data.moment_rownnz,
            self.data.moment_rowadr,
            self.data.moment_colind,
        )
        self.hand_nv = self.model.nv - 6 * self.rigid_obj_num * self.obj_freejoint
        self._qpos2ctrl_matrix = qpos2ctrl_matrix[..., : self.hand_nv]
        self.hand_nq = self.model.nq - 7 * self.rigid_obj_num * self.obj_freejoint

        self.debug_viewer = None
        self.debug_render = None
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
        add_mocap: bool,
        hand_exclude_table_contact: bool,
    ):
        # Read hand xml
        child_spec = mujoco.MjSpec.from_file(xml_path)
        for m in child_spec.meshes:
            m.file = os.path.join(os.path.dirname(xml_path), child_spec.meshdir, m.file)
        child_spec.meshdir = self.spec.meshdir

        for g in child_spec.geoms:
            # This solimp and solref comes from the Shadow Hand xml
            # The body will be more "rigid" and less "soft"
            g.solimp[:3] = [0.5, 0.99, 0.0001]
            g.solref[:2] = [0.005, 1]
            g.margin = self.hand_margin

        attach_frame = self.spec.worldbody.add_frame()
        child_world = attach_frame.attach_body(
            child_spec.worldbody, self.hand_prefix, ""
        )
        # Add freejoint and mocap of hand root
        if add_mocap:
            child_world.add_freejoint(name="hand_freejoint")
            self.spec.worldbody.add_body(name="mocap_body", mocap=True)
            self.spec.add_equality(
                type=mujoco.mjtEq.mjEQ_WELD,
                name1="mocap_body",
                name2=f"{self.hand_prefix}world",
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                solimp=self.hand_mocap_solimp,
                data=self.hand_mocap_data,
            )
        if hand_exclude_table_contact is not None:
            for body_name in hand_exclude_table_contact:
                self.spec.add_exclude(
                    bodyname1="world", bodyname2=f"{self.hand_prefix}{body_name}"
                )
        return

    def _add_rigid_object(self, mesh_dir, pose, scale, density=1000):
        obj_body = self.spec.worldbody.add_body(
            name=f"rigid_object_{self.rigid_obj_num}",
            pos=pose[:3],
            quat=pose[3:],
        )
        if self.obj_freejoint:
            obj_body.add_freejoint(name=f"rigid_object_freejoint_{self.rigid_obj_num}")

        for file in os.listdir(mesh_dir):
            file_name = file.replace(".obj", "")
            cvpart_id = file_name.replace("convex_piece_", "")
            mesh_name = f"rigid_{self.rigid_obj_num}_{file_name}"

            self.spec.add_mesh(
                name=mesh_name,
                file=os.path.join(mesh_dir, file),
                scale=[scale, scale, scale],
            )
            obj_body.add_geom(
                name=f"rigid_object_visual_{self.rigid_obj_num}_{cvpart_id}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                density=0,
                contype=0,
                conaffinity=0,
            )
            obj_body.add_geom(
                name=f"rigid_object_collision_{self.rigid_obj_num}_{cvpart_id}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                density=density,
                margin=self.obj_margin,
            )
        self.rigid_obj_num += 1
        self.rigid_obj_init_pose.append(pose)
        return

    def _add_articulated_object(self, urdf_path, pose, scale):
        raise NotImplementedError

    def _add_plane(self, pose=[0.0, 0, 0], size=[0, 0, 1.0]):
        plane_geom = self.spec.worldbody.add_geom(
            name=f"plane_collision_{self.plane_num}",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=pose,
            size=size,
            margin=self.plane_margin,
        )
        self.plane_num += 1
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
            name="closeup", pos=[0.0, 1.0, 1.0], xyaxes=[-1, 0, 0, 0, -1, 1]
        )

    def _set_friction(self, friction_coef: tuple[float, float]):
        raise NotImplementedError

    def _qpos2ctrl(self, hand_qpos):
        if self.hand_add_mocap:
            return self._qpos2ctrl_matrix[:, 6:] @ hand_qpos[7:]
        else:
            return self._qpos2ctrl_matrix @ hand_qpos

    def get_interest_rigid_object_pose(self):
        return self.data.qpos[
            -7
            * self.obj_freejoint
            * (self.rigid_obj_num - self.rigid_obj_interest_id) :
        ]

    def get_hand_qpos(self):
        return self.data.qpos[: self.hand_nq]

    def get_contact_info(self, obj_margin=None):
        if obj_margin is not None:
            self.set_rigid_object_margin(obj_margin)
            mujoco.mj_forward(self.model, self.data)

        object_id = self.model.nbody - self.rigid_obj_num
        hand_id = self.model.nbody - self.rigid_obj_num - 1
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
                body1_id > world_id and body1_id < hand_id and body2_id >= object_id
            ) or (body2_id > world_id and body2_id < hand_id and body1_id >= object_id):
                # keep body1=hand and body2=object
                if body2_id >= object_id:
                    contact_normal = contact.frame[0:3]
                    hand_body_name = body1_name.removeprefix(self.hand_prefix)
                    obj_body_name = body2_name
                else:
                    contact_normal = -contact.frame[0:3]
                    hand_body_name = body2_name.removeprefix(self.hand_prefix)
                    obj_body_name = body1_name
                ho_contact.append(
                    {
                        "contact_dist": contact.dist,
                        "contact_pos": contact.pos - contact.dist * contact_normal,
                        "contact_normal": contact_normal,
                        "body1_name": hand_body_name,
                        "body2_name": obj_body_name,
                    }
                )
            # hand and hand
            elif (
                body1_id > world_id
                and body1_id < hand_id
                and body2_id > world_id
                and body2_id < hand_id
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
            self.set_rigid_object_margin(self.obj_margin)
        return ho_contact, hh_contact

    def set_rigid_object_margin(self, obj_margin):
        for i in range(self.model.ngeom):
            if "rigid_object_collision" in self.model.geom(i).name:
                self.model.geom_margin[i] = obj_margin
        return

    def set_rigid_object_extforce(self, ext_force):
        self.data.xfrc_applied[-self.rigid_obj_num :] = ext_force
        return

    def reset_qpos(self, hand_qpos):
        # set key frame
        if self.obj_freejoint:
            self.model.key_qpos[0] = np.concatenate(
                [hand_qpos, self.rigid_obj_init_pose], axis=0
            )
        else:
            self.model.key_qpos[0] = hand_qpos

        self.model.key_ctrl[0] = self._qpos2ctrl(hand_qpos)
        self.model.key_qvel[0] = 0
        self.model.key_act[0] = 0
        if self.hand_add_mocap:
            self.model.key_mpos[0] = hand_qpos[:3]
            self.model.key_mquat[0] = hand_qpos[3:7]

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        return

    def control_hand_with_interp(
        self, hand_qpos1, hand_qpos2, step_outer=10, step_inner=10
    ):
        if self.hand_add_mocap:
            pose_interp = np_interplote_pose(hand_qpos1[:7], hand_qpos2[:7], step_outer)
        qpos_interp = np_interplote_qpos(
            self._qpos2ctrl(hand_qpos1), self._qpos2ctrl(hand_qpos2), step_outer
        )
        for j in range(step_outer):
            if self.hand_add_mocap:
                self.data.mocap_pos[0] = pose_interp[j, :3]
                self.data.mocap_quat[0] = pose_interp[j, 3:7]
            self.data.ctrl[:] = qpos_interp[j]
            self.control_hand_step(step_inner)
        return

    def control_hand_step(self, step_inner):
        mujoco.mj_forward(self.model, self.data)
        for _ in range(step_inner):
            mujoco.mj_step(self.model, self.data)

        if self.debug_render is not None:
            self.debug_render.update_scene(self.data, "closeup", self.debug_options)
            pixels = self.debug_render.render()
            self.debug_images.append(pixels)

        if self.debug_viewer is not None:
            self.debug_viewer.sync()
            # pdb.set_trace()
        return

    def debug_postprocess(self, save_path=None):
        if self.debug_render is not None:
            assert save_path is not None
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.mimsave(save_path, self.debug_images)
            logging.info(f"Save GIF to {save_path}")

        if self.debug_viewer is not None:
            self.debug_viewer.close()


class MuJoCo_OptEnv(MuJoCo_BaseEnv):
    timestep = 0.002
    obj_freejoint = False
    hand_mocap_solimp = [0.01, 0.095, 1, 0.5, 2]  # looser constraints
    hand_mocap_data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10]  # looser constraints
    obj_margin = 0.001
    hand_margin = 0.001
    plane_margin = 0.02

    def _set_friction(self, friction_coef):
        self.spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        for g in self.spec.geoms:
            g.condim = 1
        return

    def get_hand_worldframe_contact(self, hand_body_name, hand_bodyframe_contact):
        hand_worldframe_contact = []
        for body_name, hf_c in zip(hand_body_name, hand_bodyframe_contact):
            body_id = self.model.body(self.hand_prefix + body_name).id
            br = self.data.xmat[body_id].reshape(3, 3)
            bp = self.data.xpos[body_id]
            hand_worldframe_contact.append(np_transform_points(hf_c, br, bp))
        return np_array32(hand_worldframe_contact)

    def get_hand_bodyframe_contact(self, hand_body_name, hand_worldframe_contact):
        hand_bodyframe_contact = []
        for body_name, wf_c in zip(hand_body_name, hand_worldframe_contact):
            body_id = self.model.body(self.hand_prefix + body_name).id
            br = self.data.xmat[body_id].reshape(3, 3)
            bp = self.data.xpos[body_id]
            hand_bodyframe_contact.append(np_inv_transform_points(wf_c, br, bp))
        return np_array32(hand_bodyframe_contact)

    def apply_force_on_hand(
        self, hand_body_name, hand_bodyframe_contact, obj_worldframe_contact
    ):
        self.data.qfrc_applied[:] = 0
        self.data.xfrc_applied[:] = 0
        total_loss = []
        for i, body_name in enumerate(hand_body_name):
            body_id = self.model.body(self.hand_prefix + body_name).id
            br = self.data.xmat[body_id].reshape(3, 3)
            bp = self.data.xpos[body_id]
            hcp_world = br @ hand_bodyframe_contact[i, :3] + bp
            ocp_world = obj_worldframe_contact[i, :3]
            total_loss.append(np.linalg.norm(ocp_world - hcp_world))
            point_force = (ocp_world - hcp_world) * 500
            mujoco.mj_applyFT(
                self.model,
                self.data,
                point_force,
                0 * point_force,
                hcp_world,
                body_id,
                self.data.qfrc_applied,
            )
        self.data.ctrl = self._qpos2ctrl(self.data.qpos)
        self.data.qvel[:] = 0
        return np_array32(total_loss)

    def keep_hand_stable(self):
        self.data.qfrc_applied[:] = 0
        self.data.xfrc_applied[:] = 0
        self.data.ctrl = self._qpos2ctrl(self.data.qpos)
        self.data.qvel[:] = 0
        return

    def get_squeeze_qpos(
        self, grasp_qpos, hand_body_name, hand_worldframe_contact, contact_wrench
    ):
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
                self.model.body(self.hand_prefix + hbn).id,
                self.data.qfrc_applied,
            )
        delta_qpos = np.copy(self.data.qfrc_applied)
        actuator_gainprm = self.model.actuator_gainprm[:, 0]
        for i in range(len(delta_qpos)):
            actuator_id = np.where(self._qpos2ctrl_matrix[:, i] != 0)[0]
            if len(actuator_id) > 0:
                delta_qpos[i] /= (
                    actuator_gainprm[actuator_id[0]]
                    * self._qpos2ctrl_matrix[actuator_id[0]].sum()
                )
        squeeze_qpos = np.concatenate([grasp_qpos[:7], grasp_qpos[7:] + delta_qpos[6:]])
        return squeeze_qpos


class MuJoCo_TestEnv(MuJoCo_BaseEnv):
    timestep = 0.004
    obj_freejoint = True
    hand_mocap_solimp = [0.9, 0.95, 0.001, 0.5, 2]
    hand_mocap_data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    obj_margin = 0.0
    hand_margin = 0.0
    plane_margin = 0.0

    def _set_friction(self, friction_coef):
        self.spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.spec.option.noslip_iterations = 2
        self.spec.option.impratio = 10
        for g in self.spec.geoms:
            g.friction[:2] = friction_coef
            g.condim = 4
        return

    def test_mocap_notable(
        self, grasp_qpos, squeeze_qpos, extforce, trans_thre, angle_thre
    ):
        self.reset_qpos(grasp_qpos)
        pre_obj_pose = self.get_interest_rigid_object_pose()
        self.control_hand_with_interp(grasp_qpos, squeeze_qpos)
        self.set_rigid_object_extforce(extforce)
        for _ in range(10):
            self.control_hand_step(step_inner=50)
            latter_obj_pose = self.get_interest_rigid_object_pose()
            delta_pos, delta_angle = np_get_delta_pose(pre_obj_pose, latter_obj_pose)
            succ_flag = (delta_pos < trans_thre) & (delta_angle < angle_thre)
            if not succ_flag:
                break
        return

    def test_arm(self):
        raise NotImplementedError


class MuJoCo_RobotFK:
    def __init__(self, xml_path, vis_mesh_mode=None):
        assert (
            vis_mesh_mode == "visual"
            or vis_mesh_mode == "collision"
            or vis_mesh_mode is None
        )
        spec = mujoco.MjSpec.from_file(xml_path)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        mujoco.mj_forward(self.model, self.data)

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
                else:
                    raise NotImplementedError
            else:  # Meshes
                mjm = self.model.mesh(mesh_id)
                vert = self.model.mesh_vert[
                    mjm.vertadr[0] : mjm.vertadr[0] + mjm.vertnum[0]
                ]
                face = self.model.mesh_face[
                    mjm.faceadr[0] : mjm.faceadr[0] + mjm.facenum[0]
                ]
                tm = trimesh.Trimesh(vertices=vert, faces=face)

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

    def forward_kinematics(self, qpos, pose=None):
        """
        pose: [7]. Robot root pose.
        qpos: [n]. n joint angles.
        """
        self.data.qpos = qpos
        mujoco.mj_kinematics(self.model, self.data)
        xmat = np_array32(self.data.xmat).reshape(-1, 3, 3)
        xpos = np_array32(self.data.xpos)
        if pose is not None:
            root_r, root_t = tq.quat2mat(pose[3:]).astype(np.float32), np_array32(
                pose[:3]
            )
            xmat = root_r[None] @ xmat
            xpos = xpos @ root_r.T + root_t[None]
        return xmat, xpos

    def get_init_meshes(self):
        return self.body_mesh_dict.keys(), self.body_mesh_dict.values()

    def get_posed_meshes(self, xmat, xpos):
        full_tm = []
        for k, v in self.body_mesh_dict.items():
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
    kinematic = MuJoCo_RobotFK(xml_path)
    hand_qpos = np.zeros((22))
    kinematic.forward_kinematics(hand_qpos)
    visual_mesh = kinematic.get_posed_meshes()
    visual_mesh.export(f"debug_hand.obj")
