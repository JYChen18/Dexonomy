import os
import pdb
from copy import deepcopy

os.environ["MUJOCO_GL"] = "osmesa"

import trimesh
import numpy as np
import mujoco
import mujoco.viewer
import transforms3d.quaternions as tq

from dexonomy.util.np_rot_util import np_interplote_pose, np_interplote_qpos, np_array32


class MjHO:

    hand_prefix: str = "child-"

    def __init__(
        self,
        obj_path,
        obj_scale,
        obj_density,
        hand_xml_path,
        hand_mocap,
        exclude_table_contact,
        friction_coef,
        has_floor_z0,
        debug_render=False,
        debug_viewer=False,
    ):
        self.hand_mocap = hand_mocap
        self.spec = mujoco.MjSpec()
        self.spec.meshdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.spec.option.timestep = 0.004
        self.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.spec.option.disableflags = mujoco.mjtDisableBit.mjDSBL_GRAVITY
        self.spec.option.enableflags = mujoco.mjtEnableBit.mjENBL_NATIVECCD
        if debug_render or debug_viewer:
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
                name="closeup", pos=[0.75, 1.0, 1.0], xyaxes=[-1, 0, 0, 0, -1, 1]
            )

        self._add_hand(hand_xml_path, hand_mocap)
        self._add_object(obj_path, obj_scale, obj_density, has_floor_z0)
        self._set_friction(friction_coef)
        self.spec.add_key()
        if exclude_table_contact is not None:
            for body_name in exclude_table_contact:
                self.spec.add_exclude(
                    bodyname1="world", bodyname2=f"{self.hand_prefix}{body_name}"
                )

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
        self._qpos2ctrl_matrix = qpos2ctrl_matrix[..., :-6]

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

    def _add_hand(self, xml_path, mocap_base):
        # Read hand xml
        child_spec = mujoco.MjSpec.from_file(xml_path)
        for m in child_spec.meshes:
            m.file = os.path.join(os.path.dirname(xml_path), child_spec.meshdir, m.file)
        child_spec.meshdir = self.spec.meshdir

        for g in child_spec.geoms:
            # This solimp and solref comes from the Shadow Hand xml
            # They can generate larger force with smaller penetration
            # The body will be more "rigid" and less "soft"
            g.solimp[:3] = [0.5, 0.99, 0.0001]
            g.solref[:2] = [0.005, 1]

        attach_frame = self.spec.worldbody.add_frame()
        child_world = attach_frame.attach_body(
            child_spec.worldbody, self.hand_prefix, ""
        )
        # Add freejoint and mocap of hand root
        if mocap_base:
            child_world.add_freejoint(name="hand_freejoint")
            self.spec.worldbody.add_body(name="mocap_body", mocap=True)
            self.spec.add_equality(
                type=mujoco.mjtEq.mjEQ_WELD,
                name1="mocap_body",
                name2=f"{self.hand_prefix}world",
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                solimp=[0.9, 0.95, 0.001, 0.5, 2],
                data=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            )
        return

    def _add_object(self, obj_path, obj_scale, obj_density, has_floor_z0):
        if has_floor_z0:
            floor_geom = self.spec.worldbody.add_geom(
                name="object_collision_floor",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                pos=[0, 0, 0],
                size=[0, 0, 1.0],
            )

        obj_body = self.spec.worldbody.add_body(name="object")
        obj_body.add_freejoint(name="obj_freejoint")
        parts_folder = os.path.join(obj_path, "urdf/meshes")
        for file in os.listdir(parts_folder):
            file_path = os.path.join(parts_folder, file)
            mesh_name = file.replace(".obj", "")
            mesh_id = mesh_name.replace("convex_piece_", "")

            self.spec.add_mesh(
                name=mesh_name,
                file=file_path,
                scale=[obj_scale, obj_scale, obj_scale],
            )
            obj_body.add_geom(
                name=f"object_visual_{mesh_id}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                density=0,
                contype=0,
                conaffinity=0,
            )
            obj_body.add_geom(
                name=f"object_collision_{mesh_id}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                density=obj_density,
            )

        return

    def _set_friction(self, test_friction):
        self.spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.spec.option.noslip_iterations = 2
        self.spec.option.impratio = 10
        for g in self.spec.geoms:
            g.friction[:2] = test_friction
            g.condim = 4
        return

    def _qpos2ctrl(self, hand_qpos):
        if self.hand_mocap:
            return self._qpos2ctrl_matrix[:, 6:] @ hand_qpos[7:]
        else:
            return self._qpos2ctrl_matrix @ hand_qpos

    def get_obj_pose(self):
        return self.data.qpos[-7:]

    def get_contact_info(self, hand_qpos, obj_pose, obj_margin=0):
        # Set margin and gap to detect contact
        for i in range(self.model.ngeom):
            if "object_collision" in self.model.geom(i).name:
                self.model.geom_margin[i] = self.model.geom_gap[i] = obj_margin

        # Set pose and qpos for hand and object
        self.reset_pose_qpos(hand_qpos, obj_pose)

        object_id = self.model.nbody - 1
        hand_id = self.model.nbody - 2
        world_id = -1 if self.hand_mocap else 0

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
                body1_id > world_id and body1_id < hand_id and body2_id == object_id
            ) or (body2_id > world_id and body2_id < hand_id and body1_id == object_id):
                # keep body1=hand and body2=object
                if body2_id == object_id:
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
                        "contact_pos": contact.pos,
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
        for i in range(self.model.ngeom):
            if "object_collision" in self.model.geom(i).name:
                self.model.geom_margin[i] = self.model.geom_gap[i] = 0
        return ho_contact, hh_contact

    def set_ext_force_on_obj(self, ext_force):
        self.data.xfrc_applied[-1] = ext_force
        return

    def reset_pose_qpos(self, hand_qpos, obj_pose):
        # set key frame
        self.model.key_qpos[0] = np.concatenate([hand_qpos, obj_pose], axis=0)
        self.model.key_ctrl[0] = self._qpos2ctrl(hand_qpos)
        self.model.key_qvel[0] = 0
        self.model.key_act[0] = 0
        if self.hand_mocap:
            self.model.key_mpos[0] = hand_qpos[:3]
            self.model.key_mquat[0] = hand_qpos[3:7]

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        return

    def control_hand_with_interp(
        self, hand_qpos1, hand_qpos2, step_outer=10, step_inner=10
    ):
        if self.hand_mocap:
            pose_interp = np_interplote_pose(hand_qpos1[:7], hand_qpos2[:7], step_outer)
        qpos_interp = np_interplote_qpos(
            self._qpos2ctrl(hand_qpos1), self._qpos2ctrl(hand_qpos2), step_outer
        )
        for j in range(step_outer):
            if self.hand_mocap:
                self.data.mocap_pos[0] = pose_interp[j, :3]
                self.data.mocap_quat[0] = pose_interp[j, 3:7]
            self.data.ctrl[:] = qpos_interp[j]
            mujoco.mj_forward(self.model, self.data)
            self.control_hand_step(step_inner)
        return

    def control_hand_step(self, step_inner):
        for _ in range(step_inner):
            mujoco.mj_step(self.model, self.data)

        if self.debug_render is not None:
            self.debug_render.update_scene(self.data, "closeup", self.debug_options)
            pixels = self.debug_render.render()
            self.debug_images.append(pixels)

        if self.debug_viewer is not None:
            raise NotImplementedError
        return


class RobotKinematics:
    def __init__(self, xml_path, vis_mesh_mode=None):
        assert (
            vis_mesh_mode == "visual"
            or vis_mesh_mode == "collision"
            or vis_mesh_mode is None
        )
        spec = mujoco.MjSpec.from_file(xml_path)
        self.mj_model = spec.compile()
        self.mj_data = mujoco.MjData(self.mj_model)

        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.body_mesh_dict = {}
        self.body_id_dict = {}
        for i in range(self.mj_model.ngeom):
            geom = self.mj_model.geom(i)
            mesh_id = geom.dataid
            body_id = geom.bodyid[0]
            body_name = self.mj_model.body(body_id).name
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
                mjm = self.mj_model.mesh(mesh_id)
                vert = self.mj_model.mesh_vert[
                    mjm.vertadr[0] : mjm.vertadr[0] + mjm.vertnum[0]
                ]
                face = self.mj_model.mesh_face[
                    mjm.faceadr[0] : mjm.faceadr[0] + mjm.facenum[0]
                ]
                tm = trimesh.Trimesh(vertices=vert, faces=face)

            geom_rot = self.mj_data.geom_xmat[i].reshape(3, 3)
            geom_trans = self.mj_data.geom_xpos[i]
            body_rot = self.mj_data.xmat[body_id].reshape(3, 3)
            body_trans = self.mj_data.xpos[body_id]
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
        self.mj_data.qpos = qpos
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        xmat = np_array32(self.mj_data.xmat).reshape(-1, 3, 3)
        xpos = np_array32(self.mj_data.xpos)
        if pose is not None:
            root_r, root_t = tq.quat2mat(pose[3:]), np_array32(pose[:3])
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
        os.path.dirname(__file__), "../../assets/hand/shadow/customized.xml"
    )
    kinematic = RobotKinematics(xml_path)
    hand_qpos = np.zeros((22))
    kinematic.forward_kinematics(hand_qpos)
    visual_mesh = kinematic.get_posed_meshes()
    visual_mesh.export(f"debug_hand.obj")
