import mujoco
import numpy as np
from copy import deepcopy
import trimesh
from transforms3d import quaternions as tq

from dexonomy.util.file_util import load_yaml
from dexonomy.util.np_rot_util import np_normalize_vector


class RobotKinematics:
    def __init__(self, xml_path, vis_mesh_mode="visual"):
        assert vis_mesh_mode == "visual" or vis_mesh_mode == "collision"
        spec = mujoco.MjSpec.from_file(xml_path)
        self.mj_model = spec.compile()
        self.mj_data = mujoco.MjData(self.mj_model)
        return

    def forward(self, kp_data):
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.body_mesh_dict = {}
        self.body_id_dict = {}
        self.body_transform_dict = {}
        for i in range(self.mj_model.ngeom):
            geom = self.mj_model.geom(i)
            mesh_id = geom.dataid
            body_name = self.mj_model.body(geom.bodyid).name
            self.body_id_dict[body_name] = geom.bodyid[0]

            if geom.contype != 0:
                continue

            if mesh_id == -1:  # Primitives
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
            body_rot = self.mj_data.xmat[geom.bodyid].reshape(3, 3)
            body_trans = self.mj_data.xpos[geom.bodyid].reshape(3)
            if body_name not in self.body_transform_dict:
                print(body_rot.shape, geom_trans.shape, body_trans.shape)
                self.body_transform_dict[body_name] = {
                    "rot": body_rot.T @ geom_rot,
                    "trans": body_rot.T @ (geom_trans - body_trans),
                }
            else:
                print("???", body_name)

        for k, v in kp_data.items():
            if isinstance(v, str):
                continue
            rot = self.body_transform_dict[k]["rot"]
            trans = self.body_transform_dict[k]["trans"]
            kp = np.array(v)
            print(k, kp, kp.shape)
            kp_data[k] = np.concatenate(
                [kp[:, :3] @ rot.T + trans, kp[:, 3:] @ rot.T],
                axis=-1,
            )
        return kp_data


if __name__ == "__main__":

    kp_path = "/mnt/disk1/jiayichen/code/Dexonomy/assets/hand/shadow/anno_keypoint.yaml"
    kp_data = load_yaml(kp_path)

    kin = RobotKinematics(
        xml_path="/mnt/disk1/jiayichen/code/Dexonomy/assets/hand/shadow/right.xml"
    )
    new_data = kin.forward(kp_data)
    np.set_printoptions(precision=6, suppress=True)
    print(new_data)
