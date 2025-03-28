import trimesh
import numpy as np

from dexonomy.util.np_rot_util import np_normalize_vector, np_normal_to_rot


def get_arrow_mesh(point_lst, normal_lst, length=0.02):
    point_mesh_lst = []
    arrow_mesh_lst = []
    line = np.stack([point_lst, point_lst + length * normal_lst], axis=1)
    for i in range(line.shape[0]):
        point_mesh_lst.append(trimesh.creation.cylinder(radius=0.001, segment=line[i]))
        cone_trans = np.eye(4)
        tmp = np_normal_to_rot(np_normalize_vector(line[i][1:] - line[i][0:1]))
        cone_trans[:3, :3] = np.stack(
            [tmp[0, :, 1], tmp[0, :, 2], tmp[0, :, 0]], axis=-1
        )
        cone_trans[:3, 3] = line[i][1]
        point_mesh_lst.append(
            trimesh.creation.cone(radius=0.003, height=0.01, transform=cone_trans)
        )
        tm_sph = trimesh.creation.icosphere(radius=0.003)
        tm_sph.apply_translation(point_lst[i])
        arrow_mesh_lst.append(tm_sph)

    point_mesh = trimesh.util.concatenate(point_mesh_lst)
    arrow_mesh = trimesh.util.concatenate(arrow_mesh_lst)

    return point_mesh, arrow_mesh
