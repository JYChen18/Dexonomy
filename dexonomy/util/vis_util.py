import trimesh
import numpy as np

from dexonomy.util.np_util import np_normalize_vector, np_normal_to_rot


def get_point_mesh(points: np.ndarray) -> trimesh.Trimesh:
    """
    Args:
        points: (N, 3) points
    Returns:
        point_mesh: point mesh
    """
    point_mesh_lst = []
    for i in range(points.shape[0]):
        tm_sph = trimesh.creation.icosphere(radius=0.003)
        tm_sph.apply_translation(points[i])
        point_mesh_lst.append(tm_sph)
    point_mesh = trimesh.util.concatenate(point_mesh_lst)
    return point_mesh


def get_arrow_mesh(point_normal: np.ndarray, length: float = 0.02) -> trimesh.Trimesh:
    """
    Args:
        point_normal: (N, 6) points and normals
        length: length of the arrow
    Returns:
        arrow_mesh: arrow mesh
    """
    points, normals = point_normal[:, :3], point_normal[:, 3:]
    arrow_mesh_lst = []
    line = np.stack([points, points + length * normals], axis=1)
    for i in range(line.shape[0]):
        arrow_mesh_lst.append(trimesh.creation.cylinder(radius=0.001, segment=line[i]))
        cone_trans = np.eye(4)
        tmp = np_normal_to_rot(np_normalize_vector(line[i][1:] - line[i][0:1]))
        cone_trans[:3, :3] = np.stack(
            [tmp[0, :, 1], tmp[0, :, 2], tmp[0, :, 0]], axis=-1
        )
        cone_trans[:3, 3] = line[i][1]
        arrow_mesh_lst.append(
            trimesh.creation.cone(radius=0.003, height=0.01, transform=cone_trans)
        )
    arrow_mesh = trimesh.util.concatenate([get_point_mesh(points), arrow_mesh_lst])
    return arrow_mesh


def get_line_mesh(lines: np.ndarray) -> trimesh.Trimesh:
    """
    Args:
        lines: (N, 2, 3) lines
    Returns:
        line_mesh: line mesh
    """
    line_mesh_lst = []
    for i in range(lines.shape[0]):
        line_mesh_lst.append(trimesh.creation.cylinder(radius=0.001, segment=lines[i]))
    line_mesh = trimesh.util.concatenate(line_mesh_lst)
    return line_mesh
