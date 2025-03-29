from typing import List
import numpy as np
import transforms3d.quaternions as tq
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np


def np_array32(x: np.ndarray) -> np.ndarray:
    return np.array(x, dtype=np.float32)


def np_normalize_vector(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def np_interplote_pose(pose1: np.ndarray, pose2: np.ndarray, step: int) -> np.ndarray:
    trans1, quat1 = pose1[:3], pose1[3:7]
    trans2, quat2 = pose2[:3], pose2[3:7]
    slerp = Slerp([0, 1], R.from_quat([quat1, quat2], scalar_first=True))
    trans_interp = np.linspace(trans1, trans2, step + 1)[1:]
    quat_interp = slerp(np.linspace(0, 1, step + 1))[1:].as_quat(scalar_first=True)
    return np.concatenate([trans_interp, quat_interp], axis=1)


def np_interplote_qpos(qpos1: np.ndarray, qpos2: np.ndarray, step: int) -> np.ndarray:
    return np.linspace(qpos1, qpos2, step + 1)[1:]


def np_normal_to_rot(
    axis_0, rot_base1=np_array32([[0, 1, 0]]), rot_base2=np_array32([[0, 0, 1]])
):
    proj_xy = np.abs(np.sum(axis_0 * rot_base1, axis=-1, keepdims=True))
    axis_1 = np.where(proj_xy > 0.99, rot_base2, rot_base1)

    axis_1 = np_normalize_vector(
        axis_1 - np.sum(axis_1 * axis_0, axis=-1, keepdims=True) * axis_0
    )
    axis_2 = np.cross(axis_0, axis_1, axis=-1)

    return np.stack([axis_0, axis_1, axis_2], axis=-1)


def np_transform_points(points, rot, trans=0.0, scale=1):
    if isinstance(points, List):
        points = np_array32(points)
    if len(points.shape) == 1:
        unsqueeze_flag = True
        points = points[None]
    else:
        unsqueeze_flag = False

    assert len(points.shape) == 2 and (points.shape[-1] == 6 or points.shape[-1] == 3)
    assert len(rot.shape) == 2
    assert isinstance(trans, float) or (len(trans.shape) <= 2 and trans.shape[-1] == 3)

    if points.shape[-1] == 6:
        resulted_points = np.concatenate(
            [
                (points[..., :3] @ rot.T) * scale + trans,
                points[..., 3:] @ rot.T,
            ],
            axis=-1,
        )
    elif points.shape[-1] == 3:
        resulted_points = (points @ rot.T) * scale + trans

    if unsqueeze_flag:
        resulted_points = resulted_points.squeeze(0)
    return resulted_points


def np_inv_transform_points(points, rot, trans=0.0, scale=1):
    if isinstance(points, List):
        points = np_array32(points)
    if len(points.shape) == 1:
        unsqueeze_flag = True
        points = points[None]
    else:
        unsqueeze_flag = False

    assert len(points.shape) == 2 and (points.shape[-1] == 6 or points.shape[-1] == 3)
    assert len(rot.shape) == 2
    assert isinstance(trans, float) or (len(trans.shape) <= 2 and trans.shape[-1] == 3)

    if points.shape[-1] == 6:
        resulted_points = np.concatenate(
            [
                ((points[..., :3] - trans) @ rot) / scale,
                points[..., 3:] @ rot.T,
            ],
            axis=-1,
        )
    elif points.shape[-1] == 3:
        resulted_points = ((points[..., :3] - trans) @ rot) / scale

    if unsqueeze_flag:
        resulted_points = resulted_points.squeeze(0)
    return resulted_points


def np_get_delta_qpos(qpos1, qpos2):
    # qpos: [x, y, z, qw, qx, qy, qz]
    delta_pos = np.linalg.norm(qpos1[:3] - qpos2[:3])  # (1)
    q1_inv = tq.qinverse(qpos1[3:]).astype(np.float32)
    q_rel = tq.qmult(qpos2[3:], q1_inv).astype(np.float32)
    if np.abs(q_rel[0]) > 1:
        q_rel[0] = 1
    angle = 2 * np.arccos(q_rel[0])
    angle_degrees = np.degrees(angle)
    return delta_pos, angle_degrees


def np_even_sample_points_on_sphere(dim_num, delta_angle=45):
    """
    The method comes from https://stackoverflow.com/a/62754601
    Sample angles evenly in each dimension and finally normalize to sphere.
    """
    assert 90 % delta_angle == 0
    point_per_dim = 90 // delta_angle + 1
    point_num = point_per_dim ** (dim_num - 1) * dim_num * 2
    # print(f"Start to generate {point_num} points (with duplication) on S^{dim_num-1}!")

    comb = np.arange(point_per_dim ** (dim_num - 1))
    comb_lst = []
    for i in range(dim_num - 1):
        comb_lst.append(comb % point_per_dim)
        comb = comb // point_per_dim
    comb_array = np.stack(comb_lst, axis=-1)  # [p, d-1]

    # used to remove duplicated points!
    has_one = ((comb_array == point_per_dim - 1) | (comb_array == 0)) * np.arange(
        start=1, stop=dim_num
    )
    has_one = np.where(has_one == 0, dim_num, has_one)
    has_one = has_one.min(axis=-1)

    points_lst = []
    angle_array = (comb_array * delta_angle - 45) * np.pi / 180
    points_part = np.tan(angle_array)
    np_ones = np.ones_like(points_part[:, 0:1])  # [p, 1]
    for i in range(dim_num):
        pp1 = points_part[np.where(i < has_one)[0], :]  # remove duplicated points!
        points = np.concatenate(
            [
                np.concatenate([pp1[:, :i], np_ones[: pp1.shape[0]]], axis=-1),
                pp1[:, i:],
            ],
            axis=-1,
        )
        points_lst.append(points)

        pp2 = points_part[np.where(i < has_one)[0], :]  # remove duplicated points!
        points2 = np.concatenate(
            [
                np.concatenate([pp2[:, :i], -np_ones[: pp2.shape[0]]], axis=-1),
                pp2[:, i:],
            ],
            axis=-1,
        )
        points_lst.append(points2)

    points_array = np.concatenate(points_lst, axis=0)  # [P, d]
    points_array = np_normalize_vector(points_array)
    # print(f"Finish generating! Got {points_array.shape[0]} points (without duplication) on S^{dim_num-1}!")
    return points_array


def np_random_sample_points_on_sphere(dim_num, point_num):
    points = np.random.randn(point_num, dim_num)
    points = np_normalize_vector(points)
    return points
