import os
from glob import glob

import numpy as np
from transforms3d import quaternions as tq
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import trimesh
import logging

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

from dexonomy.util.np_util import (
    np_array32,
    np_normal_to_rot,
    np_axis_angle_rotation,
)
from dexonomy.util.file_util import load_scene_cfg


def sample_init_pose(
    tm_obj: trimesh.Trimesh, n_init_point: int, n_init_inplane: int
) -> tuple[np.ndarray, np.ndarray]:
    # Sample contact points and corresponding normals
    points, tri_ind = trimesh.sample.sample_surface_even(tm_obj, n_init_point)
    n_remain = n_init_point - points.shape[0]
    if n_remain > 0:
        new_points, new_tri_ind = trimesh.sample.sample_surface(tm_obj, n_remain)
        points = np.concatenate([points, new_points], axis=0)
        tri_ind = np.concatenate([tri_ind, new_tri_ind], axis=0)
    normals = -tm_obj.face_normals[tri_ind]

    # Contact point and normal to relative pose
    rot_base = np_normal_to_rot(normals)[None]
    angles = np.linspace(-np.pi, np.pi, n_init_inplane)
    delta_rot = np_axis_angle_rotation("X", angles).reshape(-1, 1, 3, 3)
    rot_o2c_init = (rot_base @ delta_rot).reshape(-1, 3, 3)
    trans_o2c_init = np.tile(points, (n_init_inplane, 1))

    return rot_o2c_init, trans_o2c_init


class ObjSampleDataset(Dataset):

    def __init__(self, n_init_point, n_init_inplane, cfg_path, n_cfg, mass):
        self.n_init_point = n_init_point
        self.n_init_inplane = n_init_inplane
        self.mass = mass
        self.path_lst = np.random.permutation(sorted(glob(cfg_path)))
        if n_cfg is not None and n_cfg > 0:
            self.path_lst = self.path_lst[:n_cfg]

        logging.info(f"Object number: {len(self.path_lst)}")
        return

    def __len__(self):
        return len(self.path_lst)

    def __getitem__(self, index):
        scene_cfg_path = self.path_lst[index]
        scene_cfg = load_scene_cfg(scene_cfg_path)

        obj_name = scene_cfg["task"]["obj_name"]
        obj_info = scene_cfg["scene"][obj_name]
        if obj_info["type"] == "articulated_object":
            part_name = scene_cfg["task"]["part_name"]
            obj_info = obj_info["part_info"][part_name]
        elif obj_info["type"] != "rigid_object":
            raise NotImplementedError(
                f"Unsupported target object type: {obj_info['type']}"
            )
        tm_obj = trimesh.load(obj_info["file_path"], force="mesh")
        obj_scale = obj_info["scale"]
        obj_pose = np_array32(obj_info["pose"])

        rot_o2c_init, trans_o2c_init = sample_init_pose(
            tm_obj, self.n_init_point, self.n_init_inplane
        )

        ext_center = obj_center = (
            tq.rotate_vector(tm_obj.center_mass * obj_scale, obj_pose[3:])
            + obj_pose[:3]
        )
        obj_mass = self.mass
        if scene_cfg["task"]["type"] == "hinge":  # rotation axis is fixed
            move_direction = np.cross(
                scene_cfg["task"]["axis"], obj_center - scene_cfg["task"]["pos"]
            )
            ext_wrench = np.concatenate(
                [move_direction * obj_mass, move_direction * 0.0]
            )
        elif scene_cfg["task"]["type"] == "slide":
            ext_wrench = np.concatenate(
                [scene_cfg["task"]["axis"] * obj_mass, scene_cfg["task"]["axis"] * 0.0]
            )
        elif (
            scene_cfg["task"]["type"] == "force_closure"
            or scene_cfg["task"]["type"] == "keyframe"
        ):
            ext_wrench = np.zeros(6)
        else:
            raise NotImplementedError(
                f"Unsupported task type: {scene_cfg['task']['type']}. Avaiable choices: 'hinge', 'slide', 'force_closure', 'keyframe'."
            )

        col_plane = None
        if "virtual_plane" in obj_info:
            col_plane = np_array32(obj_info["virtual_plane"])
        else:
            for obj in scene_cfg["scene"].values():
                if obj["type"] == "plane":
                    col_plane = np_array32(obj["pose"])

        return {
            "scene_cfg": scene_cfg,
            "scene_path": scene_cfg_path,
            "col_plane": col_plane,
            "col_mesh": (obj_name, tm_obj),
            "obj_scale": np_array32(obj_scale),
            "pose_w2so": obj_pose,
            "ext_center": np_array32(ext_center),
            "ext_wrench": np_array32(ext_wrench),
            "rot_o2c_init": np_array32(rot_o2c_init),
            "trans_o2c_init": np_array32(trans_o2c_init),
        }


def _customized_collate_fn(list_data):
    mesh_lst = []
    scene_cfg_lst = []
    scene_path_lst = []
    no_col_plane = list_data[0]["col_plane"] is None
    for i, data in enumerate(list_data):
        if no_col_plane:
            assert (
                data.pop("col_plane") is None
            ), "Do not support unbatchable collision planes"
        mesh_lst.append(data.pop("col_mesh"))
        scene_cfg_lst.append(data.pop("scene_cfg"))
        scene_path_lst.append(data.pop("scene_path"))
    ret_data = default_collate(list_data)
    ret_data["col_mesh"] = mesh_lst
    ret_data["scene_cfg"] = scene_cfg_lst
    ret_data["scene_path"] = scene_path_lst
    if no_col_plane:
        ret_data["col_plane"] = None
    return ret_data


def get_object_dataloader(cfg, n_worker):
    dataset = ObjSampleDataset(
        n_init_point=cfg.n_init_point,
        n_init_inplane=cfg.n_init_inplane,
        cfg_path=cfg.cfg_path,
        n_cfg=cfg.n_cfg,
        mass=cfg.mass,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=n_worker,
        shuffle=False,
        collate_fn=_customized_collate_fn,
        pin_memory=True,
    )
    return dataloader
