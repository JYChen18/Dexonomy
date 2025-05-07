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

from dexonomy.util.np_rot_util import (
    np_array32,
    np_normal_to_rot,
    np_axis_angle_rotation,
)


def sample_init_pose(tm_obj: trimesh.Trimesh, init_point_num, init_inplane_num):
    # Sample contact points and corresponding normals
    points, tri_ind = trimesh.sample.sample_surface_even(tm_obj, init_point_num)
    more_point_num = init_point_num - points.shape[0]
    if more_point_num > 0:
        new_points, new_tri_ind = trimesh.sample.sample_surface(tm_obj, more_point_num)
        points = np.concatenate([points, new_points], axis=0)
        tri_ind = np.concatenate([tri_ind, new_tri_ind], axis=0)
    normals = -tm_obj.face_normals[tri_ind]

    # Contact to pose
    rot_base = np_normal_to_rot(normals)[None]
    angles = np.linspace(-np.pi, np.pi, init_inplane_num)
    delta_rot = np_axis_angle_rotation("X", angles).reshape(-1, 1, 3, 3)
    sampled_rot = (rot_base @ delta_rot).reshape(-1, 3, 3)
    sampled_trans = np.tile(points, (init_inplane_num, 1))

    return sampled_rot, sampled_trans


class ObjSampleDataset(Dataset):

    def __init__(self, init_point_num, init_inplane_num, cfg_path, cfg_num):
        self.init_point_num = init_point_num
        self.init_inplane_num = init_inplane_num
        self.path_lst = np.random.permutation(sorted(glob(cfg_path)))
        if cfg_num is not None and cfg_num > 0:
            self.path_lst = self.path_lst[:cfg_num]

        print(f"Object number: {len(self.path_lst)}")
        return

    def __len__(self):
        return len(self.path_lst)

    def __getitem__(self, index):
        scene_cfg_path = self.path_lst[index]
        scene_cfg = np.load(scene_cfg_path, allow_pickle=True).item()

        for obj_info in scene_cfg["scene"].values():
            if "file_path" in obj_info and not os.path.isabs(obj_info["file_path"]):
                for k, v in obj_info.items():
                    if k.endswith("_path"):
                        obj_info[k] = os.path.join(os.path.dirname(scene_cfg_path), v)
            if "part_info" in obj_info:
                for k, v in obj_info["part_info"].items():
                    if not os.path.isabs(v["file_path"]):
                        v["file_path"] = os.path.join(
                            os.path.dirname(scene_cfg_path), v["file_path"]
                        )

        obj_name = scene_cfg["task"]["obj_name"]
        obj_info = scene_cfg["scene"][obj_name]
        if obj_info["type"] == "articulated_object":
            part_name = scene_cfg["task"]["part_name"]
            obj_info = obj_info["part_info"][part_name]
        elif obj_info["type"] != "rigid_object":
            raise NotImplementedError(
                f"Unsupported interest object type: {obj_info['type']}"
            )
        tm_obj = trimesh.load(obj_info["file_path"], force="mesh")
        obj_scale = obj_info["scale"]
        obj_pose = np_array32(obj_info["pose"])

        sampled_rot, sampled_trans = sample_init_pose(
            tm_obj, self.init_point_num, self.init_inplane_num
        )

        wf_ext_center = obj_center = (
            tq.rotate_vector(tm_obj.center_mass * obj_scale, obj_pose[3:])
            + obj_pose[:3]
        )
        obj_mass = obj_info["mass"] if "mass" in obj_info else 0.1
        if scene_cfg["task"]["type"] == "hinge":  # rotation axis is fixed
            move_direction = np.cross(
                scene_cfg["task"]["axis"], obj_center - scene_cfg["task"]["pos"]
            )
            wf_ext_wrench = np.concatenate(
                [move_direction * obj_mass, move_direction * 0.0]
            )
        elif scene_cfg["task"]["type"] == "slide":
            wf_ext_wrench = np.concatenate(
                [scene_cfg["task"]["axis"] * obj_mass, scene_cfg["task"]["axis"] * 0.0]
            )
        elif scene_cfg["task"]["type"] == "force_closure":
            wf_ext_wrench = np.ones(6)
        else:
            raise NotImplementedError(
                f"Unsupported task type: {scene_cfg['task']['type']}. Avaiable choices: 'hinge', 'slide'."
            )

        collision_plane = None
        for obj in scene_cfg["scene"].values():
            if obj["type"] == "plane":
                collision_plane = np_array32(obj["pose"])

        return {
            "scene_cfg": scene_cfg,
            "collision_plane": collision_plane,
            "collision_mesh": (obj_name, tm_obj),
            "obj_scale": np_array32(obj_scale),
            "wf_sof_pose": obj_pose,
            "wf_ext_center": np_array32(wf_ext_center),
            "wf_ext_wrench": np_array32(wf_ext_wrench),
            "sampled_rot": np_array32(sampled_rot),
            "sampled_trans": np_array32(sampled_trans),
        }


def _customized_collate_fn(list_data):
    mesh_lst = []
    scene_cfg_lst = []
    no_plane = list_data[0]["collision_plane"] is None
    for i, data in enumerate(list_data):
        if no_plane:
            assert (
                data.pop("collision_plane") is None
            ), "Do not support unbatchable collision planes"
        mesh_lst.append(data.pop("collision_mesh"))
        scene_cfg_lst.append(data.pop("scene_cfg"))
    ret_data = default_collate(list_data)
    ret_data["collision_mesh"] = mesh_lst
    ret_data["scene_cfg"] = scene_cfg_lst
    return ret_data


def get_object_dataloader(configs, n_worker):
    dataset = ObjSampleDataset(
        init_point_num=configs.init_point_num,
        init_inplane_num=configs.init_inplane_num,
        cfg_path=configs.cfg_path,
        cfg_num=configs.cfg_num,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=configs.batch_size,
        num_workers=n_worker,
        shuffle=False,
        collate_fn=_customized_collate_fn,
        pin_memory=True,
    )
    return dataloader
