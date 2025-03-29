import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import trimesh
import logging

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

from dexonomy.util.file_util import load_json
from dexonomy.util.np_rot_util import np_normalize_vector


class ObjSampleDataset(Dataset):

    def __init__(
        self,
        root_folder,
        init_point_num,
        load_pose,
        selection={},
    ):
        self.tm_mesh_cache = {}
        self.root_folder = root_folder
        self.init_point_num = init_point_num
        self.load_pose = load_pose

        self.object_id_lst = selection["id"]
        if self.object_id_lst is None:
            if selection["id_lst_path"] is not None:
                self.object_id_lst = load_json(selection["id_lst_path"])
            else:
                self.object_id_lst = os.listdir(root_folder)
        if selection["shuffle"]:
            self.object_id_lst = np.random.permutation(self.object_id_lst)
        if selection["start"] is not None and selection["end"] is not None:
            self.object_id_lst = self.object_id_lst[
                selection["start"] : selection["end"]
            ]

        print(f"Object number: {len(self.object_id_lst)}")
        return

    def __len__(self):
        return len(self.object_id_lst)

    def __getitem__(self, index):
        obj_name = self.object_id_lst[index]
        obj_path = os.path.join(self.root_folder, obj_name)
        if self.load_pose:
            obj_pose = load_json(os.path.join(obj_path, "info/tabletop_pose.json"))
            obj_pose_id = random.randint(0, len(obj_pose) - 1)
            obj_pose = np.array(obj_pose[obj_pose_id]).astype(np.float32)
            obj_pose[3:] = np_normalize_vector(obj_pose[3:])
        else:
            obj_pose_id = -1
            obj_pose = np.array([0.0, 0, 0, 1, 0, 0, 0]).astype(np.float32)

        if obj_name not in self.tm_mesh_cache.keys():
            mesh_path = os.path.join(self.root_folder, obj_name, "mesh/simplified.obj")
            tm_obj = trimesh.load(mesh_path, force="mesh")
            self.tm_mesh_cache[obj_name] = tm_obj
        else:
            tm_obj = self.tm_mesh_cache[obj_name]

        sampled_points, sampled_tri_ind = trimesh.sample.sample_surface_even(
            tm_obj, self.init_point_num
        )
        more_point_num = self.init_point_num - sampled_points.shape[0]
        if more_point_num > 0:
            new_sampled_points, new_sampled_tri_ind = trimesh.sample.sample_surface(
                tm_obj, more_point_num
            )
            sampled_points = np.concatenate(
                [sampled_points, new_sampled_points], axis=0
            )
            sampled_tri_ind = np.concatenate(
                [sampled_tri_ind, new_sampled_tri_ind], axis=0
            )
        sampled_normals = -tm_obj.face_normals[sampled_tri_ind]

        return {
            "obj_name": obj_name,
            "obj_path": obj_path,
            "obj_pose_id": obj_pose_id,
            "swf_of_pose": obj_pose,
            "of_ogc": np.array(tm_obj.center_mass, dtype=np.float32),
            "tm_mesh": self.tm_mesh_cache[obj_name],
            "sampled_points": np.array(sampled_points, dtype=np.float32),
            "sampled_normals": np.array(sampled_normals, dtype=np.float32),
        }


def _customized_collate_fn(list_data):
    if "tm_mesh" in list_data[0]:
        tm_mesh_lst = [i.pop("tm_mesh") for i in list_data]
    else:
        tm_mesh_lst = None
    ret_data = default_collate(list_data)
    if tm_mesh_lst is not None:
        ret_data["tm_mesh"] = tm_mesh_lst
    return ret_data


def get_object_dataloader(configs, n_worker):
    dataset = ObjSampleDataset(
        root_folder=configs.root_folder,
        init_point_num=configs.init_point_num,
        load_pose=configs.load_pose,
        selection=configs.selection,
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
