import time
import os
from glob import glob
import random
import threading
import time
from collections import deque

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transforms3d import quaternions as tq

from dexonomy.util.np_rot_util import (
    np_normal_to_rot,
    np_normalize_vector,
    np_transform_points,
    np_inv_transform_points,
)
from dexonomy.util.mujoco_util import RobotKinematics
from dexonomy.util.file_util import load_yaml


class HandDataBuffer:
    def __init__(
        self,
        xml_path,
        skeleton_path,
        template_name,
        init_template_dir,
        new_template_dir,
        batch_size,
        num_hand_workers,
        max_size=1024,
    ):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.dataloader = get_hand_dataloader(
            xml_path,
            skeleton_path,
            template_name,
            init_template_dir,
            new_template_dir,
            batch_size,
            num_hand_workers,
        )

    def start(self):
        """Start the buffer update thread"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._update_buffer, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the buffer update thread"""
        self.running = False
        if self.thread:
            self.thread.join()

    def _update_buffer(self):
        """Thread function that updates the buffer every second"""
        while self.running:
            # Safely update the buffer
            with self.lock:

                self.buffer.append(new_data)
                print(f"Buffer updated: {new_data}")  # Debug print

            # Wait for 1 second
            time.sleep(1)

    def get_buffer(self):
        """Get a copy of the current buffer contents"""
        with self.lock:
            return list(self.buffer)


class HandTemplateDataset(Dataset):
    def __init__(
        self,
        xml_path,
        skeleton_path,
        template_name,
        init_template_dir,
        new_template_dir,
    ):
        self.init_template_dir = init_template_dir
        self.new_template_dir = new_template_dir
        self.template_name = template_name
        hand_skeleton_dict = load_yaml(skeleton_path)

        self.kinematic = RobotKinematics(xml_path=xml_path)
        self.hand_skeleton = []
        self.hand_sk_body_id = []
        for k, v in hand_skeleton_dict.items():
            self.hand_skeleton.extend(v)
            self.hand_sk_body_id.extend([self.kinematic.body_id_dict[k]] * len(v))
        self.hand_skeleton = np.array(self.hand_skeleton)
        self.hand_sk_body_id = np.array(self.hand_sk_body_id)

        self.useful_keys = [
            "grasp_pose",
            "grasp_qpos",
            "hand_worldframe_contacts",
            "hand_contact_body_names",
            "necessary_contact_body_names",
            "obj_gravity_direction",
            "evolution_num",
        ]
        self.update_template_lst()
        return

    def update_template_lst(self):
        if self.new_template_dir is None:
            self.template_lst = []
        else:
            self.template_lst = glob(
                os.path.join(self.new_template_dir, self.template_name, "**/**.npy")
            )
        self.template_lst.append(
            os.path.join(self.init_template_dir, self.template_name + ".npy")
        )
        return

    def __getitem__(self, idx):
        hand_data = np.load(self.template_lst[idx], allow_pickle=True).item()
        for k in hand_data.keys():
            if k not in self.useful_keys:
                hand_data.pop(k)

        xmat, xpos = self.kinematic.forward_kinematics(hand_data["grasp_qpos"])
        body_xmat = xmat[self.hand_sk_body_id]
        body_xpos = xpos[self.hand_sk_body_id]
        hand_data["hf_hsk"] = np.concatenate(
            [
                (body_xmat @ self.hand_skeleton[..., :3, None]).squeeze(-1) + body_xpos,
                (body_xmat @ self.hand_skeleton[..., 3:, None]).squeeze(-1),
            ],
            axis=-1,
        )
        hr = tq.quat2mat(hand_data["grasp_pose"][3:])
        ht = hand_data["grasp_pose"][:3]
        hf_hc = np_inv_transform_points(hand_data["hand_worldframe_contacts"], hr, ht)
        init_idx = np.random.randint(low=0, high=len(hf_hc))
        nf_hf_rot = np_normal_to_rot(hf_hc[init_idx, 3:].reshape(1, 3)).squeeze(0).T
        nf_hf_trans = -nf_hf_rot @ hf_hc[init_idx, :3]
        hand_data["nf_hf_rot"] = nf_hf_rot
        hand_data["nf_hf_trans"] = nf_hf_trans
        hand_data["nf_hc"] = np_transform_points(hf_hc, nf_hf_rot, nf_hf_trans)
        hand_data["hf_ogd"] = np_inv_transform_points(
            hand_data.pop("obj_gravity_direction"), hr
        )
        return hand_data

    def __len__(self):
        return len(self.template_lst)


# def _customized_collate_fn(list_data):
#     if "tm_mesh" in list_data[0]:
#         tm_mesh_lst = [i.pop("tm_mesh") for i in list_data]
#     else:
#         tm_mesh_lst = None
#     ret_data = default_collate(list_data)
#     if tm_mesh_lst is not None:
#         ret_data["tm_mesh"] = tm_mesh_lst
#     return ret_data


def get_hand_dataloader(
    xml_path,
    skeleton_path,
    template_name,
    init_template_dir,
    new_template_dir,
    batch_size,
    num_hand_workers,
):
    dataset = HandTemplateDataset(
        xml_path,
        skeleton_path,
        template_name,
        init_template_dir,
        new_template_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_hand_workers,
        shuffle=False,
        pin_memory=True,
    )
    return dataloader


if __name__ == "__main__":
    hand_cfg = load_yaml("dexonomy/config/hand/shadow.yaml")

    hand_loader = get_hand_dataloader(hand_cfg, batch_size=10)

    st = time.time()
    check_lst = []
    for hd in hand_loader:
        check_lst.append(hd)
    print(time.time() - st)

    # for hd in check_lst:
    #     for h, xm, xp in zip(hd["qpos"], hd["xmat"], hd["xpos"]):
    #         new_xm, new_xp = hand_loader.dataset.kinematic.forward_kinematics(h)
    #         if (torch.tensor(new_xm) - xm).abs().max() > 1e-5:
    #             print("aa")
    #         if (torch.tensor(new_xp) - xp).abs().max() > 1e-5:
    #             print("bb")
