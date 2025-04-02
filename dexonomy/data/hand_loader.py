import time
import os
import hydra
import threading
import logging
import time
import traceback
from glob import glob
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from transforms3d import quaternions as tq

from dexonomy.util.np_rot_util import (
    np_array32,
    np_normal_to_rot,
    np_transform_points,
    np_inv_transform_points,
)
from dexonomy.sim import MuJoCo_RobotFK
from dexonomy.util.file_util import load_yaml


class HandTemplateLibrary:
    def __init__(
        self,
        xml_path,
        skeleton_path,
        template_name,
        init_template_dir,
        new_template_dir,
        max_data_path=100,
        max_data_buffer=1024,
        num_workers=4,
    ):
        # Producer components
        self.data_path_list = []
        self.data_path_lock = threading.Lock()
        self.data_path_condition = threading.Condition(self.data_path_lock)

        # Consumer components
        self.consumer_pool = ThreadPoolExecutor(max_workers=num_workers)

        # Buffer components
        self.data_buffer = {
            "hand_path": [],
            "grasp_qpos": [],
            "nf_hf_rot": [],
            "nf_hf_trans": [],
            "nf_hc": [],
            "hf_hsk": [],
            "hf_ogd": [],
            "evolution_num": [],
        }

        self.buffer_lock = threading.Lock()

        # Control flags
        self.running = False
        self.producer_thread = None
        self.num_workers = num_workers
        self.max_data_path = max_data_path
        self.max_data_buffer = max_data_buffer

        # Hand informations
        self.xml_path = xml_path
        self.init_template_dir = init_template_dir
        self.new_template_dir = new_template_dir
        self.template_name = template_name
        assert os.path.exists(os.path.join(init_template_dir, template_name + ".npy"))

        # Read and pre-process skeleton informations
        hand_skeleton_dict = load_yaml(skeleton_path)
        kinematic = MuJoCo_RobotFK(xml_path=xml_path, vis_mesh_mode=None)
        self.hand_skeleton = []
        self.hand_sk_body_id = []
        for k, v in hand_skeleton_dict.items():
            self.hand_skeleton.extend(v)
            self.hand_sk_body_id.extend([kinematic.body_id_dict[k]] * len(v))
        self.hand_skeleton = np_array32(self.hand_skeleton)
        self.hand_sk_body_id = np.array(self.hand_sk_body_id)

        # Pre-read the initial template
        hand_data = self._load_data(
            kinematic, os.path.join(init_template_dir, template_name + ".npy")
        )
        for k in self.data_buffer.keys():
            self.data_buffer[k].append(
                torch.from_numpy(hand_data[k])
                if isinstance(hand_data[k], np.ndarray)
                else hand_data[k]
            )
        self.extra_info = {
            "hand_template_name": template_name,
            "hand_contact_body_names": hand_data["hand_contact_body_names"],
            "necessary_contact_body_names": hand_data["necessary_contact_body_names"],
        }
        self.start()
        return

    def start(self):
        """Start all threads in the pipeline"""
        if self.running:
            return

        self.running = True
        # Start producer thread
        self.producer_thread = threading.Thread(
            target=self._update_data_path, daemon=True
        )
        self.producer_thread.start()

        # Start consumer threads
        for _ in range(self.num_workers):
            self.consumer_pool.submit(self._update_data_buffer)

    def stop(self):
        """Stop all threads gracefully"""
        self.running = False

        # Wake up all waiting threads
        with self.data_path_condition:
            self.data_path_condition.notify_all()

        # Shutdown thread pool
        self.consumer_pool.shutdown(wait=True)

        # Wait for producer thread
        if self.producer_thread:
            self.producer_thread.join()

    def _update_data_path(self):
        """Producer thread that generates data path lists to read"""
        while self.running:
            # Get all available data path
            if self.new_template_dir is None:
                template_lst = []
            else:
                template_lst = glob(
                    os.path.join(self.new_template_dir, self.template_name, "**/**.npy")
                )
            template_lst.append(
                os.path.join(self.init_template_dir, self.template_name + ".npy")
            )

            # Remove existing data path. NOTE: If there are workers loading data, it may generate repeat or miss path. But the problem is not severe.
            with self.buffer_lock:
                exist_path_set = set(self.data_buffer["hand_path"])
            template_lst = list(set(template_lst).difference(exist_path_set))
            logging.debug(f"(Producer) Current buffer: {len(exist_path_set)}")
            logging.debug(f"(Producer) Unduplicated path list: {len(template_lst)}")

            # Add new data path and notify workers
            with self.data_path_condition:
                if len(self.data_path_list) <= self.max_data_path:
                    new_data_num = min(
                        len(template_lst), self.max_data_path - len(self.data_path_list)
                    )
                    self.data_path_list.extend(template_lst[:new_data_num])
                    self.data_path_condition.notify(new_data_num)

            time.sleep(1.0)  # Control production rate

    def _update_data_buffer(self):
        """Consumer thread that load data and stores results in buffer"""
        kinematic = MuJoCo_RobotFK(self.xml_path, vis_mesh_mode=None)
        while self.running:
            data_path = None

            # Read from data path list
            with self.data_path_condition:
                while self.running and not self.data_path_list:
                    self.data_path_condition.wait()  # Wait for data
                if not self.running:
                    break
                data_path = self.data_path_list.pop(0)  # FIFO processing

            if data_path:
                try:
                    hand_data = self._load_data(kinematic, data_path)
                    with self.buffer_lock:
                        if len(self.data_buffer["hand_path"]) == self.max_data_buffer:
                            for k in self.data_buffer.keys():
                                self.data_buffer[k].pop(0)
                        for k in self.extra_info.keys():
                            assert (
                                self.extra_info[k] == hand_data[k]
                            ), f"{k} {self.extra_info[k]} {hand_data[k]}"
                        for k in self.data_buffer.keys():
                            self.data_buffer[k].append(
                                torch.from_numpy(hand_data[k])
                                if isinstance(hand_data[k], np.ndarray)
                                else hand_data[k]
                            )
                except Exception as e:
                    error_traceback = traceback.format_exc()
                    logging.error(f"(Hand Loader) {error_traceback}")

    def _load_data(self, kinematic: MuJoCo_RobotFK, data_path: str):
        hand_data = np.load(data_path, allow_pickle=True).item()
        xmat, xpos = kinematic.forward_kinematics(hand_data["grasp_qpos"][7:])
        body_xmat = xmat[self.hand_sk_body_id]
        body_xpos = xpos[self.hand_sk_body_id]
        hand_data["hf_hsk"] = (
            self.hand_skeleton.reshape(-1, 2, 3) @ body_xmat.transpose(0, 2, 1)
            + body_xpos[:, None, :]
        ).reshape(-1, 6)

        hr = tq.quat2mat(hand_data["grasp_qpos"][3:7]).astype(np.float32)
        ht = hand_data["grasp_qpos"][:3]
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
        hand_data["hand_path"] = data_path
        return hand_data

    def get_batched_data(self, data_size: torch.Size, device: torch.device):
        """Get processed data from buffer (thread-safe)"""
        with self.buffer_lock:
            buffer_length = len(self.data_buffer["hand_path"])
            rand_id = torch.randint(low=0, high=buffer_length, size=data_size)
            new_data = {}
            logging.info(f"(Hand Loader) Current buffer: {buffer_length}")
            for k, v in self.data_buffer.items():
                if isinstance(v[0], torch.Tensor):
                    new_data[k] = torch.stack(v)[rand_id].to(device)
            for k, v in self.extra_info.items():
                new_data[k] = v
            return new_data


@hydra.main(config_path="../config", config_name="base", version_base=None)
def test_hand_library(configs):
    template_name = os.listdir(configs.init_template_dir)[-1].split(".npy")[0]
    hand_library = HandTemplateLibrary(
        xml_path=configs.hand.xml_path,
        skeleton_path=configs.hand_skeleton_path,
        template_name=template_name,
        init_template_dir=configs.init_template_dir,
        new_template_dir=configs.new_template_dir,
        num_workers=configs.n_worker,
    )
    try:
        for _ in range(20):
            for i in range(3):
                time.sleep(0.1)
                d = hand_library.get_batched_data((10, 8192), "cpu")
                logging.info(f"Batched data shape: {d['grasp_qpos'].shape}")
            time.sleep(0.8)
    finally:
        hand_library.stop()
        logging.info("Pipeline stopped")

    return


if __name__ == "__main__":
    test_hand_library()
