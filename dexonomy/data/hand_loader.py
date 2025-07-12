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

from dexonomy.util.np_util import (
    np_array32,
    np_normal_to_rot,
    np_transform_points,
    np_inv_transform_points,
)
from dexonomy.sim import MuJoCo_VisEnv, HandCfg
from dexonomy.util.file_util import load_yaml


class HandTemplateLoader:
    def __init__(
        self,
        xml_path,
        skt_path,
        tmpl_name,
        init_tmpl_dir,
        new_tmpl_dir,
        max_data_path=100,
        max_data_buffer=1024,
        n_worker=4,
    ):
        # Producer components
        self.data_path_list = []
        self.data_path_lock = threading.Lock()
        self.data_path_condition = threading.Condition(self.data_path_lock)

        # Consumer components
        self.consumer_pool = ThreadPoolExecutor(max_workers=n_worker)

        # Buffer components
        self.data_buffer = {
            "hand_path": [],
            "grasp_qpos": [],
            "rot_c2h": [],
            "trans_c2h": [],
            "hand_cpn_c": [],
            "hand_skt_h": [],
            "n_evo": [],
        }

        self.buffer_lock = threading.Lock()

        # Control flags
        self.running = False
        self.producer_thread = None
        self.n_worker = n_worker
        self.max_data_path = max_data_path
        self.max_data_buffer = max_data_buffer

        # Hand informations
        self.xml_path = xml_path
        self.init_tmpl_dir = init_tmpl_dir
        self.new_tmpl_dir = new_tmpl_dir
        self.tmpl_name = tmpl_name
        assert os.path.exists(os.path.join(init_tmpl_dir, tmpl_name + ".npy"))

        # Read and pre-process skeleton informations
        skt_dict = load_yaml(skt_path)
        vis_env = MuJoCo_VisEnv(hand_cfg=HandCfg(xml_path=xml_path))
        self.skt_data, self.skt_bid = [], []
        for k, v in skt_dict.items():
            self.skt_data.extend(v)
            self.skt_bid.extend([vis_env.get_body_id(k)] * len(v))
        self.skt_data, self.skt_bid = np_array32(self.skt_data), np.array(self.skt_bid)

        # Pre-read the initial template
        hand_data = self._load_data(
            vis_env, os.path.join(init_tmpl_dir, tmpl_name + ".npy")
        )
        for k in self.data_buffer.keys():
            self.data_buffer[k].append(
                torch.from_numpy(hand_data[k])
                if isinstance(hand_data[k], np.ndarray)
                else hand_data[k]
            )
        self.extra_info = {
            "tmpl_name": tmpl_name,
            "hand_cbody": hand_data["hand_cbody"],
            "required_cbody": hand_data["required_cbody"],
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
        for _ in range(self.n_worker):
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
            if self.new_tmpl_dir is None:
                template_lst = []
            else:
                template_lst = glob(
                    os.path.join(self.new_tmpl_dir, self.tmpl_name, "**/*.npy"),
                    recursive=True,
                )
            template_lst.append(
                os.path.join(self.init_tmpl_dir, self.tmpl_name + ".npy")
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
                    n_new = min(
                        len(template_lst), self.max_data_path - len(self.data_path_list)
                    )
                    self.data_path_list.extend(template_lst[:n_new])
                    self.data_path_condition.notify(n_new)

            time.sleep(1.0)  # Control production rate

    def _update_data_buffer(self):
        """Consumer thread that load data and stores results in buffer"""
        vis_env = MuJoCo_VisEnv(hand_cfg=HandCfg(xml_path=self.xml_path))
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
                    hand_data = self._load_data(vis_env, data_path)
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

    def _load_data(self, vis_env: MuJoCo_VisEnv, data_path: str):
        hand_data = np.load(data_path, allow_pickle=True).item()
        hand_data["grasp_qpos"] = hand_data["grasp_qpos"].squeeze()
        xmat, xpos = vis_env.forward_kinematics(hand_data["grasp_qpos"][7:])
        skt_xmat, skt_xpos = xmat[self.skt_bid], xpos[self.skt_bid]
        hand_data["hand_skt_h"] = (
            np.reshape(self.skt_data, (-1, 2, 3)) @ skt_xmat.transpose(0, 2, 1)
            + skt_xpos[:, None, :]
        ).reshape(-1, 6)

        h_r = tq.quat2mat(hand_data["grasp_qpos"][3:7]).astype(np.float32)
        h_t = hand_data["grasp_qpos"][:3]
        h_cpn_h = np_inv_transform_points(hand_data["hand_cpn_w"], h_r, h_t)
        init_idx = np.random.randint(low=0, high=len(h_cpn_h))
        rot_c2h = np_normal_to_rot(h_cpn_h[init_idx, 3:].reshape(1, 3)).squeeze(0).T
        trans_c2h = -rot_c2h @ h_cpn_h[init_idx, :3]
        hand_data["rot_c2h"] = rot_c2h
        hand_data["trans_c2h"] = trans_c2h
        hand_data["hand_cpn_c"] = np_transform_points(h_cpn_h, rot_c2h, trans_c2h)
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
def test_hand_library(cfg):
    tmpl_name = os.listdir(cfg.init_tmpl_dir)[-1].split(".npy")[0]
    hand_library = HandTemplateLoader(
        xml_path=cfg.hand.xml_path,
        skt_path=cfg.hand_skt_path,
        tmpl_name=tmpl_name,
        init_tmpl_dir=cfg.init_tmpl_dir,
        new_tmpl_dir=cfg.new_tmpl_dir,
        n_worker=cfg.n_worker,
    )
    try:
        for _ in range(20):
            for i in range(3):
                time.sleep(0.1)
                d = hand_library.get_batched_data(
                    torch.Size([10, 8192]), torch.device("cpu")
                )
                logging.info(f"Batched data shape: {d['grasp_qpos'].shape}")
            time.sleep(0.8)
    finally:
        hand_library.stop()
        logging.info("Pipeline stopped")

    return


if __name__ == "__main__":
    test_hand_library()
