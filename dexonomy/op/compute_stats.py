import numpy as np
import os
from glob import glob
import multiprocessing
import logging
import matplotlib.pyplot as plt
import torch

from dexonomy.util.file_util import get_template_names, load_yaml
from dexonomy.util.torch_util import (
    torch_quaternion_to_matrix,
    torch_matrix_to_axis_angle,
)


def compute_cln_and_cdc_pd(ho_c: dict, body_group: list) -> tuple[float, float, float]:
    """
    Contact Link Number (CLN) and Contact Distance Consistency (CDC)
    """
    if ho_c is None or len(ho_c.get("dist", [])) == 0:
        return 0., 0.
    
    dists = np.maximum(np.array(ho_c["dist"]), 0.)
    pd = - np.minimum(np.array(ho_c["dist"]), 0.)
    bodies = np.array(ho_c["bn1"])
    
    contact_mask = dists < 0.002
    cbody_set = set(bodies[contact_mask])
    cln = float(len(cbody_set))

    # ===== CDC =====
    finger_min_dist_mm = []
    
    for finger_links in body_group[:-1]:
        finger_link_mask = np.isin(bodies, finger_links)
        if np.any(finger_link_mask):
            finger_min_dist_mm.append(float(np.min(dists[finger_link_mask]) * 1000.))
    
    cdc_mm = 0.
    if len(finger_min_dist_mm) >= 2:
        cdc_mm = float(max(finger_min_dist_mm) - min(finger_min_dist_mm))
    
    return cln, cdc_mm, max(pd)


def draw_obj_scale_fig(data_lst, save_path):
    obj_scale_lst = [float(d["obj_scale"]) for d in data_lst]

    bins = np.linspace(0.05, 0.2, 11)

    # Create the histogram
    plt.hist(
        np.array(obj_scale_lst),
        bins=bins,
        color="skyblue",
        edgecolor="black",
        rwidth=0.8,
    )

    # Add labels and title
    plt.xlabel("Scale")
    plt.ylabel("Frequency")
    plt.title("Distribution of Object Scales")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return


def get_obj_names(path_lst, template_name):
    obj_name_lst = []
    for p in path_lst:
        obj_name = os.path.dirname(p).split(template_name + "/")[1]
        obj_name_lst.append(obj_name)
    return list(set(obj_name_lst))


def read_paths(npy_path):
    data = np.load(npy_path, allow_pickle=True).item()
    return data


def get_CLN_CDC_PD(data_list, body_group):
    cln, cdc, pd = 0., 0., 0.
    for data in data_list:
        cln_tmp, cdc_tmp, pd_tmp = compute_cln_and_cdc_pd(data['ho_c'], body_group)
        cln += cln_tmp
        cdc += cdc_tmp
        pd = max(pd, pd_tmp)

    return cln / len(data_list), cdc / len(data_list), pd

def get_diversity(data_lst):
    from sklearn.decomposition import PCA

    grasp_poses = torch.tensor(
        np.stack([d["grasp_qpos"][0, :7] for d in data_lst], axis=0)
    ).float()
    grasp_qpos = torch.tensor(
        np.stack([d["grasp_qpos"][0, 7:] for d in data_lst], axis=0)
    ).float()
    obj_poses = torch.tensor(
        np.stack([d["obj_pose"] for d in data_lst], axis=0)
    ).float()

    obj_rot = torch_quaternion_to_matrix(obj_poses[:, 3:])
    hand_rot = torch_quaternion_to_matrix(grasp_poses[:, 3:])
    hand_real_trans = (
        obj_rot.transpose(-1, -2)
        @ (grasp_poses[:, :3] - obj_poses[:, :3]).unsqueeze(-1)
    ).squeeze(-1)
    hand_real_rot = obj_rot.transpose(-1, -2) @ hand_rot
    hand_final_pose = torch.cat(
        [
            hand_real_trans,
            torch_matrix_to_axis_angle(hand_real_rot),
            grasp_qpos,
        ],
        dim=-1,
    )

    pca = PCA()
    pca.fit(hand_final_pose.numpy())
    explained_variance = []
    for i in range(5):
        explained_variance.append(np.sum(pca.explained_variance_ratio_[: i + 1]))
    return explained_variance


def operate_stat(cfg):
    if not os.path.exists(cfg.init_dir) and os.path.exists(cfg.grasp_dir):
        tmpl_names = get_template_names(None, cfg.grasp_dir)
    else:
        tmpl_names = get_template_names(None, cfg.init_tmpl_dir)

    # Hand template summary
    for tn in tmpl_names:
        init_paths = glob(os.path.join(cfg.init_dir, tn, "**/*.npy"), recursive=True)
        grasp_paths = glob(os.path.join(cfg.grasp_dir, tn, "**/*.npy"), recursive=True)
        succ_grasp_paths = glob(
            os.path.join(cfg.succ_grasp_dir, tn, "**/*.npy"),
            recursive=True,
        )
        traj_paths = glob(os.path.join(cfg.traj_dir, tn, "**/*.npy"), recursive=True)
        succ_traj_paths = glob(
            os.path.join(cfg.succ_traj_dir, tn, "**/*.npy"),
            recursive=True,
        )

        if (
            len(succ_grasp_paths)
            + len(succ_traj_paths)
            + len(traj_paths)
            + len(init_paths)
            + len(grasp_paths)
            == 0
        ):
            continue

        init_obj = get_obj_names(init_paths, tn)
        grasp_obj = get_obj_names(grasp_paths, tn)
        succ_grasp_obj = get_obj_names(succ_grasp_paths, tn)
        traj_obj = get_obj_names(traj_paths, tn)
        succ_traj_obj = get_obj_names(succ_traj_paths, tn)

        if len(succ_grasp_paths) != 0 and (cfg.op.obj_scale or cfg.op.diversity or cfg.op.cln_cdc):
            with multiprocessing.Pool(processes=cfg.n_worker) as pool:
                jobs = pool.imap_unordered(read_paths, succ_grasp_paths)
                data_lst = list(jobs)

            if cfg.op.obj_scale:
                save_path = os.path.join(
                    os.path.dirname(cfg.log_path), tn + "_objscale.png"
                )
                draw_obj_scale_fig(data_lst, save_path)

            if cfg.op.diversity:
                pca_eigenvalue = get_diversity(data_lst)
                
            if cfg.op.cln_cdc:
                body_group = load_yaml(cfg.hand_bgroup_path)["body_group"]
                cln, cdc, pd = get_CLN_CDC_PD(data_lst, body_group)

        traj1 = len(succ_grasp_paths) if len(succ_grasp_paths) > 0 else len(grasp_paths)
        traj2 = len(grasp_obj) if len(grasp_obj) > 0 else len(succ_grasp_obj)
        header = "{:<20} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}".format(
            tn, "Init", "SynGrasp", "EvalGrasp", "SynTraj", "EvalTraj"
        )
        success_line = "{:<20} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}".format(
            "Grasp:",
            f"{len(init_paths)}",
            f"{len(grasp_paths)},{100 * len(grasp_paths) / max(1, len(init_paths)):.0f}%",
            f"{len(succ_grasp_paths)},{100 * len(succ_grasp_paths) / max(1, len(grasp_paths)):.0f}%",
            f"{len(traj_paths)},{100 * len(traj_paths) / max(1, traj1):.0f}%",
            f"{len(succ_traj_paths)},{100 * len(succ_traj_paths) / max(1, len(traj_paths)):.0f}%",
        )
        object_line = "{:<20} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}".format(
            "Object:",
            f"{len(init_obj)}",
            f"{len(grasp_obj)},{100 * len(grasp_obj) / max(1, len(init_obj)):.0f}%",
            f"{len(succ_grasp_obj)},{100 * len(succ_grasp_obj) / max(1, len(grasp_obj)):.0f}%",
            f"{len(traj_obj)},{100 * len(traj_obj) / max(1, traj2):.0f}%",
            f"{len(succ_traj_obj)},{100 * len(succ_traj_obj) / max(1, len(traj_obj)):.0f}%",
        )

        logging.info("-" * 65)
        logging.info(header)
        logging.info(success_line)
        logging.info(object_line)
        if cfg.op.diversity:
            logging.info(f"Diversity: {pca_eigenvalue}")
        if cfg.op.cln_cdc:
            logging.info(f"CLN: {cln}, CDC: {cdc}, PD: {pd}")
