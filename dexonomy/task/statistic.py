import numpy as np
import os
from glob import glob
import multiprocessing
import logging
import matplotlib.pyplot as plt
import torch

from dexonomy.util.file_util import get_template_name_lst
from dexonomy.util.torch_rot_util import (
    torch_quaternion_to_matrix,
    torch_matrix_to_axis_angle,
)


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


def get_obj_name_lst(path_lst, template_name):
    obj_name_lst = []
    for p in path_lst:
        obj_name = os.path.dirname(p).split(template_name + "/")[1]
        obj_name_lst.append(obj_name)
    return list(set(obj_name_lst))


def read_data(npy_path):
    data = np.load(npy_path, allow_pickle=True).item()
    return data


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


def task_stat(configs):
    if not os.path.exists(configs.init_dir):
        tmp_lst = get_template_name_lst(None, configs.grasp_dir)
    else:
        tmp_lst = get_template_name_lst(None, configs.init_template_dir)

    # Hand template summary
    for template_name in tmp_lst:
        all_init_lst = glob(
            os.path.join(configs.init_dir, template_name, "**/*.npy"), recursive=True
        )
        all_grasp_lst = glob(
            os.path.join(configs.grasp_dir, template_name, "**/*.npy"), recursive=True
        )
        all_eval_lst = glob(
            os.path.join(configs.succ_dir, template_name, "**/*.npy"), recursive=True
        )

        if len(all_eval_lst) + len(all_init_lst) + len(all_grasp_lst) == 0:
            continue

        init_obj_lst = get_obj_name_lst(all_init_lst, template_name)
        grasp_obj_lst = get_obj_name_lst(all_grasp_lst, template_name)
        succ_obj_lst = get_obj_name_lst(all_eval_lst, template_name)

        if len(all_eval_lst) != 0 and (
            configs.task.scale_fig
            or configs.task.diversity
            or configs.task.contact_number
        ):
            with multiprocessing.Pool(processes=configs.n_worker) as pool:
                result_iter = pool.imap_unordered(read_data, all_eval_lst)
                data_lst = list(result_iter)

            if configs.task.scale_fig:
                save_path = os.path.join(
                    os.path.dirname(configs.log_path), template_name + "_objscale.png"
                )
                draw_obj_scale_fig(data_lst, save_path)

            if configs.task.diversity:
                pca_eigenvalue = get_diversity(data_lst)

            if configs.task.contact_number:
                average_contact_number = np.mean(
                    [d["contact_number"] for d in data_lst]
                )

        header = "{:<20} | {:<10} | {:<10} | {:<10}".format(
            template_name, "SynObj", "SynHand", "Test"
        )
        success_line = "{:<20} | {:<10} | {:<10} | {:<10}".format(
            "Grasp:",
            f"{len(all_init_lst)}",
            f"{len(all_grasp_lst)},{100 * len(all_grasp_lst) / max(1, len(all_init_lst)):.0f}%",
            f"{len(all_eval_lst)},{100 * len(all_eval_lst) / max(1, len(all_grasp_lst)):.0f}%",
        )
        object_line = "{:<20} | {:<10} | {:<10} | {:<10}".format(
            "Object:",
            f"{len(init_obj_lst)}",
            f"{len(grasp_obj_lst)},{100 * len(grasp_obj_lst) / max(1, len(init_obj_lst)):.0f}%",
            f"{len(succ_obj_lst)},{100 * len(succ_obj_lst) / max(1, len(grasp_obj_lst)):.0f}%",
        )

        logging.info("-" * 65)
        logging.info(header)
        logging.info(success_line)
        logging.info(object_line)
        if configs.task.diversity:
            logging.info(f"Diversity: {pca_eigenvalue}")
        if configs.task.contact_number:
            logging.info(f"Contact Number: {average_contact_number}")
