from typing import List
import os
from glob import glob
import logging
import multiprocessing

import numpy as np

from dexonomy.util.file_util import load_yaml
from dexonomy.sim import MuJoCo_RobotFK
from dexonomy.util.np_rot_util import (
    np_transform_points,
    np_array32,
    np_inv_transform_points,
)


def _single_anno2temp(params):
    anno_path, hand_keypoint, hand_body_group, configs = (
        params[0],
        params[1],
        params[2],
        params[3],
    )
    anno_data = load_yaml(anno_path)
    qpos_lst = np_array32(anno_data["qpos"])
    kin = MuJoCo_RobotFK(xml_path=configs.hand.xml_path, vis_mesh_mode="collision")
    xmat, xpos = kin.forward_kinematics(qpos_lst)

    if "contact" not in anno_data or anno_data["contact"] is None:
        anno_data["contact"] = {}

    # Convert dict to list
    contact_lst = []
    for body_name, contact_anno in anno_data["contact"].items():
        if isinstance(contact_anno, List):
            # Judge whether it is a list of points or a single point list
            if (
                len(contact_anno) == 6
                and sum([isinstance(c, float) for c in contact_anno]) > 0
            ):
                contact_lst.append((body_name, contact_anno))
            else:
                for c in contact_anno:
                    contact_lst.append((body_name, c))
        else:
            contact_lst.append((body_name, contact_anno))

    # Process contact list
    hand_worldframe_contact = []
    hand_contact_body_names = []
    for body_name, contact_anno in contact_lst:
        body_mesh = kin.body_mesh_dict[body_name]
        body_id = kin.body_id_dict[body_name]
        body_mat, body_pos = xmat[body_id], xpos[body_id]
        if isinstance(contact_anno, List):  # Annotated directly in handframe
            assert len(contact_anno) == 6
            hwc = np_array32(contact_anno)
            hbc = np_inv_transform_points(hwc, body_mat, body_pos)
        else:  # Annotated in bodyframe via keypoints
            hbc = np_array32(hand_keypoint[body_name][contact_anno])
            hwc = np_transform_points(hbc, body_mat, body_pos)

        _, distance, _ = body_mesh.nearest.on_surface([hbc[:3]])
        if distance > 0.002:
            min_body_name = body_name
            min_point2mesh_dist = distance
            for bn, bm in kin.body_mesh_dict.items():
                new_body_id = kin.body_id_dict[bn]
                new_body_mat, new_body_pos = xmat[new_body_id], xpos[new_body_id]
                new_hbc = np_inv_transform_points(hwc, new_body_mat, new_body_pos)
                _, new_dist, _ = bm.nearest.on_surface([new_hbc[:3]])
                if new_dist < min_point2mesh_dist:
                    min_point2mesh_dist = new_dist
                    min_body_name = bn
            if min_point2mesh_dist > 0.002:
                error_str = (
                    "The annotated contact point is far from the collision mesh!\n"
                )
            else:
                error_str = "The annotated contact point seems to be inconsistent with the body name!\n"
            error_str += " " * 10 + f"Template path: {anno_path}\n"
            error_str += (
                " " * 10
                + f"Annotated body name: {body_name}, Point2body distance: {distance};\n"
            )
            error_str += (
                " " * 10
                + f"Nearest body name: {min_body_name}, Point2body distance: {min_point2mesh_dist}.\n"
            )
            logging.error(error_str)
        hand_worldframe_contact.append(hwc)
        hand_contact_body_names.append(body_name)

    if "necessary_contact_body_names" not in anno_data:
        necessary_contact_body_names = []
        for i in hand_body_group:
            necessary_contact_body_names.append([])
            for j in i:
                if j in anno_data["contact"].keys():
                    necessary_contact_body_names[-1].append(j)
            if len(necessary_contact_body_names[-1]) == 0:
                necessary_contact_body_names.pop(-1)
    else:
        necessary_contact_body_names = anno_data["necessary_contact_body_names"]
        check_lst = []
        for i in necessary_contact_body_names:
            for j in i:
                assert j in anno_data["contact"].keys()
                assert j not in check_lst
                check_lst.append(j)

    temp_data = {
        "hand_template_name": os.path.basename(anno_path).removesuffix(".yaml"),
        "grasp_qpos": np.concatenate([np_array32([0.0, 0, 0, 1, 0, 0, 0]), qpos_lst]),
        "hand_worldframe_contacts": (
            np.stack(hand_worldframe_contact, axis=0)
            if len(hand_worldframe_contact) > 0
            else None
        ),
        "hand_contact_body_names": hand_contact_body_names,
        "necessary_contact_body_names": necessary_contact_body_names,
        "evolution_num": np_array32([0.0]),
    }
    temp_path = anno_path.replace(
        configs.raw_anno_dir, configs.init_template_dir
    ).replace(".yaml", ".npy")
    np.save(temp_path, temp_data)

    return


def task_anno2temp(configs):

    input_path_lst = glob(os.path.join(configs.raw_anno_dir, "**.yaml"))
    if configs.debug_name is not None:
        input_path_lst = [p for p in input_path_lst if configs.debug_name in p]
    input_num = len(input_path_lst)
    logging.info(f"Find {input_num} annotation")

    if input_num == 0:
        return

    os.makedirs(configs.init_template_dir, exist_ok=True)
    hand_body_group = load_yaml(configs.hand_body_group_path)["body_group"]

    hand_keypoint = None
    if os.path.exists(configs.hand_keypoint_path):
        hand_keypoint = load_yaml(configs.hand_keypoint_path)
        if hand_keypoint is not None:
            for k, v in hand_keypoint.items():
                if isinstance(v, str):
                    hand_keypoint[k] = hand_keypoint[v]

    iterable_params = zip(
        input_path_lst,
        [hand_keypoint] * input_num,
        [hand_body_group] * input_num,
        [configs] * input_num,
    )
    if configs.n_worker == 1:
        for ip in iterable_params:
            _single_anno2temp(ip)
    else:
        with multiprocessing.Pool(processes=configs.n_worker) as pool:
            result_iter = pool.imap_unordered(_single_anno2temp, iterable_params)
            results = list(result_iter)
    return
