from typing import List
import os
from glob import glob
import logging
import multiprocessing

import numpy as np

from dexonomy.util.file_util import load_yaml, write_yaml
from dexonomy.sim import MuJoCo_RobotFK
from dexonomy.util.np_rot_util import np_transform_points, np_array32


def _single_anno2temp(params):
    anno_path, hand_keypoint, hand_body_group, configs = (
        params[0],
        params[1],
        params[2],
        params[3],
    )
    anno_data = load_yaml(anno_path)
    qpos_lst = np_array32(anno_data["qpos"])
    kin = MuJoCo_RobotFK(xml_path=configs.hand.xml_path)
    xmat, xpos = kin.forward_kinematics(qpos_lst)

    hand_worldframe_contact = []
    for body_name, contact_anno in anno_data["contact"].items():
        if isinstance(contact_anno, List):
            assert len(contact_anno) == 6
            hand_worldframe_contact.append(np_array32(contact_anno))
        else:
            hbc = hand_keypoint[body_name][contact_anno]
            body_id = kin.body_id_dict[body_name]
            hwc = np_transform_points(hbc, xmat[body_id], xpos[body_id])
            hand_worldframe_contact.append(hwc)

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

    if "obj_gravity_direction" not in anno_data:
        obj_gravity_direction = np_array32([0.0, 0, 1, 0, 0, 0])
    else:
        obj_gravity_direction = np_array32(anno_data["obj_gravity_direction"])
        assert len(obj_gravity_direction) == 6 and np.testing.assert_allclose(
            np.linalg.norm(obj_gravity_direction), 1
        )

    temp_data = {
        "hand_template_name": os.path.basename(anno_path).removesuffix(".yaml"),
        "grasp_qpos": np.concatenate([np_array32([0.0, 0, 0, 1, 0, 0, 0]), qpos_lst]),
        "hand_worldframe_contacts": np.stack(hand_worldframe_contact, axis=0),
        "hand_contact_body_names": list(anno_data["contact"].keys()),
        "necessary_contact_body_names": necessary_contact_body_names,
        "obj_gravity_direction": obj_gravity_direction,
        "evolution_num": np_array32([0.0]),
    }
    temp_path = anno_path.replace(
        configs.raw_anno_dir, configs.init_template_dir
    ).replace(".yaml", ".npy")
    np.save(temp_path, temp_data)

    return


def task_anno2temp(configs):

    input_path_lst = glob(os.path.join(configs.raw_anno_dir, configs.template_name))
    input_num = len(input_path_lst)
    logging.info(f"Find {input_num} annotation")

    if input_num == 0:
        return

    os.makedirs(configs.init_template_dir, exist_ok=True)
    hand_body_group = load_yaml(configs.hand_body_group_path)["body_group"]
    hand_keypoint = load_yaml(configs.hand_keypoint_path)
    for k, v in hand_keypoint.items():
        if isinstance(v, str):
            hand_keypoint[k] = hand_keypoint[v]

    iterable_params = zip(
        input_path_lst,
        [hand_keypoint] * input_num,
        [hand_body_group] * input_num,
        [configs] * input_num,
    )
    if configs.debug:
        for ip in iterable_params:
            _single_anno2temp(ip)
    else:
        with multiprocessing.Pool(processes=configs.n_worker) as pool:
            result_iter = pool.imap_unordered(_single_anno2temp, iterable_params)
            results = list(result_iter)
    return
