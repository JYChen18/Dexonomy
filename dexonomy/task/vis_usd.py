import os
import multiprocessing
from glob import glob
import trimesh
import logging
import omegaconf
import traceback

import numpy as np
from transforms3d import quaternions as tq

from util.hand_fk import RobotKinematics
from dexonomy.data.hand_loader import HandTemplateLoader
from util.usd_helper import UsdHelper, Material


def read_npy(params):
    npy_path, xml_path, task_config = params[0], params[1], params[2]
    data = np.load(npy_path, allow_pickle=True).item()
    hand_fk = RobotKinematics(xml_path)

    hand_pose_name_lst = []
    if "pregrasp" in task_config.data_type:
        hand_pose_name_lst.append(["pregrasp_pose", "pregrasp_qpos"])
    if "grasp" in task_config.data_type:
        hand_pose_name_lst.append(["grasp_pose", "grasp_qpos"])
    if "squeezed" in task_config.data_type:
        hand_pose_name_lst.append(["grasp_pose", "squeeze_qpos"])

    obj_pose_lst = []
    hand_pose_lst = []
    for hand_pose_names in hand_pose_name_lst:
        pose_name, qpos_name = hand_pose_names[0], hand_pose_names[1]
        if len(hand_fk.mj_data.qpos) == len(data[qpos_name]) + 7:
            qpos = np.concatenate([np.array([0.0, 0, 0, 1, 0, 0, 0]), data[qpos_name]])
        elif len(hand_fk.mj_data.qpos) == len(data[qpos_name]):
            qpos = data[qpos_name]
        else:
            raise NotImplementedError(
                f"qpos length: {len(data[qpos_name])}; xml length: {len(hand_fk.mj_data.qpos)}"
            )

        if task_config.normalize_hand:
            # Normalize to hand space
            ht = data[pose_name][:3]
            hr = tq.quat2mat(data[pose_name][3:])
            oT = data["obj_pose"][:3]
            oR = tq.quat2mat(data["obj_pose"][3:])
            new_oR = hr.T @ oR
            new_oT = hr.T @ (oT - ht)
            new_obj_pose = np.concatenate([new_oT, tq.mat2quat(new_oR)])
            new_hand_pose = np.array([0.0, 0, 0, 1, 0, 0, 0])

            delta_bias = np.array([0.0, 0.1, -0.1])
            new_obj_pose[:3] += delta_bias
            new_hand_pose[:3] += delta_bias
        else:
            new_obj_pose = data["obj_pose"]
            new_hand_pose = data[pose_name]
            # new_hand_pose = np.array([0, 0, 0.0, 1, 0, 0, 0])

        hand_fk.forward_kinematics(qpos)
        hand_link_pose = hand_fk.get_poses(new_hand_pose)

        obj_pose_lst.append(np.concatenate([new_obj_pose, data["obj_scale"]], axis=-1))
        hand_pose_lst.append(hand_link_pose)

    obj_path = data["obj_path"]
    if not obj_path.endswith(".obj"):
        obj_path = os.path.join(data["obj_path"], "mesh/coacd.obj")
    if not os.path.exists(obj_path):
        raise NotImplementedError
    return {
        "obj_path": obj_path,
        "obj_pose_scale": np.stack(obj_pose_lst, axis=0),
        "hand_link_pose": np.stack(hand_pose_lst, axis=0),
    }


def read_npy_safe(params):
    try:
        return read_npy(params)
    except Exception as e:
        error_traceback = traceback.format_exc()
        logging.warning(f"{error_traceback}")
        return None


def task_vis_usd(configs):

    hand_loader = HandTemplateLoader(**configs.hand)
    # tmp_path = (
    #     "/mnt/disk1/jiayichen/code/ContactMatch/assets/hand/shadow/taxonomy/6_v3.yaml"
    # )
    # mesh_data = hand_loader.get_template_mesh(tmp_path, False, True, False)

    init_robot_name_lst, init_robot_mesh_lst = hand_loader.kinematic.get_init_meshes()

    logging.info(
        f"Folder name is {os.path.join(configs.succ_dir, '**', configs.debug_obj, '**')}"
    )
    if configs.debug_template == "**":
        temp_name_lst = hand_loader.template_names
    elif isinstance(configs.debug_template, omegaconf.listconfig.ListConfig):
        temp_name_lst = configs.debug_template
    elif isinstance(configs.debug_template, str):
        temp_name_lst = [configs.debug_template]

    for temp_name in temp_name_lst:
        save_path = os.path.join(
            configs.visd_dir, "usd", f"{temp_name.replace('**', 'all')}.usd"
        )

        usd_helper = UsdHelper()

        input_folder = os.path.join(
            configs.succ_dir, temp_name, configs.debug_obj, "**"
        )
        input_file_lst = glob(input_folder)

        if len(input_file_lst) == 0:
            continue
        logging.info(f"Find {len(input_file_lst)} graspdata for {temp_name}")

        if configs.task.max_num > 0 and len(input_file_lst) > configs.task.max_num:
            input_file_lst = np.random.permutation(input_file_lst)[
                : configs.task.max_num
            ]

        # final_path_lst = []
        # data_dict = {}
        # for p in input_file_lst:
        #     folder_name = os.path.dirname(p)
        #     if folder_name not in data_dict.keys():
        #         data_dict[folder_name] = True
        #         final_path_lst.append(p)
        # input_file_lst = final_path_lst
        # logging.info(f"Find {len(input_file_lst)} graspdata for {temp_name}")

        param_lst = [(i, configs.hand.xml_path, configs.task) for i in input_file_lst]
        with multiprocessing.Pool(processes=configs.n_worker) as pool:
            result_iter = pool.imap_unordered(read_npy_safe, param_lst)
            result_iter = [r for r in list(result_iter) if r is not None]

        obj_path_dict = {}
        for r in result_iter:
            op = r.pop("obj_path")
            if op not in obj_path_dict:
                obj_path_dict[op] = []
            obj_path_dict[op].append(r)

        data_length = len(configs.task.data_type)

        hand_pose_scale_lst = np.ones(
            (
                len(result_iter) * data_length,
                result_iter[0]["hand_link_pose"].shape[-2],
                8,
            )
        )
        obj_pose_scale_lst = np.ones(
            (len(result_iter) * data_length, len(obj_path_dict.keys()), 8)
        )
        obj_vit_lst = []
        obj_name_lst = []
        obj_mesh_lst = []
        count = 0
        for i, (k, v_lst) in enumerate(obj_path_dict.items()):
            obj_name_lst.append(k.replace("/", "_"))
            obj_mesh_lst.append(trimesh.load(k, force="mesh"))
            obj_vit_lst.append([count, count + len(v_lst) * data_length])
            for v in v_lst:
                hand_pose_scale_lst[count : count + data_length, :, :-1] = v[
                    "hand_link_pose"
                ]
                obj_pose_scale_lst[count : count + data_length, i] = v["obj_pose_scale"]
                count += data_length

        usd_helper.create_stage(
            save_path, timesteps=len(result_iter) * data_length, dt=0.01
        )
        # import pdb
        # pdb.set_trace()
        # usd_helper.add_meshlst_to_stage(
        #     [mesh_data["contact_normal_mesh"]],
        #     ["cn"],
        #     np.array([0.0, 0, 0, 1, 0, 0, 0, 1])[None, None],
        #     obstacles_frame="cn",
        #     material=Material(color=[0.3, 0.3, 0.3, 1.0], name="cn"),
        # )
        # usd_helper.add_meshlst_to_stage(
        #     [mesh_data["contact_point_mesh"]],
        #     ["cp"],
        #     np.array([0.0, 0, 0, 1, 0, 0, 0, 1])[None, None],
        #     obstacles_frame="cp",
        #     material=Material(color=[0.5, 0.5, 0.5, 1.0], name="cp"),
        # )

        # Add hands
        usd_helper.add_meshlst_to_stage(
            init_robot_mesh_lst,
            init_robot_name_lst,
            hand_pose_scale_lst,
            obstacles_frame="robot",
            material=Material(color=configs.hand.color, name="obj"),
        )

        # Add objects
        usd_helper.add_meshlst_to_stage(
            obj_mesh_lst,
            obj_name_lst,
            obj_pose_scale_lst,
            visible_time=obj_vit_lst,
            obstacles_frame="object",
            material=Material(color=[0.5, 0.5, 0.5, 1.0], name="obj"),
        )
        logging.info(f"Save to {os.path.abspath(save_path)}")
        usd_helper.write_stage_to_file(save_path)
