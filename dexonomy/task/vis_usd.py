import os
import multiprocessing
from glob import glob
import trimesh
import logging
import omegaconf
import traceback

import numpy as np
from transforms3d import quaternions as tq

from dexonomy.sim import MuJoCo_RobotFK
from dexonomy.util.usd_helper import UsdHelper, Material


def read_npy(params):
    npy_path, kin, task_config = params[0], params[1], params[2]

    data = np.load(npy_path, allow_pickle=True).item()

    obj_pose_lst = []
    hand_pose_lst = []
    for qpos_name in task_config.qpos_type:
        hand_pose = data[f"{qpos_name}_qpos"][:7]
        hand_qpos = data[f"{qpos_name}_qpos"][7:]

        if task_config.normalize_hand:
            # Normalize to hand space
            ht = hand_pose[:3]
            hr = tq.quat2mat(hand_pose[3:])
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
            new_hand_pose = hand_pose

        xmat, xpos = kin.forward_kinematics(qpos=hand_qpos, pose=hand_pose)
        hand_link_pose = []
        for body_name in kin.body_mesh_dict.keys():
            body_id = kin.body_id_dict[body_name]
            hand_link_pose.append(
                np.concatenate([xpos[body_id], tq.mat2quat(xmat[body_id])])
            )
        hand_link_pose = np.stack(hand_link_pose)

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
    kin = MuJoCo_RobotFK(
        configs.hand.xml_path, vis_mesh_mode=configs.task.hand_mesh_mode
    )
    init_robot_name_lst, init_robot_mesh_lst = kin.get_init_meshes()

    if configs.template_name == "**":
        temp_name_lst = [
            p.removesuffix(".npy") for p in os.listdir(configs.init_template_dir)
        ]
    elif isinstance(configs.template_name, omegaconf.listconfig.ListConfig):
        temp_name_lst = configs.template_name
    elif isinstance(configs.template_name, str):
        temp_name_lst = [configs.template_name]

    for temp_name in temp_name_lst:
        usd_helper = UsdHelper()

        task_config = configs.task
        if task_config.data_type == "init_template":
            data_folder = configs.init_template_dir
            input_path_lst = glob(
                os.path.join(data_folder, configs.template_name + ".npy")
            )
        else:
            if task_config.data_type == "grasp":
                data_folder, check_folder = configs.grasp_dir, configs.succ_dir
            elif task_config.data_type == "init":
                data_folder, check_folder = configs.init_dir, configs.grasp_dir
            elif task_config.data_type == "succ":
                data_folder, check_folder = configs.succ_dir, None
            elif task_config.data_type == "new_template":
                data_folder, check_folder = configs.new_template_dir, None
            else:
                raise NotImplementedError
            input_path_example = os.path.join(
                data_folder, configs.template_name, configs.obj_name, "**"
            )
            input_path_lst = glob(input_path_example)
            if check_folder is not None and task_config.check_success is not None:
                check_path_lst = glob(
                    os.path.join(
                        check_folder, configs.template_name, configs.obj_name, "**"
                    )
                )
                if task_config.check_success:
                    input_path_lst = list(
                        set(input_path_lst).difference(check_path_lst)
                    )
                elif not task_config.check_success:
                    input_path_lst = list(
                        set(input_path_lst).intersection(check_path_lst)
                    )

        if len(input_path_lst) == 0:
            continue
        logging.info(f"Find {len(input_path_lst)} data for {input_path_example}")

        if configs.task.max_num > 0 and len(input_path_lst) > configs.task.max_num:
            input_path_lst = np.random.permutation(input_path_lst)[
                : configs.task.max_num
            ]
        logging.info(f"Use {configs.task.max_num}")

        param_lst = [(i, kin, configs.task) for i in input_path_lst]
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

        save_path = os.path.join(
            configs.vis_usd_dir, "usd", f"{temp_name.replace('**', 'all')}.usd"
        )
        usd_helper.create_stage(
            save_path, timesteps=len(result_iter) * data_length, dt=0.01
        )

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
