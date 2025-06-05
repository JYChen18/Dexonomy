import os
import multiprocessing
from glob import glob
import logging
import traceback

import numpy as np
from transforms3d import quaternions as tq

from dexonomy.sim import MuJoCo_RobotFK
from dexonomy.util.usd_helper import UsdHelper, Material
from dexonomy.util.vis_util import scene_cfg2mesh
from dexonomy.util.file_util import get_template_name_lst, load_scene_cfg


def read_npy(params):
    npy_path, kin, task_config = params[0], params[1], params[2]

    data = np.load(npy_path, allow_pickle=True).item()
    scene_cfg = load_scene_cfg(data["scene_path"])

    hand_pose_lst = []
    for qpos_name in task_config.qpos_type:
        hand_pose = data[f"{qpos_name}_qpos"][:7]
        hand_qpos = data[f"{qpos_name}_qpos"][7:]

        xmat, xpos = kin.forward_kinematics(qpos=hand_qpos, pose=hand_pose)
        hand_link_pose = []
        for body_name in kin.body_mesh_dict.keys():
            body_id = kin.body_id_dict[body_name]
            hand_link_pose.append(
                np.concatenate([xpos[body_id], tq.mat2quat(xmat[body_id])])
            )
        hand_link_pose = np.stack(hand_link_pose)

        hand_pose_lst.append(hand_link_pose)
    scene_cfg["scene_id"] += "_" + os.path.basename(npy_path)
    return {
        "scene_cfg": scene_cfg,
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
    kin = MuJoCo_RobotFK(configs.hand.xml_path, vis_mesh_mode=configs.task.hand.mode)
    init_robot_name_lst, init_robot_mesh_lst = kin.get_init_body_meshes()

    if not os.path.exists(configs.init_dir):
        tmp_lst = get_template_name_lst(None, configs.grasp_dir)
    else:
        tmp_lst = get_template_name_lst(None, configs.init_template_dir)

    for temp_name in tmp_lst:
        usd_helper = UsdHelper()

        task_config = configs.task
        if task_config.data_type == "grasp":
            data_folder, check_folder = configs.grasp_dir, configs.succ_dir
        elif task_config.data_type == "init":
            data_folder, check_folder = configs.init_dir, configs.grasp_dir
        elif task_config.data_type == "succ":
            data_folder, check_folder = configs.succ_dir, None
        elif task_config.data_type == "new_template":
            data_folder, check_folder = configs.new_template_dir, None
        else:
            raise NotImplementedError(
                f"Valid choices: 'grasp', 'init', 'succ', 'init_template', 'new_template'. Current: '{task_config.data_type}' "
            )
        input_path_lst = glob(os.path.join(data_folder, "**/*.npy"), recursive=True)
        if configs.debug_name is not None:
            input_path_lst = [p for p in input_path_lst if configs.debug_name in p]

        if check_folder is not None and task_config.check_success is not None:
            check_path_lst = glob(
                os.path.join(check_folder, "**/*.npy"), recursive=True
            )
            if configs.debug_name is not None:
                check_path_lst = [p for p in check_path_lst if configs.debug_name in p]
            check_path_lst = [
                p.replace(check_folder, data_folder) for p in check_path_lst
            ]

            if task_config.check_success:
                input_path_lst = list(set(input_path_lst).difference(check_path_lst))
            elif not task_config.check_success:
                input_path_lst = list(set(input_path_lst).intersection(check_path_lst))

        if len(input_path_lst) == 0:
            continue
        logging.info(
            f"Find {len(input_path_lst)} in {data_folder}. Debug name: {configs.debug_name}. Check success: {task_config.check_success}"
        )

        if configs.task.max_num > 0 and len(input_path_lst) > configs.task.max_num:
            input_path_lst = np.random.permutation(input_path_lst)[
                : configs.task.max_num
            ]
        logging.info(f"Use {min(len(input_path_lst), configs.task.max_num)}")

        param_lst = [(i, kin, configs.task) for i in input_path_lst]
        with multiprocessing.Pool(processes=configs.n_worker) as pool:
            result_iter = pool.imap_unordered(read_npy_safe, param_lst)
            result_iter = [r for r in list(result_iter) if r is not None]

        scene_dict = {}
        for r in result_iter:
            scene_id = r["scene_cfg"]["scene_id"]
            if scene_id not in scene_dict:
                scene_dict[scene_id] = []
            scene_dict[scene_id].append(r)

        data_length = len(configs.task.qpos_type)

        hand_pose_scale_lst = np.ones(
            (
                len(result_iter) * data_length,
                result_iter[0]["hand_link_pose"].shape[-2],
                8,
            )
        )
        obj_pose_scale_lst = np.ones(
            (len(result_iter) * data_length, len(scene_dict.keys()), 8)
        )
        obj_vit_lst = []
        obj_name_lst = []
        obj_mesh_lst = []
        count = 0
        for i, (k, v_lst) in enumerate(scene_dict.items()):
            obj_name_lst.append(k.replace("/", "_"))
            obj_mesh_lst.append(scene_cfg2mesh(v_lst[0]["scene_cfg"]))
            obj_vit_lst.append([count, count + len(v_lst) * data_length])
            for v in v_lst:
                hand_pose_scale_lst[count : count + data_length, :, :-1] = v[
                    "hand_link_pose"
                ]
                obj_pose_scale_lst[count : count + data_length, i] = np.array(
                    [0.0, 0, 0, 1, 0, 0, 0, 1]
                )
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
