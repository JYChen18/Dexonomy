import os
import multiprocessing
from glob import glob
import logging
import traceback

import numpy as np
from transforms3d import quaternions as tq

from dexonomy.sim import MuJoCo_VisEnv, HandCfg, MuJoCo_TestCfg, MuJoCo_OptCfg
from dexonomy.util.usd_helper import UsdHelper, Material
from dexonomy.util.file_util import get_template_name_lst, load_scene_cfg


def read_npy(params):
    npy_path, config = params[0], params[1]
    task_config = config.task

    data = np.load(npy_path, allow_pickle=True).item()

    if config.adding_arm:
        kin = MuJoCo_VisEnv(
            hand_cfg=HandCfg(xml_path=config.hand.hand_on_arm.xml_path),
            scene_cfg=load_scene_cfg(data["scene_path"]),
            sim_cfg=MuJoCo_TestCfg(),
            vis_mesh_mode=task_config.hand.mode,
        )
    else:
        kin = MuJoCo_VisEnv(
            hand_cfg=HandCfg(xml_path=config.hand.xml_path, freejoint=True),
            scene_cfg=load_scene_cfg(data["scene_path"]),
            sim_cfg=MuJoCo_OptCfg(),
            vis_mesh_mode=task_config.hand.mode,
        )
    init_body_name_lst, init_body_mesh_lst = kin.get_init_body_meshes()
    body_pose_lst = []
    for qpos_name in task_config.qpos_type:
        for qpos_id in range(data[f"{qpos_name}_qpos"].shape[0]):
            xmat, xpos = kin.forward_kinematics(data[f"{qpos_name}_qpos"][qpos_id])
            body_pose = []
            for body_name in kin.body_mesh_dict.keys():
                body_id = kin.body_id_dict[body_name]
                body_pose.append(
                    np.concatenate(
                        [xpos[body_id], tq.mat2quat(xmat[body_id]), np.ones(1)]
                    )
                )
            body_pose = np.stack(body_pose, axis=0)
            body_pose_lst.append(body_pose)

    return {
        "body_name": list(init_body_name_lst),
        "body_mesh": list(init_body_mesh_lst),
        "body_pose": np.stack(body_pose_lst, axis=0),
    }


def read_npy_safe(params):
    try:
        return read_npy(params)
    except Exception as e:
        error_traceback = traceback.format_exc()
        logging.warning(f"{error_traceback}")
        return None


def task_vis_usd(configs):
    if not os.path.exists(configs.init_dir):
        tmp_lst = get_template_name_lst(None, configs.grasp_dir)
    else:
        tmp_lst = get_template_name_lst(None, configs.init_template_dir)

    for temp_name in tmp_lst:
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
        input_path_lst = glob(
            os.path.join(data_folder, temp_name, "**/*.npy"), recursive=True
        )
        if configs.debug_name is not None:
            input_path_lst = [p for p in input_path_lst if configs.debug_name in p]

        if check_folder is not None and task_config.check_success is not None:
            check_path_lst = glob(
                os.path.join(check_folder, temp_name, "**/*.npy"), recursive=True
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

        param_lst = [(i, configs) for i in input_path_lst]
        with multiprocessing.Pool(processes=configs.n_worker) as pool:
            result_iter = pool.imap_unordered(read_npy_safe, param_lst)
            result_iter = [r for r in list(result_iter) if r is not None]

        data_length = result_iter[0]["body_pose"].shape[0]
        save_path = os.path.join(
            configs.vis_usd_dir, "usd", f"{temp_name.replace('**', 'all')}.usd"
        )
        usd_helper = UsdHelper(
            save_path, timesteps=len(result_iter) * data_length, dt=0.01
        )

        count = 0
        for r in result_iter:
            usd_helper.add_meshlst_to_stage(
                r["body_mesh"],
                r["body_name"],
                r["body_pose"],
                vis_time_lst=[(count, count + data_length)] * len(r["body_name"]),
                material=Material(color=configs.hand.color, name="obj"),
            )
            count += data_length

        logging.info(f"Save to {os.path.abspath(save_path)}")
        usd_helper.write_stage_to_file(save_path)
