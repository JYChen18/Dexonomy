import os
import multiprocessing
import logging
import numpy as np
import glob
import traceback

from dexonomy.sim import MuJoCo_TestEnv, HandCfg, MuJoCo_TestCfg
from dexonomy.util.file_util import load_scene_cfg, load_json
from dexonomy.util.traj_util import get_full_traj


def _single_validation(params):
    input_npy_path, configs = params[0], params[1]
    data_dir = configs.traj_dir if configs.adding_arm else configs.grasp_dir
    task_config = configs.task
    hand_config = configs.hand

    grasp_data = np.load(input_npy_path, allow_pickle=True).item()
    scene_cfg = load_scene_cfg(grasp_data["scene_path"])

    plane_flag = False
    for obj_name, obj_cfg in scene_cfg["scene"].items():
        if obj_cfg["type"] == "rigid_object":
            obj_info = load_json(obj_cfg["info_path"])
            obj_coef = obj_info["mass"] / (
                obj_info["density"] * (obj_info["scale"] ** 3)
            )
            obj_cfg["density"] = configs.obj_mass / (
                obj_coef * np.prod(obj_cfg["scale"])
            )
        elif obj_cfg["type"] == "plane":
            plane_flag = True
    if configs.adding_arm and not plane_flag:
        logging.warning(
            f"Using an arm but no table is in scene cfg: {grasp_data['scene_path']}"
        )

    sim_env = MuJoCo_TestEnv(
        hand_cfg=(
            HandCfg(arm_flag=True, **hand_config.hand_on_arm)
            if configs.adding_arm
            else HandCfg(xml_path=hand_config.xml_path, freejoint=True)
        ),
        scene_cfg=scene_cfg,
        sim_cfg=MuJoCo_TestCfg(
            friction_coef=task_config.miu_coef,
        ),
        debug_render=configs.debug_render,
        debug_viewer=configs.debug_viewer,
    )

    init_obj_pose = np.copy(sim_env.get_interest_object_pose())

    qpos_lst, ctype_lst, extdir_lst, interp_lst, target_obj_pose = get_full_traj(
        init_obj_pose=init_obj_pose,
        move_cfg=scene_cfg["task"],
        grasp_qpos=grasp_data["grasp_qpos"],
        squeeze_qpos=grasp_data["squeeze_qpos"],
        pregrasp_qpos=grasp_data["pregrasp_qpos"],
        approach_qpos=(
            grasp_data["robot_pose"] if "robot_pose" in grasp_data else None
        ),
    )

    if scene_cfg["task"]["type"] == "force_closure":
        eval_func = sim_env.test_fc
    else:
        eval_func = sim_env.test_move

    succ_flag, real_qpos_lst = eval_func(
        qpos_lst,
        ctype_lst,
        extdir_lst,
        interp_lst,
        target_obj_pose,
        task_config.trans_thre,
        task_config.angle_thre,
    )

    sim_env.debug_postprocess(
        save_path=input_npy_path.replace(data_dir, configs.debug_dir).replace(
            ".npy", ".gif"
        )
    )

    if succ_flag:
        if configs.adding_arm:
            output_npy_path = input_npy_path.replace(data_dir, configs.succ_traj_dir)
            os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
            np.save(
                output_npy_path,
                {"scene_path": grasp_data["scene_path"], "traj_qpos": real_qpos_lst},
            )
        else:
            output_npy_path = input_npy_path.replace(data_dir, configs.succ_grasp_dir)
            os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
            os.system(
                f"ln -s {os.path.relpath(input_npy_path, os.path.dirname(output_npy_path))} {output_npy_path}"
            )

        if (
            configs.update_template is not False
            and grasp_data["evolution_num"] <= configs.max_template_evolution
        ):
            tmp_npy_path = input_npy_path.replace(data_dir, configs.new_template_dir)
            if not os.path.exists(tmp_npy_path):
                original_npy_path = input_npy_path.replace(data_dir, configs.grasp_dir)
                os.makedirs(os.path.dirname(tmp_npy_path), exist_ok=True)
                os.system(
                    f"ln -s {os.path.relpath(original_npy_path, os.path.dirname(tmp_npy_path))} {tmp_npy_path}"
                )

    return input_npy_path


def safe_validation(params):
    try:
        return _single_validation(params)
    except Exception as e:
        error_traceback = traceback.format_exc()
        logging.info(f"{error_traceback}")
        return params[0]


def task_test(configs):
    data_dir = configs.traj_dir if configs.adding_arm else configs.grasp_dir
    input_path_lst = glob.glob(os.path.join(data_dir, "**/*.npy"), recursive=True)
    if configs.debug_name is not None:
        input_path_lst = [p for p in input_path_lst if configs.debug_name in p]

    logged_paths = []
    if configs.skip and os.path.exists(configs.log_path):
        with open(configs.log_path, "r") as f:
            logged_paths = f.readlines()

        logged_paths = [p.split("\n")[0] for p in logged_paths]
        input_path_lst = list(set(input_path_lst).difference(set(logged_paths)))

    logging.info(f"Find {len(input_path_lst)} grasp data")

    if len(input_path_lst) == 0:
        return

    iterable_params = zip(input_path_lst, [configs] * len(input_path_lst))
    if configs.n_worker == 1 or configs.debug_viewer:
        for ip in iterable_params:
            _single_validation(ip)
    else:
        with multiprocessing.Pool(processes=3 * configs.n_worker) as pool:
            result_iter = pool.imap_unordered(safe_validation, iterable_params)
            results = list(result_iter)
            write_mode = "a" if configs.skip else "w"
            with open(configs.log_path, write_mode) as f:
                f.write("\n".join(results) + "\n")

    logging.info(f"Finish")

    return
