import os
import multiprocessing
import logging
import numpy as np
import glob
import traceback

from dexonomy.sim import MuJoCo_TestEnv
from dexonomy.util.file_util import load_json


def _single_validation(params):
    input_npy_path, configs = params[0], params[1]
    output_npy_path = input_npy_path.replace(configs.grasp_dir, configs.succ_dir)
    task_config = configs.task
    hand_config = configs.hand

    grasp_data = np.load(input_npy_path, allow_pickle=True).item()

    scene_cfg = grasp_data["scene_cfg"]
    for obj_cfg in scene_cfg["scene"].values():
        if obj_cfg["type"] == "rigid_mesh":
            obj_info = load_json(
                os.path.join(
                    os.path.dirname(obj_cfg["file_path"]), "../info/simplified.json"
                )
            )
            obj_coef = obj_info["mass"] / (
                obj_info["density"] * (obj_info["scale"] ** 3)
            )
            obj_cfg["density"] = configs.obj_mass / (
                obj_coef * np.prod(obj_cfg["scale"])
            )

    sim_env = MuJoCo_TestEnv(
        hand_xml_path=hand_config.xml_path,
        hand_add_mocap=hand_config.add_mocap,
        hand_exclude_table_contact=hand_config.exclude_table_contact,
        friction_coef=task_config.miu_coef,
        scene_cfg=scene_cfg,
        debug_render=configs.debug_render,
        debug_viewer=configs.debug_viewer,
    )

    assert (
        not task_config.force_closure
        or task_config.test_mode == "static"
        or sim_env.plane_num == 0
    )
    if task_config.force_closure:
        external_force_direction = np.array(
            [
                [1.0, 0, 0, 0, 0, 0],
                [-1.0, 0, 0, 0, 0, 0],
                [0.0, 1, 0, 0, 0, 0],
                [0.0, -1, 0, 0, 0, 0],
                [0.0, 0, 1, 0, 0, 0],
                [0.0, 0, -1, 0, 0, 0],
            ]
        )
    else:
        external_force_direction = [grasp_data["scene_cfg"]["interest_direction"]]

    for i, extforce_direction in enumerate(external_force_direction):
        succ_flag = eval(f"sim_env.test_mocap_{task_config.test_mode}")(
            grasp_data["grasp_qpos"],
            grasp_data["squeeze_qpos"],
            task_config.trans_thre,
            task_config.angle_thre,
            moving_distance=0.1 * extforce_direction[:3],
            extforce=-10 * extforce_direction * configs.obj_mass,
        )
        if not succ_flag:
            break

    sim_env.debug_postprocess(
        save_path=input_npy_path.replace(configs.grasp_dir, configs.debug_dir).replace(
            ".npy", ".gif"
        )
    )

    if succ_flag:
        os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
        os.system(
            f"ln -s {os.path.relpath(input_npy_path, os.path.dirname(output_npy_path))} {output_npy_path}"
        )
        if (
            grasp_data["evolution_num"] <= configs.max_template_evolution
            and configs.update_template is not False
        ):
            tmp_npy_path = input_npy_path.replace(
                configs.grasp_dir, configs.new_template_dir
            )
            os.makedirs(os.path.dirname(tmp_npy_path), exist_ok=True)
            os.system(
                f"ln -s {os.path.relpath(input_npy_path, os.path.dirname(tmp_npy_path))} {tmp_npy_path}"
            )

    return input_npy_path


def safe_validation(params):
    try:
        return _single_validation(params)
    except Exception as e:
        error_traceback = traceback.format_exc()
        logging.info(f"{error_traceback}")
        return params[0]


def task_syn_test(configs):
    input_path_lst = glob.glob(
        os.path.join(
            configs.grasp_dir,
            configs.template_name,
            configs.obj_name,
            configs.data_name + ".npy",
        )
    )
    logged_paths = []
    if configs.skip and os.path.exists(configs.log_path):
        with open(configs.log_path, "r") as f:
            logged_paths = f.readlines()

        logged_paths = [p.split("\n")[0] for p in logged_paths]
        input_path_lst = list(set(input_path_lst).difference(set(logged_paths)))

    logging.info(f"Find {len(input_path_lst)} graspdata")

    if len(input_path_lst) == 0:
        return

    iterable_params = zip(input_path_lst, [configs] * len(input_path_lst))
    if configs.debug or configs.debug_viewer:
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
