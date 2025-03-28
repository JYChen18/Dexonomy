import os
import multiprocessing
import logging
import numpy as np
import mujoco
import mujoco.viewer
import pdb
import glob

from copy import deepcopy

from util.rot_utils import np_get_delta_qpos
from util.mujoco_utils import build_spec_for_test
from util.file_utils import load_json


def _single_mjtest(params):
    input_npy_path, configs = params[0], params[1]
    output_npy_path = input_npy_path.replace(configs.grasp_dir, configs.succ_dir)
    task_config = configs.task
    hand_config = configs.hand

    grasp_data = np.load(input_npy_path, allow_pickle=True).item()

    obj_info = load_json(os.path.join(grasp_data["obj_path"], "info/simplified.json"))
    obj_coef = obj_info["mass"] / (obj_info["density"] * (obj_info["scale"] ** 3))
    new_obj_density = configs.obj_mass / (obj_coef * (grasp_data["obj_scale"] ** 3))
    hospec = build_spec_for_test(
        obj_path=grasp_data["obj_path"],
        obj_pose=grasp_data["obj_pose"],
        obj_scale=grasp_data["obj_scale"],
        has_floor_z0=False,
        obj_density=new_obj_density,
        hand_xml_path=hand_config.xml_path,
        grasp_pose=grasp_data["grasp_pose"],
        grasp_qpos=grasp_data["grasp_qpos"],
        hand_tendon=hand_config.tendon,
        friction_coef=task_config.miu_coef,
        ho_margin=task_config.ho_margin,
        ho_target_dist=task_config.ho_target_dist,
    )

    # Get ready for simulation
    mj_model = hospec.spec.compile()
    mj_data = mujoco.MjData(mj_model)

    # Initialize hand pose by setting keyframe
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, mj_model.nkey - 1)
    mujoco.mj_forward(mj_model, mj_data)

    if configs.debug:
        with open("debug.xml", "w") as f:
            f.write(hospec.spec.to_xml())

    if configs.debug_view:
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
        viewer.sync()
        pdb.set_trace()

    _, _, pre_obj_pose = hospec.split_qpos_pose(mj_data.qpos)
    pre_obj_qpos = deepcopy(pre_obj_pose)

    grasp_qpos = grasp_data["grasp_qpos"]
    if task_config.pregrasp_type == "real" and "pregrasp_qpos" in grasp_data:
        pregrasp_qpos = grasp_data["pregrasp_qpos"]
    elif task_config.pregrasp_type == "minus":
        pregrasp_qpos = deepcopy(grasp_qpos) - task_config.pregrasp_minus
    elif task_config.pregrasp_type == "multiply":
        pregrasp_qpos = grasp_qpos * np.where(
            grasp_qpos > 0,
            task_config.pregrasp_multiply,
            2 - task_config.pregrasp_multiply,
        )

    if task_config.squeeze_type == "real" and "squeeze_qpos" in grasp_data:
        squeeze_qpos = grasp_data["squeeze_qpos"]
        pregrasp_qpos = grasp_data["grasp_qpos"]
    else:
        squeeze_qpos = (
            grasp_qpos - pregrasp_qpos
        ) * task_config.squeeze_ratio + grasp_qpos

    dense_actuator_moment = np.zeros((mj_model.nu, mj_model.nv))
    mujoco.mju_sparse2dense(
        dense_actuator_moment,
        mj_data.actuator_moment,
        mj_data.moment_rownnz,
        mj_data.moment_rowadr,
        mj_data.moment_colind,
    )
    squeeze_hand_ctrl = dense_actuator_moment[:, 6:-6] @ squeeze_qpos
    grasp_hand_ctrl = dense_actuator_moment[:, 6:-6] @ grasp_qpos

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
        external_force_direction = [grasp_data["obj_gravity_direction"]]

    for i in range(len(external_force_direction)):
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, mj_model.nkey - 1)
        mj_data.qfrc_applied[:] = 0.0
        mj_data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)
        for j in range(10):
            mj_data.ctrl[:] = (j + 1) / 10 * (
                squeeze_hand_ctrl - grasp_hand_ctrl
            ) + grasp_hand_ctrl
            mujoco.mj_forward(mj_model, mj_data)
            for _ in range(10):
                mujoco.mj_step(mj_model, mj_data)
        mj_data.xfrc_applied[-1] = 10 * external_force_direction[i] * configs.obj_mass
        for j in range(10):
            for _ in range(50):
                mujoco.mj_step(mj_model, mj_data)

            if configs.debug_view:
                viewer.sync()
                pdb.set_trace()

            _, _, latter_obj_qpos = hospec.split_qpos_pose(mj_data.qpos)
            delta_pos, delta_angle = np_get_delta_qpos(pre_obj_qpos, latter_obj_qpos)
            succ_flag = (delta_pos < 0.05) & (delta_angle < 15)
            if not succ_flag:
                break
        if not succ_flag:
            break

    if configs.debug or configs.debug_view:
        print(succ_flag, delta_pos, delta_angle, input_npy_path)

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
                configs.grasp_dir, configs.template_dir
            )
            os.makedirs(os.path.dirname(tmp_npy_path), exist_ok=True)
            os.system(
                f"ln -s {os.path.relpath(input_npy_path, os.path.dirname(tmp_npy_path))} {tmp_npy_path}"
            )

    return input_npy_path


def safe_mjtest(params):
    try:
        return _single_mjtest(params)
    except:
        logging.warning("MuJoCo error during Testing!")
        return params[0]


def task_mjtest(configs):
    input_path_lst = glob.glob(
        os.path.join(
            configs.grasp_dir, configs.debug_template, configs.debug_obj, "**.npy"
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
    if configs.debug or configs.debug_view:
        for ip in iterable_params:
            _single_mjtest(ip)
    else:
        with multiprocessing.Pool(processes=3 * configs.n_worker) as pool:
            result_iter = pool.imap_unordered(safe_mjtest, iterable_params)
            results = list(result_iter)
            write_mode = "a" if configs.skip else "w"
            with open(configs.log_path, write_mode) as f:
                f.write("\n".join(results) + "\n")

    logging.info(f"Finish")

    return
