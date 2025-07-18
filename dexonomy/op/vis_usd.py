import os
import multiprocessing
from glob import glob
import logging

import numpy as np
from transforms3d import quaternions as tq

from dexonomy.sim import MuJoCo_VisEnv, HandCfg, MuJoCo_EvalCfg, MuJoCo_OptCfg
from dexonomy.util.usd_util import UsdStage, Material
from dexonomy.util.file_util import get_template_names, load_scene_cfg, safe_wrapper


@safe_wrapper
def _single_vusd(param):
    npy_path, cfg = param[0], param[1]
    op_cfg = cfg.op

    data = np.load(npy_path, allow_pickle=True).item()

    if op_cfg.data == "succ_traj":
        vis_env = MuJoCo_VisEnv(
            hand_cfg=HandCfg(cfg.hand.arm_hand.xml_path),
            scene_cfg=load_scene_cfg(data["scene_path"]),
            sim_cfg=MuJoCo_EvalCfg(),
            vis_mode=op_cfg.hand.mode,
        )
    else:
        vis_env = MuJoCo_VisEnv(
            hand_cfg=HandCfg(cfg.hand.xml_path, freejoint=True),
            scene_cfg=load_scene_cfg(data["scene_path"]),
            sim_cfg=MuJoCo_OptCfg(),
            vis_mode=op_cfg.hand.mode,
        )
    all_body_name = vis_env.get_all_body_names()
    all_body_mesh = [vis_env.get_body_mesh(body_name) for body_name in all_body_name]
    all_body_pose = []
    if isinstance(op_cfg.qpos, str):
        qpos_name_lst = [op_cfg.qpos]
    else:
        qpos_name_lst = op_cfg.qpos
    for qpos_name in qpos_name_lst:
        for qpos_id in range(data[f"{qpos_name}_qpos"].shape[0]):
            xmat, xpos = vis_env.forward_kinematics(data[f"{qpos_name}_qpos"][qpos_id])
            body_pose = []
            for body_name in all_body_name:
                body_id = vis_env.get_body_id(body_name)
                body_pose.append(
                    np.concatenate(
                        [xpos[body_id], tq.mat2quat(xmat[body_id]), np.ones(1)]
                    )
                )
            body_pose = np.stack(body_pose, axis=0)
            all_body_pose.append(body_pose)

    return {
        "body_name": all_body_name,
        "body_mesh": all_body_mesh,
        "body_pose": np.stack(all_body_pose, axis=0),
    }


def operate_vusd(cfg):
    if not os.path.exists(cfg.init_dir) and os.path.exists(cfg.grasp_dir):
        tmp_lst = get_template_names(None, cfg.grasp_dir)
    else:
        tmp_lst = get_template_names(None, cfg.init_tmpl_dir)

    for temp_name in tmp_lst:
        op_cfg = cfg.op
        if op_cfg.data == "init":
            data_dir, check_dir = cfg.init_dir, cfg.grasp_dir
        elif op_cfg.data == "grasp":
            data_dir, check_dir = cfg.grasp_dir, cfg.succ_grasp_dir
            if not os.path.exists(check_dir):
                check_dir = cfg.succ_traj_dir
        elif op_cfg.data == "succ_grasp":
            data_dir, check_dir = cfg.succ_grasp_dir, None
        elif op_cfg.data == "succ_traj":
            data_dir, check_dir = cfg.succ_traj_dir, None
        elif op_cfg.data == "new_tmpl":
            data_dir, check_dir = cfg.new_tmpl_dir, None
        else:
            raise NotImplementedError(
                f"Valid choices: 'init', 'grasp', 'succ_grasp', 'succ_traj', 'new_tmpl'. Current: '{op_cfg.data}' "
            )
        input_path_lst = glob(
            os.path.join(data_dir, temp_name, "**/*.npy"), recursive=True
        )
        if cfg.debug_name is not None:
            input_path_lst = [p for p in input_path_lst if cfg.debug_name in p]

        if check_dir is not None and op_cfg.succ is not None:
            check_path_lst = glob(
                os.path.join(check_dir, temp_name, "**/*.npy"), recursive=True
            )
            if cfg.debug_name is not None:
                check_path_lst = [p for p in check_path_lst if cfg.debug_name in p]
            check_path_lst = [p.replace(check_dir, data_dir) for p in check_path_lst]

            if op_cfg.succ:
                input_path_lst = list(set(input_path_lst).intersection(check_path_lst))
            else:
                input_path_lst = list(set(input_path_lst).difference(check_path_lst))

        if len(input_path_lst) == 0:
            continue
        logging.info(
            f"Find {len(input_path_lst)} in {data_dir}. Debug name: {cfg.debug_name}. Check success: {op_cfg.succ}"
        )

        if op_cfg.n_max > 0 and len(input_path_lst) > op_cfg.n_max:
            input_path_lst = np.random.permutation(input_path_lst)[: op_cfg.n_max]
        logging.info(f"Use {min(len(input_path_lst), op_cfg.n_max)}")

        param_lst = [(i, cfg) for i in input_path_lst]
        with multiprocessing.Pool(processes=cfg.n_worker) as pool:
            jobs = pool.imap_unordered(_single_vusd, param_lst)
            results = [r for r in list(jobs) if isinstance(r, dict)]

        data_length = results[0]["body_pose"].shape[0]
        save_path = os.path.join(
            cfg.vis_usd_dir, "usd", f"{temp_name.replace('**', 'all')}.usd"
        )
        usd_stage = UsdStage(save_path, timestep=len(results) * data_length, dt=0.01)

        count = 0
        for r in results:
            usd_stage.add_mesh_lst(
                r["body_mesh"],
                r["body_name"],
                r["body_pose"],
                [(count, count + data_length)] * len(r["body_name"]),
                material=Material(color=cfg.hand.color, name="obj"),
            )
            count += data_length

        logging.info(f"Save to {os.path.abspath(save_path)}")
        usd_stage.write_stage_to_file(save_path)
