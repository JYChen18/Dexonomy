import os
import multiprocessing
import logging
import glob
import numpy as np
from hydra.utils import to_absolute_path

from dexonomy.sim import MuJoCo_EvalEnv, HandCfg, MuJoCo_EvalCfg
from dexonomy.util.file_util import load_scene_cfg, load_json, safe_wrapper
from dexonomy.util.traj_util import get_planner


@safe_wrapper
def _single_eval(param):
    input_path, cfg = param[0], param[1]
    data_dir = cfg.traj_dir if not cfg.skip_traj else cfg.grasp_dir
    op_cfg, hand_cfg = cfg.op, cfg.hand
    grasp_data = np.load(input_path, allow_pickle=True).item()
    
    if cfg.skip_pregrasp:
        # start from grasp pose rather than pre grasp pose
        grasp_data["pregrasp_qpos"] = grasp_data["grasp_qpos"].copy()
    
    if cfg.legacy_api:
        if "evolution_num" in grasp_data:
            key_map = {"evolution_num": "n_evo",
                      "hand_type": "hand_name",
                      "hand_template_name": "tmpl_name",
                      "hand_worldframe_contacts": "hand_cpn_w",
                      "hand_contact_body_names": "hand_cbody",
                      "necessary_contact_body_names": "required_cbody",
                      "obj_worldframe_contacts": "obj_cpn_w"
                      }
            grasp_data = {key_map.get(k, k): v for k, v in grasp_data.items()}
            if 'grasp_qpos' in grasp_data:
                grasp_data['grasp_qpos'] = grasp_data['grasp_qpos'][None,...]
            if 'squeeze_qpos' in grasp_data:
                grasp_data['squeeze_qpos'] = grasp_data['squeeze_qpos'][None,...]
            if 'pregrasp_qpos' in grasp_data:
                grasp_data['pregrasp_qpos'] = grasp_data['pregrasp_qpos'][None,...]
            # scene_cfg change to absolute path recursively
            def update_relative_path(d: dict):
                for k, v in d.items():
                    if isinstance(v, dict):
                        update_relative_path(v)
                    elif k.endswith("_path") and isinstance(v, str):
                        d[k] = os.path.abspath(to_absolute_path(v))
                return
            update_relative_path(grasp_data['scene_cfg'])
        scene_cfg = grasp_data['scene_cfg']
    else:
        scene_cfg = load_scene_cfg(grasp_data["scene_path"])

    plane_flag = False
    for o_cfg in scene_cfg["scene"].values():
        if o_cfg["type"] == "rigid_object":
            if cfg.legacy_api:
                info_path = o_cfg["urdf_path"].replace("urdf/coacd.urdf", "info/simplified.json")
                o_info = load_json(info_path)
            else:
                o_info = load_json(o_cfg["info_path"])
            o_coef = o_info["mass"] / (o_info["density"] * (o_info["scale"] ** 3))
            o_cfg["density"] = cfg.obj_mass / (o_coef * np.prod(o_cfg["scale"]))
        elif o_cfg["type"] == "plane":
            plane_flag = True
    if not cfg.skip_traj and not plane_flag:
        if cfg.legacy_api:
            logging.warning(
              f"Using an arm but no table is in scene cfg: {grasp_data['scene_cfg']}"
            )
        else:
            logging.warning(
                f"Using an arm but no table is in scene cfg: {grasp_data['scene_path']}"
            )

    sim_env = MuJoCo_EvalEnv(
        hand_cfg=(
            HandCfg(arm_flag=True, **hand_cfg.arm_hand)
            if not cfg.skip_traj
            else HandCfg(hand_cfg.xml_path, freejoint=True)
        ),
        scene_cfg=scene_cfg,
        sim_cfg=MuJoCo_EvalCfg(miu_coef=op_cfg.miu_coef),
        debug_render=cfg.debug_render,
        debug_view=cfg.debug_view,
        hard=op_cfg.hard,
    )

    init_obj_pose = np.copy(sim_env.get_active_obj_pose())
    planner = get_planner(scene_cfg["task"])
    ctrl_qpos, ctrl_type, ext_dir, target_obj_pose = planner.plan_trajectory(
        init_obj_pose,
        grasp_data["pregrasp_qpos"],
        grasp_data["grasp_qpos"],
        grasp_data["squeeze_qpos"],
        grasp_data["robot_pose"] if "robot_pose" in grasp_data else None,
    )

    if scene_cfg["task"]["type"] == "force_closure":
        eval_func = sim_env.eval_fc
    else:
        eval_func = sim_env.eval_move

    succ_flag, real_state_qpos, real_ctrl_qpos = eval_func(
        ctrl_qpos,
        ctrl_type,
        ext_dir,
        target_obj_pose,
        op_cfg.trans_thre,
        op_cfg.rot_thre,
    )

    sim_env.save_debug(input_path.replace(data_dir, cfg.debug_dir))

    if succ_flag:
        if not cfg.skip_traj:
            output_path = input_path.replace(data_dir, cfg.succ_traj_dir)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if cfg.legacy_api:
                np.save(
                    output_path,
                    {
                        "scene_cfg": grasp_data["scene_cfg"],
                        "traj_state_qpos": real_state_qpos,
                        "traj_ctrl_qpos": real_ctrl_qpos,
                    },
                )
            else:
                np.save(
                    output_path,
                    {
                        "scene_path": grasp_data["scene_path"],
                        "traj_state_qpos": real_state_qpos,
                        "traj_ctrl_qpos": real_ctrl_qpos,
                    },
                )
        else:
            output_path = input_path.replace(data_dir, cfg.succ_grasp_dir)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            os.system(
                f"ln -s {os.path.relpath(input_path, os.path.dirname(output_path))} {output_path}"
            )

        if cfg.tmpl_upd_mode != "disabled" and grasp_data["n_evo"] <= cfg.tmpl_max_evo:
            tmpl_path = input_path.replace(data_dir, cfg.new_tmpl_dir)
            if not os.path.exists(tmpl_path):
                original_path = input_path.replace(data_dir, cfg.grasp_dir)
                os.makedirs(os.path.dirname(tmpl_path), exist_ok=True)
                os.system(
                    f"ln -s {os.path.relpath(original_path, os.path.dirname(tmpl_path))} {tmpl_path}"
                )

    return input_path


def operate_eval(cfg):
    data_dir = cfg.traj_dir if not cfg.skip_traj else cfg.grasp_dir
    input_path_lst = glob.glob(os.path.join(data_dir, "**/*.npy"), recursive=True)
    if cfg.debug_name is not None:
        input_path_lst = [p for p in input_path_lst if cfg.debug_name in p]

    logged_path_lst = []
    if cfg.skip_done and os.path.exists(cfg.log_path):
        with open(cfg.log_path, "r") as f:
            logged_path_lst = f.readlines()
        logged_path_lst = [p.split("\n")[0] for p in logged_path_lst]
        input_path_lst = list(set(input_path_lst).difference(set(logged_path_lst)))

    logging.info(f"Find {len(input_path_lst)} grasp data")

    if len(input_path_lst) == 0:
        return

    param_lst = zip(input_path_lst, [cfg] * len(input_path_lst))
    if cfg.n_worker == 1 or cfg.debug_view:
        for ip in param_lst:
            _single_eval(ip)
    else:
        with multiprocessing.Pool(processes=3 * cfg.n_worker) as pool:
            jobs = pool.imap_unordered(_single_eval, param_lst)
            results = list(jobs)
            write_mode = "a" if cfg.skip_done else "w"
            with open(cfg.log_path, write_mode) as f:
                f.write("\n".join(results) + "\n")

    logging.info(f"Finish evaluation")

    return
