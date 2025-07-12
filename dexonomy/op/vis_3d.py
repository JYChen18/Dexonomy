import trimesh
import numpy as np
import os
from glob import glob
import logging
import multiprocessing

from dexonomy.sim import MuJoCo_VisEnv, HandCfg, MuJoCo_OptCfg
from dexonomy.util.vis_util import get_arrow_mesh, get_line_mesh
from dexonomy.util.file_util import load_yaml, load_scene_cfg, safe_wrapper


@safe_wrapper
def _single_v3d(param):
    data_path, data_dir, cfg = param[0], param[1], param[2]
    op_cfg = cfg.op

    out_path = os.path.join(
        cfg.vis_3d_dir,
        op_cfg.data,
        os.path.relpath(data_path, data_dir),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    grasp_data = np.load(data_path, allow_pickle=True).item()
    vis_env = MuJoCo_VisEnv(
        hand_cfg=HandCfg(xml_path=cfg.hand.xml_path, freejoint=True),
        scene_cfg=(
            load_scene_cfg(grasp_data["scene_path"])
            if op_cfg.data != "init_tmpl"
            else None
        ),
        sim_cfg=MuJoCo_OptCfg(),
        vis_mode=op_cfg.hand.mode,
    )

    xmat, xpos = vis_env.forward_kinematics(grasp_data["grasp_qpos"][0])
    hand_mesh = vis_env.get_posed_mesh(xmat, xpos, body_type="hand")

    # Object
    if op_cfg.data != "init_tmpl":
        obj_tm = vis_env.get_posed_mesh(xmat, xpos, body_type="obj")
        if op_cfg.object.contact:
            arrow_mesh = get_arrow_mesh(grasp_data["obj_cpn_w"])
            obj_tm = trimesh.util.concatenate([obj_tm, arrow_mesh])
        obj_tm.export(out_path.replace(".npy", "_obj.obj"))

    if op_cfg.hand.contact:
        arrow_mesh = get_arrow_mesh(grasp_data["hand_cpn_w"])
        hand_mesh = trimesh.util.concatenate([hand_mesh, arrow_mesh])

    if op_cfg.hand.skeleton:
        skt_dict = load_yaml(cfg.hand_skt_path)
        skt_data, skt_bid = [], []
        for k, v in skt_dict.items():
            skt_data.extend(v)
            skt_bid.extend([vis_env.get_body_id(k)] * len(v))
        skt_data, skt_bid = np.array(skt_data), np.array(skt_bid)
        skt_xmat, skt_xpos = xmat[skt_bid], xpos[skt_bid]
        posed_sk = (
            skt_data.reshape(-1, 2, 3) @ skt_xmat.transpose(0, 2, 1)
            + skt_xpos[:, None, :]
        )
        sk_mesh = get_line_mesh(posed_sk)
        save_path = out_path.replace(".npy", "_hand_skt.obj")
        sk_mesh.export(save_path)
        logging.info(f"save to {os.path.abspath(save_path)}")

    save_path = out_path.replace(".npy", "_hand.obj")
    hand_mesh.export(save_path)
    logging.info(f"save to {os.path.abspath(save_path)}")
    return


def operate_v3d(cfg):
    op_cfg = cfg.op
    if op_cfg.data == "init_tmpl":
        data_dir = cfg.init_tmpl_dir
        input_path_lst = glob(os.path.join(data_dir, "**.npy"))
        if cfg.debug_name is not None:
            input_path_lst = [p for p in input_path_lst if cfg.debug_name in p]
    else:
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
                f"Valid choices: 'init', 'grasp', 'succ_grasp', 'succ_traj', 'init_tmpl', 'new_tmpl'. Current: '{op_cfg.data}' "
            )
        input_path_lst = glob(os.path.join(data_dir, "**/*.npy"), recursive=True)
        if cfg.debug_name is not None:
            input_path_lst = [p for p in input_path_lst if cfg.debug_name in p]
        if check_dir is not None and op_cfg.succ is not None:
            check_path_lst = glob(os.path.join(check_dir, "**/*.npy"), recursive=True)
            if cfg.debug_name is not None:
                check_path_lst = [p for p in check_path_lst if cfg.debug_name in p]
            check_path_lst = [p.replace(check_dir, data_dir) for p in check_path_lst]

            if op_cfg.succ:
                input_path_lst = list(set(input_path_lst).intersection(check_path_lst))
            else:
                input_path_lst = list(set(input_path_lst).difference(check_path_lst))
    if op_cfg.n_max > 0 and len(input_path_lst) > op_cfg.n_max:
        input_path_lst = np.random.permutation(input_path_lst)[: op_cfg.n_max]
    logging.info(
        f"Find {len(input_path_lst)} in {data_dir}. Debug name: {cfg.debug_name}. Check success: {op_cfg.succ}"
    )

    vis_env = MuJoCo_VisEnv(
        hand_cfg=HandCfg(cfg.hand.xml_path, freejoint=True),
        vis_mode=op_cfg.hand.mode,
    )
    if op_cfg.hand.init_body:
        save_dir = os.path.join(cfg.vis_3d_dir, "hand_init_body_mesh")
        os.makedirs(save_dir, exist_ok=True)
        for body_name in vis_env.get_all_body_names():
            body_mesh = vis_env.get_body_mesh(body_name)
            body_mesh.export(os.path.join(save_dir, f"{body_name}.obj"))
        logging.info(f"save hand initial body meshes to {os.path.abspath(save_dir)}")

    param_lst = [(inp, data_dir, cfg) for inp in input_path_lst]
    with multiprocessing.Pool(processes=cfg.n_worker) as pool:
        jobs = pool.imap_unordered(_single_v3d, param_lst)
        results = list(jobs)
    return
