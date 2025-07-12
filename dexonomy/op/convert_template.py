import os
from glob import glob
import logging
import multiprocessing

import numpy as np

from dexonomy.util.file_util import load_yaml, safe_wrapper
from dexonomy.sim import MuJoCo_VisEnv, HandCfg
from dexonomy.util.np_util import (
    np_transform_points,
    np_array32,
    np_inv_transform_points,
)


@safe_wrapper
def _single_anno2tmpl(param):
    anno_path, hand_kp, hand_bgroup, cfg = param[0], param[1], param[2], param[3]
    anno_data = load_yaml(anno_path)
    qpos = np_array32(anno_data["qpos"])
    vis_env = MuJoCo_VisEnv(hand_cfg=HandCfg(cfg.hand.xml_path), vis_mode="collision")
    xmat, xpos = vis_env.forward_kinematics(qpos)

    if "contact" not in anno_data or anno_data["contact"] is None:
        anno_data["contact"] = {}

    # Convert dict to list
    contact_lst = []
    for b_name, c_anno in anno_data["contact"].items():
        if isinstance(c_anno, list):
            # Judge whether it is a list of points or a single point list
            if len(c_anno) == 6 and sum([isinstance(c, float) for c in c_anno]) > 0:
                contact_lst.append((b_name, c_anno))
            else:
                for c in c_anno:
                    contact_lst.append((b_name, c))
        else:
            contact_lst.append((b_name, c_anno))

    # Process contact list
    hand_cpn_w, hand_cbody = [], []
    for b_name, c_anno in contact_lst:
        b_mesh, b_id = vis_env.get_body_mesh(b_name), vis_env.get_body_id(b_name)
        b_mat, b_pos = xmat[b_id], xpos[b_id]
        if isinstance(c_anno, list):  # Annotated directly in handframe
            assert len(c_anno) == 6
            h_cpn_w = np_array32(c_anno)
            h_cpn_b = np_inv_transform_points(h_cpn_w, b_mat, b_pos)
        else:  # Annotated in bodyframe via keypoints
            h_cpn_b = np_array32(hand_kp[b_name][c_anno])
            h_cpn_w = np_transform_points(h_cpn_b, b_mat, b_pos)

        _, dist, _ = b_mesh.nearest.on_surface([h_cpn_b[:3]])
        if dist > 0.002:
            min_b_name = b_name
            min_dist = dist
            for bn in vis_env.get_all_body_names():
                new_b_id = vis_env.get_body_id(bn)
                new_b_mesh = vis_env.get_body_mesh(bn)
                new_b_mat, new_b_pos = xmat[new_b_id], xpos[new_b_id]
                new_h_cpn_b = np_inv_transform_points(h_cpn_w, new_b_mat, new_b_pos)
                _, new_dist, _ = new_b_mesh.nearest.on_surface([new_h_cpn_b[:3]])
                if new_dist < min_dist:
                    min_dist = new_dist
                    min_b_name = bn
            if min_dist > 0.002:
                error_str = (
                    "The annotated contact point is far from all collision meshes!\n"
                )
            else:
                error_str = "The annotated contact point seems to be inconsistent with the body name!\n"
            logging.error(
                f"{error_str}\n \
                    Template path: {anno_path}\n \
                    Annotated body name: {b_name}, Point2body distance: {dist};\n \
                    Nearest body name: {min_b_name}, Point2body distance: {min_dist}.\n"
            )
        hand_cpn_w.append(h_cpn_w)
        hand_cbody.append(b_name)

    if "required_cbody" not in anno_data:
        required_cbody = []
        for i in hand_bgroup:
            required_cbody.append([])
            for j in i:
                if j in anno_data["contact"].keys():
                    required_cbody[-1].append(j)
            if len(required_cbody[-1]) == 0:
                required_cbody.pop(-1)
    else:
        required_cbody = anno_data["required_cbody"]
        check_lst = []
        for i in required_cbody:
            for j in i:
                assert j in anno_data["contact"].keys()
                assert j not in check_lst
                check_lst.append(j)

    tmpl_data = {
        "tmpl_name": os.path.basename(anno_path).removesuffix(".yaml"),
        "grasp_qpos": np.concatenate([np_array32([0, 0, 0, 1, 0, 0, 0]), qpos])[None],
        "hand_cpn_w": np.stack(hand_cpn_w, axis=0) if len(hand_cpn_w) > 0 else None,
        "hand_cbody": hand_cbody,
        "required_cbody": required_cbody,
        "n_evo": np_array32([0]),
    }
    tmpl_path = anno_path.replace(cfg.raw_anno_dir, cfg.init_tmpl_dir)
    np.save(tmpl_path.replace(".yaml", ".npy"), tmpl_data)
    return


def operate_tmpl(cfg):
    input_path_lst = glob(os.path.join(cfg.raw_anno_dir, "**.yaml"))
    if cfg.debug_name is not None:
        input_path_lst = [p for p in input_path_lst if cfg.debug_name in p]
    n_input = len(input_path_lst)
    logging.info(f"Find {n_input} annotations")

    if n_input == 0:
        return

    os.makedirs(cfg.init_tmpl_dir, exist_ok=True)
    hand_bgroup = load_yaml(cfg.hand_bgroup_path)["body_group"]

    hand_kp = None
    if os.path.exists(cfg.hand_kp_path):
        hand_kp = load_yaml(cfg.hand_kp_path)
        if hand_kp is not None:
            for k, v in hand_kp.items():
                if isinstance(v, str):
                    hand_kp[k] = hand_kp[v]

    param_lst = zip(
        input_path_lst, [hand_kp] * n_input, [hand_bgroup] * n_input, [cfg] * n_input
    )
    if cfg.n_worker == 1:
        for ip in param_lst:
            _single_anno2tmpl(ip)
    else:
        with multiprocessing.Pool(processes=cfg.n_worker) as pool:
            jobs = pool.imap_unordered(_single_anno2tmpl, param_lst)
            results = list(jobs)

    logging.info(f"Finish template conversion")

    return
