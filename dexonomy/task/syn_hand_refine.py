import os
import multiprocessing
import logging
import glob

import numpy as np

from dexonomy.qp.qp_single import ContactQP
from dexonomy.sim import MuJoCo_OptEnv
from dexonomy.util.np_rot_util import np_array32
from dexonomy.util.file_util import load_scene_cfg


def _collision_filter(ho_contact_lst, hh_contact_lst, filter_config, skip_logging=True):
    if len(hh_contact_lst) > 0:
        for c in hh_contact_lst:
            if c["contact_dist"] < filter_config.hh_threshold:
                if not skip_logging:
                    logging.debug(
                        f"Collision {c['contact_dist']} between {c['body1_name']} and {c['body2_name']}"
                    )
                return False
    if len(ho_contact_lst) > 0:
        for c in ho_contact_lst:
            if c["contact_dist"] < filter_config.ho_threshold:
                if not skip_logging:
                    logging.debug(
                        f"Collision {c['contact_dist']} between {c['body1_name']} and {c['body2_name']}"
                    )
                return False
    return True


def _body_filter(ho_contact_lst, necessary_contact_body_names, filter_config):
    if not filter_config:
        return True
    curr_contact_body_set = set([c["body1_name"] for c in ho_contact_lst])
    for ncb in necessary_contact_body_names:
        if len(curr_contact_body_set.intersection(set(ncb))) == 0:
            logging.debug(
                f"Grasp stage misses contact {ncb}. Current contacts are {curr_contact_body_set}"
            )
            return False
    return True


def _qp_filter(
    contact_pos,
    contact_normal,
    ext_wrench,
    ext_center,
    filter_config,
):
    contact_wrench, wrench_error = ContactQP(miu_coef=filter_config.miu_coef).solve(
        contact_pos,
        contact_normal,
        ext_wrench,
        ext_center,
    )
    if wrench_error > filter_config.threshold:
        logging.debug(f"Grasp stage got bad QP error {wrench_error}")
        return None
    return contact_wrench


def _single_hand_refine(params):

    input_npy_path, configs = params[0], params[1]
    grasp_npy_path = input_npy_path.replace(configs.init_dir, configs.grasp_dir)
    task_config = configs.task

    grasp_data = np.load(input_npy_path, allow_pickle=True).item()

    sim_env = MuJoCo_OptEnv(
        hand_xml_path=configs.hand.xml_path,
        hand_with_arm=False,
        friction_coef=None,
        scene_cfg=load_scene_cfg(grasp_data["scene_path"]),
        debug_render=configs.debug_render,
        debug_viewer=configs.debug_viewer,
        obj_margin=task_config.pregrasp.ho_target_dist,
    )

    sim_env.set_ctrl(grasp_data["grasp_qpos"][0])
    sim_env.set_obj_margin(task_config.grasp.ho_target_dist)

    hand_contact_body_names = grasp_data["hand_contact_body_names"]
    hand_bodyframe_contact = sim_env.get_hand_bodyframe_contact(
        hand_contact_body_names, grasp_data["hand_worldframe_contacts"]
    )
    for ii in range(task_config.grasp.outer_iter):
        total_loss = sim_env.apply_force_on_hand(
            hand_contact_body_names,
            hand_bodyframe_contact,
            grasp_data["obj_worldframe_contacts"],
        )

        sim_env.simulation_step(task_config.grasp.inner_iter)

        if ii == 0 or (
            ii < task_config.grasp.outer_iter - 1
            and np.max(prev_total_loss - total_loss) > 1e-4
        ):
            prev_total_loss = np.copy(total_loss)
            continue
        else:
            prev_total_loss = np.copy(total_loss)

        ho_contact_lst, hh_contact_lst = sim_env.get_contact_info(
            task_config.grasp.contact_threshold
        )
        if not _collision_filter(
            ho_contact_lst, hh_contact_lst, task_config.grasp.coll_filter
        ):
            continue
        break

    grasp_data["grasp_qpos"] = np_array32(sim_env.get_hand_qpos())[None]

    if (
        len(ho_contact_lst) == 0
        or not _collision_filter(
            ho_contact_lst,
            hh_contact_lst,
            task_config.grasp.coll_filter,
            skip_logging=False,
        )
        or not _body_filter(
            ho_contact_lst,
            grasp_data["necessary_contact_body_names"],
            task_config.grasp.body_filter,
        )
    ):
        sim_env.debug_postprocess(
            save_path=input_npy_path.replace(
                configs.init_dir, configs.debug_dir
            ).replace(".npy", ".gif")
        )
        logging.debug(f"Fail (Grasp): {len(ho_contact_lst)} {input_npy_path}")
        return input_npy_path

    hand_point = np.array([c["contact_pos"] for c in ho_contact_lst])
    hand_normal = np.array([c["contact_normal"] for c in ho_contact_lst])
    hand_body = [c["body1_name"] for c in ho_contact_lst]
    contact_wrench = _qp_filter(
        hand_point,
        hand_normal,
        grasp_data["ext_wrench"],
        grasp_data["ext_center"],
        task_config.grasp.qp_filter,
    )
    if contact_wrench is None:
        sim_env.debug_postprocess(
            save_path=input_npy_path.replace(
                configs.init_dir, configs.debug_dir
            ).replace(".npy", ".gif")
        )
        logging.debug(f"Fail (QP): {input_npy_path}")
        return input_npy_path

    grasp_data["squeeze_qpos"] = np_array32(
        sim_env.get_squeeze_qpos(
            grasp_data["grasp_qpos"][0],
            hand_body,
            hand_point,
            10 * contact_wrench,
        )
    )[None]

    if configs.update_template == "body":
        hand_worldframe_contacts = sim_env.get_hand_worldframe_contact(
            hand_contact_body_names, hand_bodyframe_contact
        )
    elif configs.update_template == "arbi":
        hand_worldframe_contacts = np.concatenate([hand_point, hand_normal], axis=-1)
        grasp_data["hand_contact_body_names"] = hand_body
    elif configs.update_template == "nearest" or configs.update_template == False:
        hand_worldframe_contacts = sim_env.get_hand_worldframe_contact(
            hand_contact_body_names, hand_bodyframe_contact
        )
        for body_name, wf_hc in zip(hand_contact_body_names, hand_worldframe_contacts):
            for c in ho_contact_lst:
                if c["body1_name"] == body_name:
                    dist = np.linalg.norm(wf_hc[:3] - c["contact_pos"])
                    angle = np.arccos(
                        np.clip(
                            (wf_hc[3:] * c["contact_normal"]).sum(), a_min=-1, a_max=1
                        )
                    )
                    if dist < 0.03 and angle < np.pi / 4:
                        wf_hc[:3] = c["contact_pos"]
                        wf_hc[3:] = c["contact_normal"]
                        break
    else:
        raise NotImplementedError(
            f"Undefined update_template strategy: {configs.update_template}. Available choices: [False, 'nearest', 'body', 'arbi']"
        )
    grasp_data["hand_worldframe_contacts"] = hand_worldframe_contacts

    if task_config.pregrasp:
        pregrasp_lst = []
        step_group_num = 2
        for ii in range(task_config.pregrasp.outer_iter):
            curr_ho_margin = task_config.pregrasp.ho_target_dist * min(
                (ii + 1) / task_config.pregrasp.outer_iter * 2, 1
            )
            sim_env.set_obj_margin(curr_ho_margin)
            sim_env.keep_hand_stable()
            sim_env.simulation_step(task_config.grasp.inner_iter)
            pregrasp_lst.append(np.copy(sim_env.get_hand_qpos()))
            if (
                ii % step_group_num == 0
                and curr_ho_margin >= task_config.pregrasp.ho_target_dist
            ):
                ho_contact_lst, hh_contact_lst = sim_env.get_contact_info()

                if not _collision_filter(
                    ho_contact_lst, hh_contact_lst, task_config.pregrasp.coll_filter
                ):
                    continue
                break

        if not _collision_filter(
            ho_contact_lst,
            hh_contact_lst,
            task_config.pregrasp.coll_filter,
            skip_logging=False,
        ):
            sim_env.debug_postprocess(
                save_path=input_npy_path.replace(
                    configs.init_dir, configs.debug_dir
                ).replace(".npy", ".gif")
            )
            logging.debug(f"Fail (Pregrasp): {input_npy_path}")
            return input_npy_path

        grasp_data["pregrasp_qpos"] = np_array32(np.stack(pregrasp_lst, axis=0))[
            ::-step_group_num
        ]

    sim_env.debug_postprocess(
        save_path=input_npy_path.replace(configs.init_dir, configs.debug_dir).replace(
            ".npy", ".gif"
        )
    )

    os.makedirs(os.path.dirname(grasp_npy_path), exist_ok=True)
    grasp_data["evolution_num"] += 1
    np.save(grasp_npy_path, grasp_data)
    logging.debug(f"save to {grasp_npy_path}")

    return input_npy_path


def task_syn_hand(configs):
    input_path_lst = glob.glob(
        os.path.join(configs.init_dir, "**/*.npy"), recursive=True
    )
    if configs.debug_name is not None:
        input_path_lst = [p for p in input_path_lst if configs.debug_name in p]

    logged_paths = []
    if configs.skip and os.path.exists(configs.log_path):
        with open(configs.log_path, "r") as f:
            logged_paths = f.readlines()

        logged_paths = [p.split("\n")[0] for p in logged_paths]
        input_path_lst = list(set(input_path_lst).difference(set(logged_paths)))

    if len(input_path_lst) == 0:
        return

    logging.info(f"Find {len(input_path_lst)} initialization")

    iterable_params = zip(input_path_lst, [configs] * len(input_path_lst))
    if configs.n_worker == 1 or configs.debug_viewer:
        for ip in iterable_params:
            _single_hand_refine(ip)
    else:
        with multiprocessing.Pool(processes=configs.n_worker) as pool:
            result_iter = pool.imap_unordered(_single_hand_refine, iterable_params)
            results = list(result_iter)
            write_mode = "a" if configs.skip else "w"
            with open(configs.log_path, write_mode) as f:
                f.write("\n".join(results) + "\n")

    logging.info(f"Finish")

    return
