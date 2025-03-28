import os
import multiprocessing
import logging
import glob
from copy import deepcopy
import time

import pdb
import numpy as np
import mujoco
import mujoco.viewer
import transforms3d.quaternions as tq

from util.qp_single import ContactQP
from util.mujoco_utils import build_spec_for_opt


def _check_collision(mj_model, contact_lst, coll_config, enable_logging=False):
    if len(contact_lst) > 0:
        for c in contact_lst:
            if (
                c.dist < coll_config.hh_threshold
                and "object_collision" not in mj_model.geom(c.geom1).name
                and "object_collision" not in mj_model.geom(c.geom2).name
            ):
                if enable_logging:
                    coll_bid1 = mj_model.geom(c.geom1).bodyid[0]
                    coll_bid2 = mj_model.geom(c.geom2).bodyid[0]
                    logging.debug(
                        f"Hand self collision {c.dist} between {mj_model.body(coll_bid1).name} and {mj_model.body(coll_bid2).name}"
                    )
                return False
            if c.dist < coll_config.ho_threshold and (
                "object_collision" in mj_model.geom(c.geom1).name
                or "object_collision" in mj_model.geom(c.geom2).name
            ):
                if enable_logging:
                    if "object_collision" in mj_model.geom(c.geom1).name:
                        coll_hid = mj_model.geom(c.geom2).bodyid[0]
                    else:
                        coll_hid = mj_model.geom(c.geom1).bodyid[0]
                    logging.debug(
                        f"Hand object collision {c.dist} between {mj_model.body(coll_hid).name} and {mj_model.geom(c.geom2).name}"
                    )
                return False
    return True


def _check_contact(
    mj_model,
    mj_data,
    contact_lst,
    hospec,
    grasp_data,
    obj_mass,
    grasp_config,
    update_template,
):
    final_contact_info = []
    for jj, contact in enumerate(contact_lst):
        geom_names = [
            mj_model.geom(contact.geom1).name,
            mj_model.geom(contact.geom2).name,
        ]
        body_ids = [
            mj_model.geom(contact.geom1).bodyid[0],
            mj_model.geom(contact.geom2).bodyid[0],
        ]
        body_names = [
            mj_model.body(body_ids[0]).name,
            mj_model.body(body_ids[1]).name,
        ]
        contact_dist = contact.dist
        contact_normal = contact.frame[0:3]
        contact_positions = [
            contact.pos - contact_normal * contact_dist / 2,
            contact.pos + contact_normal * contact_dist / 2,
        ]

        if "object_collision" in geom_names[0]:
            hand_id, obj_id = 1, 0
            contact_normal = -contact_normal
        elif "object_collision" in geom_names[1]:
            hand_id, obj_id = 0, 1
        else:
            continue

        final_contact_info.append(
            {
                "hand_body_name": body_names[hand_id].removeprefix(hospec.hand_prefix),
                "hand_body_id": body_ids[hand_id],
                "obj_point": contact_positions[obj_id],
                "hand_point": contact_positions[hand_id],
                "ho_normal": contact_normal,
            }
        )
    if len(final_contact_info) == 0:
        return False, grasp_data

    hand_body_ids = [c["hand_body_id"] for c in final_contact_info]
    hand_body_names = [c["hand_body_name"] for c in final_contact_info]
    hand_points = np.stack([c["hand_point"] for c in final_contact_info], axis=0)
    obj_points = np.stack([c["obj_point"] for c in final_contact_info], axis=0)
    ho_normals = np.stack([c["ho_normal"] for c in final_contact_info], axis=0)

    grasp_data["contact_number"] = len(set(hand_body_names))

    if grasp_config.body_filter:
        curr_contact_body_set = set(hand_body_names)
        for ncb in grasp_data["necessary_contact_body_names"]:
            if len(curr_contact_body_set.intersection(set(ncb))) == 0:
                logging.debug(
                    f"Grasp stage misses contact {ncb}. Current contacts are {curr_contact_body_set}"
                )
                return False, grasp_data

    if grasp_config.qp_filter:
        if grasp_config.qp_filter.force_closure:
            gravity = grasp_data["obj_gravity_direction"] * 0.0
        else:
            gravity = grasp_data["obj_gravity_direction"] * obj_mass
        contact_wrench, wrench_error = ContactQP(
            miu_coef=grasp_config.qp_filter.miu_coef
        ).solve(
            hand_points,
            ho_normals,
            gravity,
            grasp_data["obj_gravity_center"],
        )
        if wrench_error > grasp_config.qp_filter.threshold:
            logging.debug(f"Grasp stage got bad QP error {wrench_error}")
            return False, grasp_data
        grasp_data["hand_contact_wrenches"] = contact_wrench
        mj_data.qfrc_applied[:] = 0
        for hbi, hcw, hwc in zip(
            hand_body_ids,
            contact_wrench,
            np.concatenate([hand_points, ho_normals], axis=-1),
        ):
            mujoco.mj_applyFT(
                mj_model,
                mj_data,
                10 * hcw[:3],
                0 * hcw[3:],
                hwc[:3],
                hbi,
                mj_data.qfrc_applied,
            )
        # compute the actuator force
        target_qforce = deepcopy(mj_data.qfrc_applied)
        if grasp_data["hand_type"] == "shadow":
            tendon_id = [[8, 9], [12, 13], [16, 17], [21, 22]]
            for a, b in tendon_id:
                target_qforce[a] /= 2
                target_qforce[b] /= 2
        dense_actuator_moment = np.zeros((mj_model.nu, mj_model.nv))
        mujoco.mju_sparse2dense(
            dense_actuator_moment,
            mj_data.actuator_moment,
            mj_data.moment_rownnz,
            mj_data.moment_rowadr,
            mj_data.moment_colind,
        )
        actuator_gainprm = mj_model.actuator_gainprm[:, 0]
        for i in range(len(target_qforce)):
            actuator_id = np.where(dense_actuator_moment[:, i] != 0)[0]
            if len(actuator_id) > 0:
                target_qforce[i] /= actuator_gainprm[actuator_id[0]]
        grasp_data["squeeze_pose"] = grasp_data["grasp_pose"]
        grasp_data["squeeze_qpos"] = grasp_data["grasp_qpos"] + target_qforce[6:]

    if update_template == "body":
        contact_hand_links = [
            hospec.hand_prefix + b for b in grasp_data["hand_contact_body_names"]
        ]
        contact_body_ids = [mj_model.body(b).id for b in contact_hand_links]
        for i, body_id in enumerate(contact_body_ids):
            br = mj_data.xmat[body_id].reshape(3, 3)
            bp = mj_data.xpos[body_id]
            grasp_data["hand_worldframe_contacts"][i] = np.concatenate(
                [
                    br @ grasp_data["hand_bodyframe_contacts"][i, :3] + bp,
                    br @ grasp_data["hand_bodyframe_contacts"][i, 3:],
                ],
                axis=-1,
            )
    elif update_template == "arbi":
        grasp_data["hand_worldframe_contacts"] = np.concatenate(
            [hand_points, ho_normals], axis=-1
        )
        grasp_data["hand_contact_body_names"] = hand_body_names
        grasp_data["obj_worldframe_contacts"] = np.concatenate(
            [obj_points, ho_normals], axis=-1
        )
    elif update_template == "nearest" or update_template == False:
        contact_hand_links = [
            hospec.hand_prefix + b for b in grasp_data["hand_contact_body_names"]
        ]
        contact_body_ids = [mj_model.body(b).id for b in contact_hand_links]
        for i, body_id in enumerate(contact_body_ids):
            br = mj_data.xmat[body_id].reshape(3, 3)
            bp = mj_data.xpos[body_id]
            hbc = np.concatenate(
                [
                    br @ grasp_data["hand_bodyframe_contacts"][i, :3] + bp,
                    br @ grasp_data["hand_bodyframe_contacts"][i, 3:],
                ],
                axis=-1,
            )
            for c in final_contact_info:
                if c["hand_body_name"] == grasp_data["hand_contact_body_names"][i]:
                    dist = np.linalg.norm(hbc[:3] - c["hand_point"])
                    angle = np.arccos(
                        np.clip((hbc[3:] * c["ho_normal"]).sum(), a_min=-1, a_max=1)
                    )
                    if dist < 0.03:  # and angle < np.pi / 4:
                        hbc[:3] = c["hand_point"]
                        hbc[3:] = c["ho_normal"]
                        break
            grasp_data["hand_worldframe_contacts"][i] = hbc
    else:
        raise NotImplementedError(
            f"Undefined update_template strategy: {update_template}. Available choices: [False, 'nearest', 'body', 'arbi']"
        )
    return True, grasp_data


def _mj_simulate(
    mj_model,
    mj_data,
    hospec,
    grasp_data,
    task_name,
    grasp_config,
    debug_view,
    override_ogd,
    outer_iter=20,
    inner_iter=10,
):
    if debug_view:
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
        viewer.sync()
        pdb.set_trace()

    contact_hand_links = [
        hospec.hand_prefix + b for b in grasp_data["hand_contact_body_names"]
    ]
    contact_body_ids = [mj_model.body(b).id for b in contact_hand_links]

    for ii in range(outer_iter):
        mj_data.qfrc_applied[:] = 0
        mj_data.xfrc_applied[:] = 0

        if task_name == "grasp":
            total_loss = []
            for i, body_id in enumerate(contact_body_ids):
                br = mj_data.xmat[body_id].reshape(3, 3)
                bp = mj_data.xpos[body_id]
                hcp_body = grasp_data["hand_bodyframe_contacts"][i, :3]
                hcp_world = br @ hcp_body + bp
                ocp_world = grasp_data["obj_worldframe_contacts"][i, :3]
                total_loss.append(np.linalg.norm(ocp_world - hcp_world))
                point_force = (ocp_world - hcp_world) * 500
                mujoco.mj_applyFT(
                    mj_model,
                    mj_data,
                    point_force,
                    0 * point_force,
                    hcp_world,
                    body_id,
                    mj_data.qfrc_applied,
                )
        _, joint_qpos, _ = hospec.split_qpos_pose(mj_data.qpos)
        mj_data.ctrl = hospec.qpos_to_ctrl(joint_qpos)
        mj_data.qvel[:] = 0

        mujoco.mj_forward(mj_model, mj_data)
        for _ in range(inner_iter):
            mujoco.mj_step(mj_model, mj_data)

        if debug_view:
            viewer.sync()
            pdb.set_trace()

        # Early-stop optimization
        if task_name == "grasp":
            if ii == 0 or np.max(prev_total_loss - total_loss) > 1e-4:
                prev_total_loss = deepcopy(np.array(total_loss))
                continue
            else:
                prev_total_loss = deepcopy(np.array(total_loss))

        if not _check_collision(mj_model, mj_data.contact, grasp_config.coll_filter):
            continue
        break

    if debug_view:
        viewer.close()

    if task_name == "grasp":
        grasp_pose, grasp_qpos, _ = hospec.split_qpos_pose(mj_data.qpos)
        if override_ogd:
            grasp_data["obj_gravity_direction"] = np.array([0.0, 0, -1, 0, 0, 0])
        else:
            new_hr = tq.quat2mat(grasp_pose[3:])
            old_hr = tq.quat2mat(grasp_data["grasp_pose"][3:])
            grasp_data["obj_gravity_direction"][:3] = (
                new_hr @ old_hr.T @ (grasp_data["obj_gravity_direction"][:3])
            )
            grasp_data["obj_gravity_direction"][3:] = (
                new_hr @ old_hr.T @ (grasp_data["obj_gravity_direction"][3:])
            )
        grasp_data["grasp_qpos"] = deepcopy(grasp_qpos)
        grasp_data["grasp_pose"] = deepcopy(grasp_pose)
    else:
        grasp_data["pregrasp_pose"], grasp_data["pregrasp_qpos"], _ = (
            hospec.split_qpos_pose(mj_data.qpos)
        )

    return grasp_data


def _single_mjopt(params):

    input_npy_path, configs = params[0], params[1]
    grasp_npy_path = input_npy_path.replace(configs.init_dir, configs.grasp_dir)

    task_config = configs.task
    hand_config = configs.hand

    grasp_data = np.load(input_npy_path, allow_pickle=True).item()

    hospec = build_spec_for_opt(
        obj_path=grasp_data["obj_path"],
        obj_pose=grasp_data["obj_pose"],
        obj_scale=grasp_data["obj_scale"],
        has_floor_z0=configs.has_floor_z0,
        hand_xml_path=hand_config.xml_path,
        grasp_pose=grasp_data["grasp_pose"],
        grasp_qpos=grasp_data["grasp_qpos"],
        hand_tendon=hand_config.tendon,
        ho_margin=task_config.grasp.ho_margin,
        ho_target_dist=task_config.grasp.ho_target_dist,
    )

    # Get ready for simulation
    mj_model = hospec.spec.compile()
    mj_data = mujoco.MjData(mj_model)

    if configs.debug or configs.debug_view:
        with open("debug.xml", "w") as f:
            f.write(hospec.spec.to_xml())

    # Initialize hand pose by setting keyframe
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, mj_model.nkey - 1)
    mujoco.mj_forward(mj_model, mj_data)

    grasp_data = _mj_simulate(
        mj_model=mj_model,
        mj_data=mj_data,
        hospec=hospec,
        grasp_data=grasp_data,
        task_name="grasp",
        grasp_config=task_config.grasp,
        debug_view=configs.debug_view,
        override_ogd=configs.has_floor_z0 and configs.override_ogd,
        outer_iter=task_config.grasp.outer_iter,
        inner_iter=task_config.grasp.inner_iter,
    )

    if not _check_collision(
        mj_model, mj_data.contact, task_config.grasp.coll_filter, enable_logging=True
    ):
        return input_npy_path

    contact_flag, grasp_data = _check_contact(
        mj_model,
        mj_data,
        mj_data.contact,
        hospec,
        grasp_data,
        configs.obj_mass,
        task_config.grasp,
        configs.update_template,
    )
    if not contact_flag:
        return input_npy_path

    # os.makedirs(os.path.dirname(grasp_npy_path), exist_ok=True)
    # np.save(grasp_npy_path, grasp_data)
    # logging.info(f"save to {grasp_npy_path}")

    if task_config.pregrasp:
        # change margin and gap
        for i in range(mj_model.ngeom):
            if "object_collision" in mj_model.geom(i).name:
                mj_model.geom_margin[i] = task_config.pregrasp.ho_margin
                mj_model.geom_gap[i] = (
                    task_config.pregrasp.ho_margin - task_config.pregrasp.ho_target_dist
                )

        grasp_data = _mj_simulate(
            mj_model=mj_model,
            mj_data=mj_data,
            hospec=hospec,
            grasp_data=grasp_data,
            task_name="pregrasp",
            grasp_config=task_config.pregrasp,
            debug_view=configs.debug_view,
            override_ogd=configs.has_floor_z0 and configs.override_ogd,
            outer_iter=task_config.pregrasp.outer_iter,
            inner_iter=task_config.pregrasp.inner_iter,
        )

        if not _check_collision(
            mj_model,
            mj_data.contact,
            task_config.pregrasp.coll_filter,
            enable_logging=True,
        ):
            return input_npy_path

    os.makedirs(os.path.dirname(grasp_npy_path), exist_ok=True)
    grasp_data["evolution_num"] += 1
    np.save(grasp_npy_path, grasp_data)
    # logging.info(f"save to {grasp_npy_path}")

    return input_npy_path


def task_syn_hand(configs):
    input_path_lst = glob.glob(
        os.path.join(
            configs.init_dir, configs.debug_template, configs.debug_obj, "**.npy"
        )
    )

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
    if configs.debug or configs.debug_view:
        for ip in iterable_params:
            _single_mjopt(ip)
    else:
        with multiprocessing.Pool(processes=configs.n_worker) as pool:
            result_iter = pool.imap_unordered(_single_mjopt, iterable_params)
            results = list(result_iter)
            write_mode = "a" if configs.skip else "w"
            with open(configs.log_path, write_mode) as f:
                f.write("\n".join(results) + "\n")

    logging.info(f"Finish")

    return
