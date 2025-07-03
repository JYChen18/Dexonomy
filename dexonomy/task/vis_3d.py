import trimesh
import numpy as np
import os
from glob import glob
import logging
import multiprocessing

from dexonomy.sim import MuJoCo_VisEnv, HandCfg, MuJoCo_OptCfg
from dexonomy.util.vis_util import get_arrow_mesh, get_line_mesh
from dexonomy.util.file_util import load_yaml, load_scene_cfg


def _single_visd(params):
    data_path, data_folder, configs = (
        params[0],
        params[1],
        params[2],
    )
    task_config = configs.task

    out_path = os.path.join(
        configs.vis_3d_dir,
        task_config.data_type,
        os.path.relpath(data_path, data_folder),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    grasp_data = np.load(data_path, allow_pickle=True).item()
    kin = MuJoCo_VisEnv(
        hand_cfg=HandCfg(xml_path=configs.hand.xml_path, freejoint=True),
        scene_cfg=(
            load_scene_cfg(grasp_data["scene_path"])
            if task_config.data_type != "init_template"
            else None
        ),
        sim_cfg=MuJoCo_OptCfg(),
        vis_mesh_mode=task_config.hand.mode,
    )

    xmat, xpos = kin.forward_kinematics(grasp_data["grasp_qpos"][0])
    hand_mesh = kin.get_posed_meshes(xmat, xpos, vis_prefix=[kin.sim_cfg.hand_prefix])

    # Object
    if task_config.data_type != "init_template":
        obj_tm = kin.get_posed_meshes(
            xmat, xpos, vis_prefix=[kin.sim_cfg.obj_prefix, kin.sim_cfg.plane_prefix]
        )
        if task_config.object.contact:
            point_mesh, arrow_mesh = get_arrow_mesh(
                grasp_data["obj_worldframe_contacts"][:, :3],
                grasp_data["obj_worldframe_contacts"][:, 3:],
            )
            obj_tm = trimesh.util.concatenate([point_mesh, arrow_mesh])
        obj_tm.export(out_path.replace(".npy", "_obj.obj"))

    if task_config.hand.contact:
        point_mesh, arrow_mesh = get_arrow_mesh(
            grasp_data["hand_worldframe_contacts"][:, :3],
            grasp_data["hand_worldframe_contacts"][:, 3:],
        )
        hand_mesh = trimesh.util.concatenate([hand_mesh, point_mesh, arrow_mesh])

    if task_config.hand.skeleton:
        hand_skeleton_dict = load_yaml(configs.hand_skeleton_path)
        hand_skeleton = []
        hand_sk_body_id = []
        for k, v in hand_skeleton_dict.items():
            hand_skeleton.extend(v)
            hand_sk_body_id.extend([kin.body_id_dict[k]] * len(v))
        hand_skeleton = np.array(hand_skeleton)
        hand_sk_body_id = np.array(hand_sk_body_id)
        body_xmat = xmat[hand_sk_body_id]
        body_xpos = xpos[hand_sk_body_id]
        posed_sk = (
            hand_skeleton.reshape(-1, 2, 3) @ body_xmat.transpose(0, 2, 1)
            + body_xpos[:, None, :]
        )
        sk_mesh = get_line_mesh(posed_sk)
        save_path = out_path.replace(".npy", "_handskeleton.obj")
        sk_mesh.export(save_path)
        logging.info(f"save to {os.path.abspath(save_path)}")

    save_path = out_path.replace(".npy", "_hand.obj")
    hand_mesh.export(save_path)
    logging.info(f"save to {os.path.abspath(save_path)}")
    return


def task_vis_3d(configs):
    task_config = configs.task
    if task_config.data_type == "init_template":
        data_folder = configs.init_template_dir
        input_path_lst = glob(os.path.join(data_folder, "**.npy"))
        if configs.debug_name is not None:
            input_path_lst = [p for p in input_path_lst if configs.debug_name in p]
    else:
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
        input_path_lst = glob(os.path.join(data_folder, "**/*.npy"), recursive=True)
        if configs.debug_name is not None:
            input_path_lst = [p for p in input_path_lst if configs.debug_name in p]

        if check_folder is not None and task_config.check_success is not None:
            check_path_lst = glob(
                os.path.join(check_folder, "**/*.npy"), recursive=True
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
    if configs.task.max_num > 0 and len(input_path_lst) > configs.task.max_num:
        input_path_lst = np.random.permutation(input_path_lst)[: configs.task.max_num]
    logging.info(
        f"Find {len(input_path_lst)} in {data_folder}. Debug name: {configs.debug_name}. Check success: {task_config.check_success}"
    )

    kin = MuJoCo_VisEnv(
        hand_cfg=HandCfg(xml_path=configs.hand.xml_path, freejoint=True),
        vis_mesh_mode=task_config.hand.mode,
    )
    if task_config.hand.init_body:
        save_folder = os.path.join(configs.vis_3d_dir, "hand_init_body_mesh")
        os.makedirs(save_folder, exist_ok=True)
        init_mesh_name, init_mesh_lst = kin.get_init_body_meshes()
        for init_name, init_mesh in zip(init_mesh_name, init_mesh_lst):
            init_mesh.export(os.path.join(save_folder, f"{init_name}.obj"))
        logging.info(f"save hand initial body meshes to {os.path.abspath(save_folder)}")

    iterable_params = [(inp, data_folder, configs) for inp in input_path_lst]

    with multiprocessing.Pool(processes=configs.n_worker) as pool:
        result_iter = pool.imap_unordered(_single_visd, iterable_params)
        results = list(result_iter)

    return
