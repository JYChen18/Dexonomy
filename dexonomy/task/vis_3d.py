import trimesh
import numpy as np
import os
from glob import glob
import logging
import multiprocessing

from dexonomy.sim import MuJoCo_RobotFK
from dexonomy.util.vis_util import get_arrow_mesh, get_line_mesh, scene_cfg2mesh
from dexonomy.util.file_util import load_yaml


def _single_visd(params):
    data_path, data_folder, kin, configs = (
        params[0],
        params[1],
        params[2],
        params[3],
    )
    task_config = configs.task

    out_path = os.path.join(
        configs.vis_3d_dir,
        task_config.data_type,
        os.path.relpath(data_path, data_folder),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    grasp_data = np.load(data_path, allow_pickle=True).item()

    # Object
    if task_config.data_type != "init_template":
        obj_tm = scene_cfg2mesh(grasp_data["scene_cfg"])
        if task_config.object.contact:
            point_mesh, arrow_mesh = get_arrow_mesh(
                grasp_data["obj_worldframe_contacts"][:, :3],
                grasp_data["obj_worldframe_contacts"][:, 3:],
            )
            obj_tm = trimesh.util.concatenate([obj_tm, point_mesh, arrow_mesh])
        obj_tm.export(out_path.replace(".npy", "_obj.obj"))

    # Hand
    xmat, xpos = kin.forward_kinematics(
        qpos=grasp_data["grasp_qpos"][7:], pose=grasp_data["grasp_qpos"][:7]
    )
    hand_mesh = kin.get_posed_meshes(xmat, xpos)

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
        input_path_example = os.path.join(data_folder, configs.template_name + ".npy")
        input_path_lst = glob(input_path_example)
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
        input_path_example = os.path.join(
            data_folder,
            configs.template_name,
            configs.obj_name,
            configs.data_name + ".npy",
        )
        input_path_lst = glob(input_path_example)
        if check_folder is not None and task_config.check_success is not None:
            check_path_lst = glob(input_path_example.replace(data_folder, check_folder))
            if task_config.check_success:
                input_path_lst = list(set(input_path_lst).difference(check_path_lst))
            elif not task_config.check_success:
                input_path_lst = list(set(input_path_lst).intersection(check_path_lst))
    if configs.task.max_num > 0 and len(input_path_lst) > configs.task.max_num:
        input_path_lst = np.random.permutation(input_path_lst)[: configs.task.max_num]
    logging.info(f"Find {len(input_path_lst)} data for {input_path_example}")

    kin = MuJoCo_RobotFK(configs.hand.xml_path, vis_mesh_mode=task_config.hand.mode)
    if task_config.hand.init_body:
        save_folder = os.path.join(configs.vis_3d_dir, "hand_init_body_mesh")
        os.makedirs(save_folder, exist_ok=True)
        init_mesh_name, init_mesh_lst = kin.get_init_body_meshes()
        for init_name, init_mesh in zip(init_mesh_name, init_mesh_lst):
            init_mesh.export(os.path.join(save_folder, f"{init_name}.obj"))
        logging.info(f"save hand initial body meshes to {os.path.abspath(save_folder)}")

    iterable_params = [(inp, data_folder, kin, configs) for inp in input_path_lst]

    with multiprocessing.Pool(processes=configs.n_worker) as pool:
        result_iter = pool.imap_unordered(_single_visd, iterable_params)
        results = list(result_iter)

    return
