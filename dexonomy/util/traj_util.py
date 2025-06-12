import numpy as np

from dexonomy.util.np_rot_util import (
    np_normalize_vector,
    np_interp_hinge,
    np_interp_slide,
)


def get_full_traj(
    init_obj_pose,
    move_cfg,
    grasp_qpos,
    squeeze_qpos,
    pregrasp_qpos,
    approach_qpos=None,
    move_step=10,
):
    qpos_lst = []
    interp_lst = []
    extdir_lst = []
    ctype_lst = []
    if approach_qpos is not None:
        qpos_lst.extend(list(approach_qpos))
        approach_num = approach_qpos.shape[0] - 1
        interp_lst.extend([3 if i % 5 == 4 else 1 for i in range(approach_num)])
        extdir_lst.extend([None] * approach_num)
        ctype_lst.extend(["joint_angle"] * approach_num)

    interp_lst.extend([3] * pregrasp_qpos.shape[0])
    extdir_lst.extend([None] * pregrasp_qpos.shape[0])
    ctype_lst.extend(["ee_pose"] * pregrasp_qpos.shape[0])
    qpos_lst.extend(list(pregrasp_qpos))

    interp_lst.extend([3])
    extdir_lst.extend([None])
    ctype_lst.extend(["ee_pose"])
    qpos_lst.extend(list(grasp_qpos))
    interp_lst.extend([10])
    extdir_lst.extend([None])
    ctype_lst.extend(["ee_pose"])
    qpos_lst.extend(list(squeeze_qpos))

    if move_cfg["type"] != "force_closure":
        move_qpos = np.repeat(squeeze_qpos, move_step, axis=0)
        move_qpos[:, :7] = interp_move_traj(squeeze_qpos[0, :7], move_cfg, move_step)
        qpos_lst.extend(list(move_qpos))
        interp_lst.extend([3] * move_qpos.shape[0])
        ctype_lst.extend(["ee_pose"] * move_qpos.shape[0])
        move_obj_lst = interp_move_traj(init_obj_pose, move_cfg, move_step)
        target_obj_pose = move_obj_lst[-1]
        if move_cfg["type"] == "slide":
            extdir_lst.extend([-move_cfg["axis"]] * move_qpos.shape[0])
        elif move_cfg["type"] == "hinge":
            extdir_lst.extend(
                [
                    -np.cross(
                        move_cfg["axis"], np_normalize_vector(p[:3] - move_cfg["pos"])
                    )
                    for p in move_obj_lst
                ]
            )
    else:
        target_obj_pose = init_obj_pose
    return qpos_lst, ctype_lst, extdir_lst, interp_lst, target_obj_pose


def interp_move_traj(init_pose: np.ndarray, move_cfg: dict, step: int):
    if move_cfg["type"] == "slide":
        target_pose = np.copy(init_pose)
        target_pose[:3] += move_cfg["axis"] * move_cfg["distance"]
        move_pose_lst = np_interp_slide(init_pose, target_pose, step)
    elif move_cfg["type"] == "hinge":
        move_pose_lst = np_interp_hinge(
            pose1=init_pose,
            hinge_pos=move_cfg["pos"],
            hinge_axis=move_cfg["axis"],
            move_angle=move_cfg["distance"],
            step=step,
        )
    elif move_cfg["type"] != "force_closure":
        raise NotImplementedError(
            f"Unsupported task type: {move_cfg['type']}. Avaiable choices: 'hinge', 'slide', 'force_closure'."
        )
    return move_pose_lst
