import os


def operate_traj(cfg):
    grasp_dir = cfg.grasp_dir if cfg.skip_grasp_eval else cfg.succ_grasp_dir
    traj_cmd = f"cd third_party/BODex && DISABLE_GRASP_SYN=true python example_grasp/multi_gpu.py \
        -t mogen_dexonomy \
        -c {cfg.hand_name}/dexonomy.yml \
        -g {' '.join([str(g) for g in cfg.traj_gpu])} \
        -e {os.path.abspath(cfg.traj_dir)} \
        -i '{os.path.abspath(grasp_dir)}/**/*.npy' "
    if not cfg.skip_done:
        traj_cmd += " -k"
    os.system(traj_cmd)
    return
