# load successful grasp instance as initial guess for optimization
# load files in ../output/exp_name/succ_grasp/tmpl_name/obj_name/floating/*/instance_name.npy

import os
import numpy as np
import logging

from dexonomy.util.file_util import load_yaml, load_scene_cfg, safe_wrapper

@safe_wrapper
def load_instance(cfg):
    base_dir = f"{cfg.project_base_dir}/output/{cfg.exp_name}/{cfg.stage_name}/{cfg.tmpl_name}/{cfg.obj_name}/floating"
    # if scale is null
    if cfg.scale is None:
        scale_dir = os.path.join(base_dir, os.listdir(base_dir)[0])
    else:
        scale_dir = os.path.join(base_dir, f"scale{int(cfg.scale * 100):03d}")
    instance_path = os.path.join(scale_dir, cfg.instance_name + ".npy")
    try:
        data = np.load(instance_path, allow_pickle=True).item()
    except FileNotFoundError:
        raise FileNotFoundError(f"Instance file not found: {instance_path}")
    logging.info(f"Loaded instance from {instance_path}")
    logging.info(f"Instance data keys: {list(data.keys())}")
    logging.info(f"hand: {data['hand_name']}, template: {data['tmpl_name']}, object: {cfg.obj_name}")
    
    data['scene_path'] = os.path.join(cfg.project_base_dir, data['scene_path'])
    logging.debug(f"scene_path: {data['scene_path']}")
    return data

if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig
    from utils.log_util import set_logging
    from dexonomy.sim import MuJoCo_VisEnv, HandCfg, MuJoCo_OptCfg
    
    @hydra.main(config_path="config", config_name="base", version_base=None)
    def main(cfg: DictConfig):
        set_logging(cfg.verbose)
        data = load_instance(cfg)

        vis_env = MuJoCo_VisEnv(
        hand_cfg=HandCfg(xml_path=cfg.hand.xml_path, freejoint=True),
            scene_cfg=(load_scene_cfg(data["scene_path"])),
            sim_cfg=MuJoCo_OptCfg(),
            vis_mode="visual",
        )

        from scipy.spatial.transform import Rotation as R
        qpos_dummy = np.zeros(23)
        qpos_dummy[3] = 1.
        qpos_dummy[3:7] = R.from_rotvec(np.array([1, 0, 0]) * np.pi / 2).as_quat(scalar_first=True)
        xmat, xpos = vis_env.forward_kinematics(qpos_dummy)
        hand_mesh = vis_env.get_posed_mesh(xmat, xpos, body_type="hand")
        hand_mesh.export("outputs/debug_hand.obj")
        obj_tm = vis_env.get_posed_mesh(xmat, xpos, body_type="obj")
        obj_tm.export("outputs/debug_obj.obj")
        
        from utils.vis_util import points_to_obj
        points_to_obj(data["obj_cpn_w"][:, :3], 1e-2, "outputs/debug_contact.obj")

        # output qpos to fast-graspd
        qpos = data["grasp_qpos"][0]
        # move thumb's dof to the front
        thumb_qpos = qpos[-4:]
        f1_qpos = qpos[-8:-4]
        f2_qpos = qpos[-12:-8]
        f3_qpos = qpos[-16:-12]
        qpos = np.concatenate([qpos[:7], thumb_qpos, f1_qpos, f2_qpos, f3_qpos])
        
        # move quaternion's w to the back
        w = qpos[3]
        qpos[3:6] = qpos[4:7]
        qpos[6] = w
        r = R.from_quat(qpos[3:7])
        r1 = R.from_rotvec(-np.array([0, 1, 0]) * np.pi / 2)
        r2 = R.from_rotvec(np.array([0, 0, 1]) * np.pi)
        qpos[3:7] = (r * r2 * r1).as_quat()
        np.save("outputs/debug_qpos.npy", qpos)
        # load scene_cfg
        scene_cfg = load_scene_cfg(data["scene_path"])
        
    main()