# load successful grasp instance as initial guess for optimization
# load files in ../output/exp_name/succ_grasp/tmpl_name/obj_name/floating/*/instance_name.npy

import os
import numpy as np
import logging

from dexonomy.util.file_util import load_yaml, load_scene_cfg, safe_wrapper

@safe_wrapper
def load_instance(cfg):
    base_dir = f"{cfg.project_base_dir}/output/{cfg.exp_name}/succ_grasp/{cfg.tmpl_name}/{cfg.obj_name}/floating"
    scale_dir = os.path.join(base_dir, os.listdir(base_dir)[0])
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

        xmat, xpos = vis_env.forward_kinematics(data["grasp_qpos"][0])
        hand_mesh = vis_env.get_posed_mesh(xmat, xpos, body_type="hand")
        hand_mesh.export("outputs/debug_hand.obj")
        obj_tm = vis_env.get_posed_mesh(xmat, xpos, body_type="obj")
        obj_tm.export("outputs/debug_obj.obj")
        
        from utils.vis_util import points_to_obj
        points_to_obj(data["obj_cpn_w"][:, :3], 1e-2, "outputs/debug_contact.obj")


    main()