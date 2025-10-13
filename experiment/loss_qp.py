import os
import numpy as np
import logging
from qp_solver import ContactQP
from dexonomy.sim import MuJoCo_OptEnv, MuJoCo_OptCfg, HandCfg
from dexonomy.util.file_util import load_scene_cfg

from utils.vis_util import points_to_obj, arrows_to_obj

class LossQP:
    def __init__(self, cfg: dict, grasp_data: dict):
        sim_env = MuJoCo_OptEnv(
            hand_cfg=HandCfg(xml_path=cfg.hand.xml_path, freejoint=True),
            scene_cfg=load_scene_cfg(grasp_data["scene_path"]),
            sim_cfg=MuJoCo_OptCfg(obj_margin=cfg.cdist),
            debug_render=cfg.debug_render,
            debug_view=cfg.debug_view,
        )
        sim_env.reset_qpos(grasp_data["grasp_qpos"][0])
        sim_env.set_obj_margin(cfg.cdist)
        self.sim_env = sim_env
        self.ho_thre = cfg.ho_thre
        
        self.ext_center = grasp_data["ext_center"]
        self.ext_wrench = grasp_data["ext_wrench"]
        
        self.contact_qp = ContactQP(cfg.miu_coeff)
      
    def gradient(self, qpos) -> float:
        # collision
        self.sim_env.reset_qpos(qpos[0])
        ho_c, _ = self.sim_env.get_contacts(self.ho_thre)
        logging.debug(ho_c["pos"].shape)
        logging.debug(ho_c["pos"])
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            points_to_obj(ho_c["pos"], 1e-2, "outputs/debug_contact.obj")
            arrows_to_obj(ho_c["pos"], ho_c["normal"], 2e-3, 1e-1, "outputs/debug_normal.obj")

        self.contact_qp.num_contact = -1 # reset
        grad, contact_wrenches, wrench_error = self.contact_qp.gradient(
            ho_c["pos"], ho_c["normal"], self.ext_wrench, self.ext_center
        )
        logging.debug(contact_wrenches.sum(axis=0))
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            arrows_to_obj(ho_c["pos"], contact_wrenches[:, :3], 2e-3, None, "outputs/debug_wrench.obj")
            scale = np.min(1./np.linalg.norm(grad, axis=1))
            arrows_to_obj(ho_c["pos"], - grad * scale * 1e-1, 2e-3, None, "outputs/debug_grad.obj")
        
        
  

    
        
if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig
    from utils.log_util import set_logging
    from loader import load_instance
    
    @hydra.main(config_path="config", config_name="base", version_base=None)
    def main(cfg: DictConfig):
        set_logging(cfg.verbose)
        data = load_instance(cfg)

        loss = LossQP(cfg, data)
        loss.gradient(data["grasp_qpos"])
    main()