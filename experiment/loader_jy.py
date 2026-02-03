# load successful grasp instance as initial guess for optimization
# load files in ../output/exp_name/succ_grasp/tmpl_name/obj_name/floating/*/instance_name.npy

import os
import numpy as np
import logging
from hydra.utils import to_absolute_path

from dexonomy.util.file_util import load_yaml, load_scene_cfg, safe_wrapper

@safe_wrapper
def load_instance_legacy(cfg):
    if cfg.stage_name == 'init_data':
        base_dir = f"{cfg.project_base_dir}/output/{cfg.exp_name}/{cfg.stage_name}/{cfg.tmpl_name}/floating_{cfg.obj_name}"
    else:
        base_dir = f"{cfg.project_base_dir}/output/{cfg.exp_name}/{cfg.stage_name}/{cfg.tmpl_name}/{cfg.obj_name}_floating"
    instance_path = os.path.join(base_dir, cfg.instance_name + ".npy")
    try:
        data = np.load(instance_path, allow_pickle=True).item()
    except FileNotFoundError:
        raise FileNotFoundError(f"Instance file not found: {instance_path}")
    # print(data)
    # change data key
    key_map = {"evolution_num": "n_evo",
              "hand_type": "hand_name",
              "hand_template_name": "tmpl_name",
              "hand_worldframe_contacts": "hand_cpn_w",
              "hand_contact_body_names": "hand_cbody",
              "necessary_contact_body_names": "required_cbody",
              "obj_worldframe_contacts": "obj_cpn_w"
              }
    data = {key_map.get(k, k): v for k, v in data.items()}
    if 'grasp_qpos' in data:
        data['grasp_qpos'] = data['grasp_qpos'][None,...]
    if 'squeeze_qpos' in data:
        data['squeeze_qpos'] = data['squeeze_qpos'][None,...]
    if 'pregrasp_qpos' in data:
        data['pregrasp_qpos'] = data['pregrasp_qpos'][None,...]
    # scene_cfg change to absolute path recursively
    def update_relative_path(d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                update_relative_path(v)
            elif k.endswith("_path") and isinstance(v, str):
                d[k] = os.path.abspath(to_absolute_path(v))
                d[k] = d[k].replace("experiment/", "")
            elif isinstance(v, str) and v == 'rigid_mesh':
                d[k] = 'rigid_object'
        return
    update_relative_path(data['scene_cfg'])
    if 'task' not in data['scene_cfg']:
        data['scene_cfg']['task'] = {
          'type': 'force_closure',
          'obj_name': data['scene_cfg']['interest_obj_name']
        }
        data['ext_center'] = np.array(data['obj_gravity_center'])
        data['ext_wrench'] = np.zeros(6)
    logging.info(f"Loaded instance from {instance_path}")
    logging.info(f"Instance data keys: {list(data.keys())}")
    logging.info(f"hand_cbody: {data['hand_cbody']}")
    logging.info(f"hand: {data['hand_name']}, template: {data['tmpl_name']}, object: {cfg.obj_name}")
    
    # data['scene_path'] = os.path.join(cfg.project_base_dir, data['scene_path'])
    # logging.debug(f"scene_path: {data['scene_path']}")
    return data

if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig
    from utils.log_util import set_logging
    from dexonomy.sim import MuJoCo_VisEnv, HandCfg, MuJoCo_OptCfg
    
    @hydra.main(config_path="config", config_name="base", version_base=None)
    def main(cfg: DictConfig):
        set_logging(cfg.verbose)
        data = load_instance_legacy(cfg)

        vis_env = MuJoCo_VisEnv(
        hand_cfg=HandCfg(xml_path=cfg.hand.xml_path, freejoint=True),
            scene_cfg=data['scene_cfg'],
            sim_cfg=MuJoCo_OptCfg(),
            vis_mode="visual",
        )
        
        # print(data["ho_c"]['dist'])
        # print(data["ho_c"]['bn1'])

        print(data["grasp_qpos"])
        
        from scipy.spatial.transform import Rotation as R
        qpos_dummy = np.zeros_like(data["grasp_qpos"][0])
        qpos_dummy[3] = 1.
        qpos_dummy[3:7] = R.from_rotvec(np.array([1, 0, 0]) * np.pi / 2).as_quat(scalar_first=True)
        xmat, xpos = vis_env.forward_kinematics(data["grasp_qpos"][0])
        hand_mesh = vis_env.get_posed_mesh(xmat, xpos, body_type="hand")
        hand_mesh.export("outputs/debug_hand.obj")
        obj_tm = vis_env.get_posed_mesh(xmat, xpos, body_type="obj")
        obj_tm.export("outputs/debug_obj.obj")
        
        from utils.vis_util import points_to_obj
        points_to_obj(data["obj_cpn_w"][:, :3], 1e-2, "outputs/debug_contact.obj")

        # output qpos to fast-graspd
        qpos = data["grasp_qpos"][0]
        print(qpos.tolist(), sep=",")
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
        print(qpos[6], qpos[3], qpos[4], qpos[5])
        np.save("obj_qpos.npy", qpos)
        
    main()