import os
import multiprocessing
import logging
import glob
import numpy as np

from dexonomy.sim import MuJoCo_OptEnv, MuJoCo_OptCfg, HandCfg
from dexonomy.qp.qp_single import ContactQP
from dexonomy.util.np_util import np_array32
from dexonomy.util.file_util import load_scene_cfg, safe_wrapper


class GraspFilter:
    def __init__(self, cfg: dict, sim_env: MuJoCo_OptEnv, log_name: str):
        self.cfg = cfg
        self.sim_env = sim_env
        self.log_name = log_name

    def forward(self, info: dict) -> tuple[bool, dict]:
        contact_thre = self.cfg.contact.thre if "contact" in self.cfg else None
        ho_c, hh_c = self.sim_env.get_contacts(contact_thre)
        if "limit" in self.cfg and self.cfg.limit and not self._limit_filter():
            return False, ho_c
        if "collision" in self.cfg and self.cfg.collision:
            if not self._collision_filter(ho_c, hh_c):
                return False, ho_c
        if "contact" in self.cfg and self.cfg.contact:
            if not self._contact_filter(ho_c, info["required_cbody"]):
                return False, ho_c
        if "qp" in self.cfg and self.cfg.qp:
            if not self._qp_filter(ho_c, info["ext_wrench"], info["ext_center"]):
                return False, ho_c
        return True, ho_c

    def early_stop(self) -> bool:
        contact_thre = self.cfg.contact.thre if "contact" in self.cfg else None
        ho_c, hh_c = self.sim_env.get_contacts(contact_thre)
        return self._collision_filter(ho_c, hh_c, slience=True)

    def _collision_filter(self, ho_c: dict, hh_c: dict, slience: bool = False) -> bool:
        col_cfg = self.cfg.collision
        if len(hh_c["dist"]) > 0 and min(hh_c["dist"]) < col_cfg.hh_thre:
            if not slience:
                logging.debug(f"{self.log_name} collision hh {min(hh_c['dist'])}")
            return False

        if len(ho_c["dist"]) > 0 and min(ho_c["dist"]) < col_cfg.ho_thre:
            if not slience:
                logging.debug(f"{self.log_name} collision ho {min(ho_c['dist'])}")
            return False
        return True

    def _contact_filter(self, ho_c: dict, required_cbody: list[str]):
        contact_cfg = self.cfg.contact
        if len(ho_c["dist"]) == 0:
            logging.debug(f"{self.log_name} no contact")
            return False
        ho_cb = set(ho_c["bn1"])
        if contact_cfg.check_body:
            for rcb in required_cbody:
                if len(ho_cb.intersection(set(rcb))) == 0:
                    logging.debug(f"{self.log_name} required: {rcb} current: {ho_cb}")
                    return False
        return True

    def _qp_filter(
        self, ho_c: dict, ext_wrench: np.ndarray, ext_center: np.ndarray
    ) -> bool:
        qp_cfg = self.cfg.qp
        contact_wrenches, wrench_error = ContactQP(miu_coef=qp_cfg.miu_coef).solve(
            ho_c["pos"], ho_c["normal"], ext_wrench, ext_center
        )
        if wrench_error > qp_cfg.thre or contact_wrenches is None:
            logging.debug(f"{self.log_name} bad QP error {wrench_error}")
            return False
        ho_c["wrench"] = 10 * contact_wrenches
        return True

    def _limit_filter(self) -> bool:
        return self.sim_env.check_qpos_limit(self.cfg.limit.thre)


@safe_wrapper
def _single_grasp(param):
    input_path, cfg = param[0], param[1]
    grasp_path = input_path.replace(cfg.init_dir, cfg.grasp_dir)
    grasp_cfg, pregrasp_cfg = cfg.op.grasp, cfg.op.pregrasp
    debug_path = input_path.replace(cfg.init_dir, cfg.debug_dir)

    grasp_data = np.load(input_path, allow_pickle=True).item()
    sim_env = MuJoCo_OptEnv(
        hand_cfg=HandCfg(xml_path=cfg.hand.xml_path, freejoint=True),
        scene_cfg=load_scene_cfg(grasp_data["scene_path"]),
        sim_cfg=MuJoCo_OptCfg(obj_margin=pregrasp_cfg.cdist),
        debug_render=cfg.debug_render,
        debug_view=cfg.debug_view,
    )
    sim_env.reset_qpos(grasp_data["grasp_qpos"][0])
    sim_env.set_obj_margin(grasp_cfg.cdist)

    grasp_filter = GraspFilter(grasp_cfg.filter, sim_env, f"{input_path} grasp")
    pregrasp_filter = GraspFilter(
        pregrasp_cfg.filter, sim_env, f"{input_path} pregrasp"
    )

    # Generate grasp qpos
    hand_cbody, obj_cpn_w = grasp_data["hand_cbody"], grasp_data["obj_cpn_w"]
    hand_cpn_b = sim_env.transform_cpn_w2b(hand_cbody, grasp_data["hand_cpn_w"])
    for ii in range(grasp_cfg.step):
        curr_diff = sim_env.apply_contact_forces(hand_cbody, hand_cpn_b, obj_cpn_w)
        sim_env.step_sim(grasp_cfg.substep)
        if ii == 0 or np.max(prev_diff - curr_diff) > 1e-4:
            prev_diff = np.copy(curr_diff)
            continue
        else:
            prev_diff = np.copy(curr_diff)
        if grasp_filter.early_stop():
            break

    grasp_data["grasp_qpos"] = np_array32(sim_env.get_hand_qpos())[None]

    succ_flag, ho_c = grasp_filter.forward(grasp_data)
    grasp_data["ho_c"] = ho_c
    if not succ_flag:
        sim_env.save_debug(debug_path)
        return input_path

    # Generate squeeze qpos
    squeeze_qpos = sim_env.get_squeeze_qpos(
        grasp_data["grasp_qpos"][0], ho_c["bn1"], ho_c["pos"], ho_c["wrench"]
    )
    grasp_data["squeeze_qpos"] = np_array32(squeeze_qpos)[None]

    # Update contact information in template
    if cfg.tmpl_upd_mode == "orig":
        hand_cpn_w = sim_env.transform_cpn_b2w(hand_cbody, hand_cpn_b)
    elif cfg.tmpl_upd_mode == "real" or cfg.tmpl_upd_mode == "disabled":
        hand_cpn_w = np.concatenate([ho_c["pos"], ho_c["normal"]], axis=-1)
        grasp_data["hand_cbody"] = ho_c["bn1"]
    elif cfg.tmpl_upd_mode == "hybrid":
        hand_cpn_w = sim_env.transform_cpn_b2w(hand_cbody, hand_cpn_b)
        for h_cb, h_cpn_w in zip(hand_cbody, hand_cpn_w):
            for c_bn1, c_pos, c_normal in zip(ho_c["bn1"], ho_c["pos"], ho_c["normal"]):
                if c_bn1 == h_cb:
                    dist = np.linalg.norm(h_cpn_w[:3] - c_pos)
                    angle = np.arccos(
                        np.clip((h_cpn_w[3:] * c_normal).sum(), a_min=-1, a_max=1)
                    )
                    if dist < 0.03 and angle < np.pi / 4:
                        h_cpn_w[:3], h_cpn_w[3:] = c_pos, c_normal
                        break
    else:
        raise NotImplementedError(
            f"Undefined template update strategy: {cfg.tmpl_upd_mode}. Available choices: [orig, real, hybrid, disabled]"
        )
    grasp_data["hand_cpn_w"] = hand_cpn_w

    # Generate pregrasp qpos list
    if pregrasp_cfg:
        pregrasp_lst, n_interval = [], 2
        for ii in range(pregrasp_cfg.step):
            curr_margin = pregrasp_cfg.cdist * min((ii + 1) / pregrasp_cfg.step * 2, 1)
            sim_env.set_obj_margin(curr_margin)
            sim_env.keep_hand_stable()
            sim_env.step_sim(grasp_cfg.substep)
            pregrasp_lst.append(np.copy(sim_env.get_hand_qpos()))
            if ii % n_interval == 0 and curr_margin >= pregrasp_cfg.cdist:
                if pregrasp_filter.early_stop():
                    break

        succ_flag, _ = pregrasp_filter.forward(None)
        if not succ_flag:
            sim_env.save_debug(debug_path)
            return input_path

        pregrasp_qpos = np.stack(pregrasp_lst, axis=0)[::-n_interval]
        grasp_data["pregrasp_qpos"] = np_array32(pregrasp_qpos)

    sim_env.save_debug(debug_path)
    os.makedirs(os.path.dirname(grasp_path), exist_ok=True)
    grasp_data["n_evo"] += 1
    np.save(grasp_path, grasp_data)
    logging.debug(f"save to {grasp_path}")
    return input_path


def operate_grasp(cfg: dict):
    input_path_lst = glob.glob(os.path.join(cfg.init_dir, "**/*.npy"), recursive=True)

    # Debug mode: only process the debug name
    if cfg.debug_name is not None:
        input_path_lst = [p for p in input_path_lst if cfg.debug_name in p]

    # Skip already logged paths
    logged_paths = []
    if cfg.skip_done and os.path.exists(cfg.log_path):
        with open(cfg.log_path, "r") as f:
            logged_paths = f.readlines()
        logged_paths = [p.split("\n")[0] for p in logged_paths]
        input_path_lst = list(set(input_path_lst).difference(set(logged_paths)))

    logging.info(f"Find {len(input_path_lst)} initialization")
    if len(input_path_lst) == 0:
        return

    param_lst = zip(input_path_lst, [cfg] * len(input_path_lst))
    if cfg.n_worker == 1 or cfg.debug_view:
        for ip in param_lst:
            _single_grasp(ip)
    else:
        with multiprocessing.Pool(processes=cfg.n_worker) as pool:
            jobs = pool.imap_unordered(_single_grasp, param_lst)
            results = list(jobs)
            # Log the processed paths
            write_mode = "a" if cfg.skip_done else "w"
            with open(cfg.log_path, write_mode) as f:
                f.write("\n".join(results) + "\n")

    logging.info(f"Finish grasp generation")

    return
