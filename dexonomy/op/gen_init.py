import numpy as np
import torch
import random
import math
import os
import logging
import traceback
from glob import glob

from dexonomy.util.warp_util import WarpCollisionEnv
from dexonomy.util.np_util import np_array32
from dexonomy.util.torch_util import (
    torch_transform_points,
    torch_quaternion_to_matrix,
    torch_matrix_to_quaternion,
    torch_normalize_vector,
    torch_multiply_pose,
    torch_inv_transform_points,
)
from dexonomy.data.obj_loader import get_object_dataloader
from dexonomy.data.hand_loader import HandTemplateLoader
from dexonomy.qp.qp_batched import get_qp_error_batched
from dexonomy.qp.qp_single import ContactQP
from dexonomy.util.file_util import get_template_names


def valid_index_padding(
    obj_id: torch.Tensor, valid_bool: torch.Tensor, n_min: int, n_max: int
) -> torch.Tensor:
    """Padding the valid index of each object to satisfy the minimum and maximum number."""
    padded_valid_idx = []
    for i in range(1 + obj_id.max()):
        i_old_valid = torch.where(obj_id == i)[0]
        i_new_valid = torch.where(valid_bool & (obj_id == i))[0]
        if len(i_new_valid) < n_min:
            i_other = i_old_valid[~torch.isin(i_old_valid, i_new_valid)]
            i_new_valid = torch.cat(
                [
                    i_new_valid,
                    i_other[torch.randperm(len(i_other))[: n_min - len(i_new_valid)]],
                ]
            )
        elif len(i_new_valid) > n_max:
            i_new_valid = i_new_valid[torch.randperm(len(i_new_valid))[:n_max]]
        padded_valid_idx.extend(i_new_valid)
    padded_valid_idx = torch.tensor(padded_valid_idx, device=valid_bool.device)
    return padded_valid_idx


class InitFilter:
    def __init__(self, cfg: dict, device: torch.device, coll_env: WarpCollisionEnv):
        self.cfg = cfg
        self.device = device
        self.coll_env = coll_env
        self.valid_filter = []
        for k in list(self.cfg.keys()):
            if k != "general" and self.cfg[k] is not None:
                self.valid_filter.append(k)
        return

    @torch.no_grad()
    def _wrapped_filter(self, data, filter_name, n_filter_remained):
        n_old = len(data["obj_id"])
        valid_bool = eval(f"self._{filter_name}_filter")(data)
        gen_cfg = self.cfg.general
        n_min = int(gen_cfg.n_final / ((1 - gen_cfg.min_ratio) ** n_filter_remained))
        n_max = int(gen_cfg.n_final / ((1 - gen_cfg.max_ratio) ** n_filter_remained))
        valid_idx = valid_index_padding(data["obj_id"], valid_bool, n_min, n_max)
        for k, v in data.items():
            data[k] = v[valid_idx]
        logging.debug(f"{filter_name} filtering remain {len(valid_idx)} out of {n_old}")
        return data

    def _pose_filter(self, data):
        hand_rot = torch_quaternion_to_matrix(data["grasp_pose"][..., 3:7])
        obj_direction = torch_normalize_vector(data["obj_pose"][:, :3])
        finger_align = (hand_rot[:, :, 2] * obj_direction).sum(dim=-1)
        valid_bool = (finger_align > self.cfg.pose.thre) | (
            data["obj_pose"][:, :3].norm(dim=-1) < 0.1
        )  # If object is not around the origin, the hand direction should be aligned with the object
        return valid_bool

    def _loss_filter(self, data):
        valid_bool = (
            data["loss_normal"].max(dim=-1)[0] < self.cfg.loss.normal_thre
        ) & (data["loss_dist"].max(dim=-1)[0] < self.cfg.loss.pos_thre)
        return valid_bool

    def _collision_filter(self, data):
        hand_skt_h = data["hand_skt_h"]
        grasp_pose = data["grasp_pose"]
        obj_pose = data["obj_pose"]
        obj_scale = data["obj_scale"]
        obj_id = data["obj_id"]
        coll_cfg = self.cfg.collision

        hand_skt_w = torch_transform_points(
            hand_skt_h,
            torch_quaternion_to_matrix(grasp_pose[..., 3:]),
            grasp_pose[..., :3].view(-1, 1, 3),
        )
        valid_bool1 = self.coll_env.check_plane(hand_skt_w, obj_id, coll_cfg.plane_thre)
        hand_skt_o = torch_inv_transform_points(
            hand_skt_w,
            torch_quaternion_to_matrix(obj_pose[..., 3:]),
            obj_pose[..., :3].view(-1, 1, 3),
            obj_scale.view(-1, 1, 3),
        )
        valid_bool2 = self.coll_env.check_mesh(hand_skt_o, obj_id)
        return valid_bool1 & valid_bool2

    def _qp_filter(self, data):
        obj_point = data["obj_cp_w"]
        obj_normal = data["obj_cn_w"]
        ext_wrench = data["ext_wrench"]
        ext_center = data["ext_center"]

        qp_cfg = self.cfg.qp
        miu_coef = qp_cfg.miu_coef
        qp_error = torch.ones(
            obj_point.shape[0], dtype=torch.float32, device=self.device
        )
        if qp_cfg.batch_size is not None:
            logging.debug(f"qp shape: {obj_point.shape} {qp_cfg.batch_size}")
            batch_size = qp_cfg.batch_size // obj_point.shape[-2]
            n_batch = math.ceil(obj_point.shape[0] / batch_size)
            for i in range(n_batch):
                start = i * batch_size
                end = min((i + 1) * batch_size, obj_point.shape[0])
                _, qp_error[start:end] = get_qp_error_batched(
                    obj_point[start:end],
                    obj_normal[start:end],
                    ext_wrench[start:end],
                    ext_center[start:end],
                    miu_coef,
                )
        else:
            single_solver = ContactQP(miu_coef)
            for i in range(obj_point.shape[0]):
                _, qp_error[i] = single_solver.solve(
                    obj_point[i].cpu().numpy(),
                    obj_normal[i].cpu().numpy(),
                    ext_wrench[i].cpu().numpy(),
                    ext_center[i].cpu().numpy(),
                )
        valid_bool = qp_error < qp_cfg.thre
        return valid_bool

    def _fps_filter(self, data):
        obj_cp_w = data["obj_cp_w"]
        obj_pose = data["obj_pose"]
        obj_id = data["obj_id"]
        fps_cfg = self.cfg.fps

        obj_cp_o = torch_inv_transform_points(
            obj_cp_w,
            torch_quaternion_to_matrix(obj_pose[..., 3:]),
            obj_pose[..., :3].view(-1, 1, 3),
        )
        valid_bool = torch.zeros(obj_cp_o.shape[0], device=self.device).bool()
        obj_cp_dict = {}
        corr_index = {}
        for i in range(obj_id.shape[0]):
            o_id = str(obj_id[i].data.item())
            if o_id not in obj_cp_dict.keys():
                obj_cp_dict[o_id] = [obj_cp_o[i]]
                corr_index[o_id] = [i]
            else:
                obj_cp_dict[o_id].append(obj_cp_o[i])
                corr_index[o_id].append(i)

        for k, v in obj_cp_dict.items():
            all_points = torch.stack(v, dim=0)
            rand_start = random.randint(0, all_points.shape[0] - 1)
            valid_points = all_points[rand_start].unsqueeze(0)
            valid_bool[corr_index[k][rand_start]] = True
            for i in range(min(len(all_points), self.cfg.general.n_final - 1)):
                farthest_dist, farthest_id = (
                    (valid_points.unsqueeze(0) - all_points.unsqueeze(1))
                    .norm(dim=-1)
                    .mean(dim=-1)
                    .min(dim=-1)[0]
                    .max(dim=-1)
                )
                if fps_cfg.dist_thre is not None and farthest_dist > fps_cfg.dist_thre:
                    break
                valid_points = torch.cat(
                    [valid_points, all_points[farthest_id].unsqueeze(0)], dim=0
                )
                valid_bool[corr_index[k][farthest_id]] = True
        return valid_bool

    def forward(self, data: dict):
        for i, f in enumerate(self.valid_filter):
            data = self._wrapped_filter(data, f, len(self.valid_filter) - i - 1)
        return data


class HandObjMatcher:

    def __init__(self, cfg: dict, device: torch.device, coll_env: WarpCollisionEnv):
        self.cfg = cfg
        self.device = device
        self.coll_env = coll_env

    def forward(self, tmpl_sample: dict, obj_sample: dict):
        opt_q = torch_matrix_to_quaternion(obj_sample["rot_o2c_init"])
        opt_t = obj_sample["trans_o2c_init"]
        opt_q.requires_grad_()
        opt_t.requires_grad_()
        optimizer = torch.optim.Adam(
            [{"params": opt_q, "lr": 1e-2}, {"params": opt_t, "lr": 1e-2}]
        )

        b, h = opt_t.shape[:-1]
        obj_scale = obj_sample["obj_scale"].view(b, 1, 1, 3)
        h_cpn_c = tmpl_sample["hand_cpn_c"]

        for i in range(self.cfg.opt_step):
            optimizer.zero_grad()
            opt_r = torch_quaternion_to_matrix(opt_q)
            h_cpn_o = torch_transform_points(
                h_cpn_c, opt_r, opt_t.view(b, h, 1, 3), 1 / obj_scale
            )
            o_cp_o, o_cn_o = self.coll_env.query_point(h_cpn_o[..., :3])
            loss_dist = ((obj_scale * (o_cp_o - h_cpn_o[..., :3])) ** 2).sum(dim=-1)
            loss_normal = ((o_cn_o - h_cpn_o[..., 3:]) ** 2).sum(dim=-1)  # [b,h,n]
            loss = (1000 * loss_dist + loss_normal).mean(dim=-1).mean(dim=-1).sum()
            if i == self.cfg.opt_step - 1:
                break

            loss.backward()
            optimizer.step()

        # Post processing
        quat_so2c = torch_normalize_vector(opt_q)
        rot_so2c = torch_quaternion_to_matrix(quat_so2c)
        trans_so2c = opt_t.view(b, h, 1, 3) * obj_scale

        pose_w2so = obj_sample["pose_w2so"]
        rot_w2so = torch_quaternion_to_matrix(pose_w2so[..., 3:]).view(b, 1, 3, 3)
        trans_w2so = pose_w2so[..., :3].view(b, 1, 1, 3)

        rot_w2c, trans_w2c = torch_multiply_pose(
            rot_w2so, trans_w2so, rot_so2c, trans_so2c
        )

        h_cpn_w = torch_transform_points(h_cpn_o, rot_w2so, trans_w2so, obj_scale)
        o_cp_w = torch_transform_points(o_cp_o, rot_w2so, trans_w2so, obj_scale)
        o_cn_w = torch_transform_points(o_cn_o, rot_w2so)

        rot_c2h = tmpl_sample["rot_c2h"]
        trans_c2h = tmpl_sample["trans_c2h"].unsqueeze(-2)
        rot_w2h, trans_w2h = torch_multiply_pose(rot_w2c, trans_w2c, rot_c2h, trans_c2h)
        quat_w2h = torch_matrix_to_quaternion(rot_w2h)
        trans_w2h = trans_w2h.squeeze(-2)

        ret_dict = {
            "loss_normal": loss_normal.view(b * h, -1),
            "loss_dist": loss_dist.view(b * h, -1),
            "grasp_pose": torch.cat([trans_w2h, quat_w2h], dim=-1).view(b * h, 7),
            "hand_cpn_w": h_cpn_w.view(b * h, -1, 6),
            "hand_skt_h": tmpl_sample["hand_skt_h"].view(b * h, -1, 3),
            "grasp_qpos": tmpl_sample["grasp_qpos"][..., 7:].view(b * h, -1),
            "n_evo": tmpl_sample["n_evo"].view(b * h),
            "obj_cp_w": o_cp_w.view(b * h, -1, 3),
            "obj_cn_w": o_cn_w.view(b * h, -1, 3),
            "obj_scale": obj_scale.repeat(1, h, 1, 1).view(b * h, 3),
            "obj_id": torch.arange(b * h, device=self.device) // h,
            "obj_pose": pose_w2so.repeat_interleave(h, dim=0),
            "ext_center": obj_sample["ext_center"].repeat_interleave(h, dim=0),
            "ext_wrench": obj_sample["ext_wrench"].repeat_interleave(h, dim=0),
        }
        return ret_dict


def operate_init(cfg):
    op_cfg = cfg.op
    tmpl_name = get_template_names(cfg.tmpl_name, cfg.init_tmpl_dir)[0]  # only one
    assert op_cfg.device in ["cpu", "cuda"]
    if op_cfg.device == "cuda":
        torch.cuda.set_device(cfg.init_gpu[0])
        op_cfg.device += f":{cfg.init_gpu[0]}"
    logging.info(f"Template name: {tmpl_name}, device: {op_cfg.device}")

    obj_loader = get_object_dataloader(op_cfg.object, cfg.n_worker)
    coll_env = WarpCollisionEnv(op_cfg.device)
    ho_matcher = HandObjMatcher(op_cfg.matcher, op_cfg.device, coll_env)
    init_filter = InitFilter(op_cfg.filter, op_cfg.device, coll_env)

    try:
        tmpl_loader = HandTemplateLoader(
            xml_path=cfg.hand.xml_path,
            skt_path=cfg.hand_skt_path,
            tmpl_name=tmpl_name,
            init_tmpl_dir=cfg.init_tmpl_dir,
            new_tmpl_dir=cfg.new_tmpl_dir,
            max_data_buffer=cfg.tmpl_max_buf,
            n_worker=cfg.n_worker,
        )

        for eee in range(op_cfg.epoch):
            logging.info(f"Epoch {eee}")
            for obj_sample in obj_loader:
                if cfg.skip_done:
                    # Check if the result already exists
                    continue_flag = False
                    for scene_cfg in obj_sample["scene_cfg"]:
                        check_dir = os.path.join(
                            cfg.init_dir,
                            tmpl_name,
                            scene_cfg["scene_id"],
                            f"{eee}*.npy",
                        )
                        if len(glob(check_dir)) > 0:
                            continue_flag = True
                            break
                    if continue_flag:
                        logging.info("Skip")
                        continue

                obj_sample = {
                    k: (
                        v.to(op_cfg.device, non_blocking=True)
                        if isinstance(v, torch.Tensor)
                        else v
                    )
                    for k, v in obj_sample.items()
                }

                coll_env.load(obj_sample["col_plane"], obj_sample["col_mesh"])

                tmpl_sample = tmpl_loader.get_batched_data(
                    obj_sample["rot_o2c_init"].shape[:2], op_cfg.device
                )
                ret_dict = ho_matcher.forward(tmpl_sample, obj_sample)
                ret_dict = init_filter.forward(ret_dict)

                n_valid = ret_dict["grasp_pose"].shape[0]
                logging.info(f"Get {n_valid} valid")
                if n_valid == 0:
                    continue

                for k, v in ret_dict.items():
                    ret_dict[k] = v.cpu().numpy()

                for count, o_id in enumerate(ret_dict["obj_id"]):
                    g_pose = ret_dict["grasp_pose"][count]
                    g_qpos = ret_dict["grasp_qpos"][count]
                    o_cp_w = ret_dict["obj_cp_w"][count]
                    o_cn_w = ret_dict["obj_cn_w"][count]
                    scene_cfg = obj_sample["scene_cfg"][o_id]
                    grasp_dir = os.path.join(
                        cfg.init_dir,
                        tmpl_sample["tmpl_name"],
                        scene_cfg["scene_id"],
                    )
                    os.makedirs(grasp_dir, exist_ok=True)
                    np.save(
                        f"{grasp_dir}/{eee}_{count}_grasp.npy",
                        {
                            "hand_name": cfg.hand_name,
                            "tmpl_name": tmpl_sample["tmpl_name"],
                            "hand_cbody": tmpl_sample["hand_cbody"],
                            "required_cbody": tmpl_sample["required_cbody"],
                            "grasp_qpos": np.concatenate([g_pose, g_qpos])[None],
                            "obj_cpn_w": np.concatenate([o_cp_w, -o_cn_w], axis=-1),
                            "hand_cpn_w": ret_dict["hand_cpn_w"][count],
                            "ext_center": ret_dict["ext_center"][count],
                            "ext_wrench": ret_dict["ext_wrench"][count],
                            "scene_path": obj_sample["scene_path"][o_id],
                            "n_evo": np_array32([ret_dict["n_evo"][count]]),
                        },
                    )
        logging.info("Finish initialization.")
    except BaseException as e:
        tmpl_loader.stop()
        error_traceback = traceback.format_exc()
        logging.info(f"{error_traceback}")
    finally:
        tmpl_loader.stop()

    return
