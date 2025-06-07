import numpy as np
import torch
import random
import math
import os
import warp as wp
import logging
import traceback
from glob import glob

from dexonomy.util.warp_util import MeshQueryPoint, MeshLineSegCollision
from dexonomy.util.np_rot_util import np_array32
from dexonomy.util.torch_rot_util import (
    torch_transform_points,
    torch_quaternion_to_matrix,
    torch_matrix_to_quaternion,
    torch_normalize_vector,
    torch_pose_multiply,
    torch_inv_transform_points,
)
from dexonomy.data.obj_loader import get_object_dataloader
from dexonomy.data.hand_loader import HandTemplateLibrary
from dexonomy.qp.qp_batched import get_qp_error_batched
from dexonomy.qp.qp_single import ContactQP


def valid_index_padding(parallel_id_lst, valid_idx_bool, num_min, num_max):
    padded_valid_idx = []
    for i in range(1 + parallel_id_lst.max()):
        i_old_valid = torch.where(parallel_id_lst == i)[0]
        i_new_valid = torch.where(valid_idx_bool & (parallel_id_lst == i))[0]
        if len(i_new_valid) < num_min:
            i_other = i_old_valid[~torch.isin(i_old_valid, i_new_valid)]
            i_new_valid = torch.cat(
                [
                    i_new_valid,
                    i_other[torch.randperm(len(i_other))[: num_min - len(i_new_valid)]],
                ]
            )
        elif len(i_new_valid) > num_max:
            i_new_valid = i_new_valid[torch.randperm(len(i_new_valid))[:num_max]]
        padded_valid_idx.extend(i_new_valid)
    padded_valid_idx = torch.tensor(padded_valid_idx, device=valid_idx_bool.device)
    return padded_valid_idx


class ObjectAligner:

    def __init__(self, device, opt_iter, filter):
        self.device = device
        self._wp_device = wp.torch.device_from_torch(device)
        self._wp_mesh_cache = {}
        self.buffer_shape = torch.Size([0])
        self.coll_buffer_length = 0
        self.opt_iter = opt_iter
        self.filter_cfg = filter
        return

    def load_warp_mesh(self, collision_mesh_lst):
        wm_id_lst = []
        for collision_mesh in collision_mesh_lst:
            obj_name, tm_obj = collision_mesh[0], collision_mesh[1]
            if obj_name not in self._wp_mesh_cache:
                v = wp.array(tm_obj.vertices, dtype=wp.vec3, device=self._wp_device)
                f = wp.array(np.ravel(tm_obj.faces), dtype=int, device=self._wp_device)
                vn = wp.array(
                    tm_obj.vertex_normals, dtype=wp.vec3, device=self._wp_device
                )
                self._wp_mesh_cache[obj_name] = wp.Mesh(
                    points=v, indices=f, velocities=vn
                )
            wm_id_lst.append(self._wp_mesh_cache[obj_name].id)
        wm_id_lst = torch.tensor(wm_id_lst, dtype=torch.int64, device=self.device)
        return wm_id_lst

    def matching(
        self,
        nf_hc,
        nf_hf_rot,
        nf_hf_trans,
        hf_hsk,
        grasp_qpos,
        hand_evolution_num,
        sampled_rot,
        sampled_trans,
        obj_scale,
        wf_sof_pose,
        wf_ext_center,
        wf_ext_wrench,
        collision_plane,
        collision_mesh_id,
    ):

        opt_q = torch_matrix_to_quaternion(sampled_rot)
        opt_q.requires_grad_()
        opt_t = sampled_trans
        opt_t.requires_grad_()
        optimizer = torch.optim.Adam(
            [{"params": opt_q, "lr": 1e-2}, {"params": opt_t, "lr": 1e-2}]
        )

        b, h = opt_t.shape[:-1]
        n_points = nf_hc.shape[-2]
        obj_scale = obj_scale.view(b, 1, 1, 3)

        if self.buffer_shape != torch.Size([b, h, n_points]):
            self.out_points = torch.zeros([b, h, n_points, 3], device=self.device)
            self.out_normals = torch.zeros([b, h, n_points, 3], device=self.device)
            self.adj_points = torch.zeros([b, h, n_points, 3], device=self.device)
            self.buffer_shape = torch.Size([b, h, n_points])

        for i in range(self.opt_iter):
            optimizer.zero_grad()
            opt_r = torch_quaternion_to_matrix(opt_q)
            of_hc = torch_transform_points(
                nf_hc, opt_r, opt_t.view(b, h, 1, 3), 1 / obj_scale
            )

            of_op, of_on = MeshQueryPoint.apply(
                of_hc[..., :3].contiguous(),
                self.out_points,
                self.out_normals,
                self.adj_points,
                collision_mesh_id,
            )

            loss_dist = ((obj_scale * (of_op - of_hc[..., :3])) ** 2).sum(
                dim=-1
            )  # [b,h,n]
            loss_normal = ((of_on - of_hc[..., 3:]) ** 2).sum(dim=-1)  # [b,h,n]
            loss = (1000 * loss_dist + loss_normal).mean(dim=-1).mean(dim=-1).sum()
            if i == self.opt_iter - 1:
                break

            loss.backward()
            optimizer.step()

        # Post processing
        sof_nf_quat = torch_normalize_vector(opt_q)
        sof_nf_rot = torch_quaternion_to_matrix(sof_nf_quat)
        sof_nf_trans = opt_t.view(b, h, 1, 3) * obj_scale

        wf_sof_rot = torch_quaternion_to_matrix(wf_sof_pose[..., 3:]).view(b, 1, 3, 3)
        wf_sof_trans = wf_sof_pose[..., :3].view(b, 1, 1, 3)

        wf_nf_rot, wf_nf_trans = torch_pose_multiply(
            wf_sof_rot, wf_sof_trans, sof_nf_rot, sof_nf_trans
        )

        wf_hc = torch_transform_points(of_hc, wf_sof_rot, wf_sof_trans, obj_scale)
        wf_op = torch_transform_points(of_op, wf_sof_rot, wf_sof_trans, obj_scale)
        wf_on = torch_transform_points(of_on, wf_sof_rot)

        wf_hf_rot, wf_hf_trans = torch_pose_multiply(
            wf_nf_rot,
            wf_nf_trans,
            nf_hf_rot,
            nf_hf_trans.unsqueeze(-2),
        )

        result_dict = {
            "grasp_pose": torch.cat(
                [wf_hf_trans.squeeze(-2), torch_matrix_to_quaternion(wf_hf_rot)],
                dim=-1,
            ).view(b * h, 7),
            "hand_contacts": wf_hc.view(b * h, -1, 6),
            "hand_skeleton": hf_hsk.view(b * h, -1, 3),
            "grasp_qpos": grasp_qpos.view(b * h, -1),
            "hand_evolution_num": hand_evolution_num.view(b * h),
            "obj_cp": wf_op.view(b * h, -1, 3),
            "obj_cn": wf_on.view(b * h, -1, 3),
            "obj_scale": obj_scale.repeat(1, h, 1, 1).view(b * h, 3),
            "parallel_id_lst": torch.arange(b * h, device=self.device) // h,
            "obj_pose": wf_sof_pose.repeat_interleave(h, dim=0),
            "ext_center": wf_ext_center.repeat_interleave(h, dim=0),
            "ext_wrench": wf_ext_wrench.repeat_interleave(h, dim=0),
        }

        # Filtering
        result_dict = self.wrapped_filter(
            result_dict,
            "loss_based",
            3,
            loss_normal=loss_normal.view(b * h, -1),
            loss_dist=loss_dist.view(b * h, -1),
        )
        result_dict = self.wrapped_filter(
            result_dict,
            "collision_based",
            2,
            collision_plane=collision_plane,
            collision_mesh_id=collision_mesh_id,
        )
        result_dict = self.wrapped_filter(result_dict, "qp_based", 1)
        result_dict = self.wrapped_filter(result_dict, "fps_based", 0)
        return result_dict

    @torch.no_grad()
    def wrapped_filter(self, result_dict, filter_name, later_filter_num, **kwargs):
        old_valid_num = len(result_dict["parallel_id_lst"])
        if self.filter_cfg[filter_name]["enable"]:
            valid_idx_bool = eval(f"self.{filter_name}_filter")(result_dict, **kwargs)
        else:
            valid_idx_bool = torch.ones(old_valid_num, dtype=bool, device=self.device)
        num_min = int(
            self.filter_cfg.final_num
            / ((1 - self.filter_cfg.min_ratio) ** later_filter_num)
        )
        num_max = int(
            self.filter_cfg.final_num
            / ((1 - self.filter_cfg.max_ratio) ** later_filter_num)
        )
        new_valid_idx = valid_index_padding(
            result_dict["parallel_id_lst"],
            valid_idx_bool,
            num_min,
            num_max,
        )
        for k, v in result_dict.items():
            result_dict[k] = v[new_valid_idx]
        logging.debug(
            f"{filter_name} filtering remain {len(new_valid_idx)} out of {old_valid_num}"
        )
        return result_dict

    def loss_based_filter(self, result_dict, loss_normal, loss_dist):
        loss_valid_idx = (
            loss_normal.max(dim=-1)[0] < self.filter_cfg["loss_based"].normal_thre
        ) & (loss_dist.max(dim=-1)[0] < self.filter_cfg["loss_based"].pos_thre)
        return loss_valid_idx

    def collision_based_filter(self, result_dict, collision_plane, collision_mesh_id):
        handframe_skeleton, grasp_pose, obj_pose, obj_scale, collision_mesh_id = (
            result_dict["hand_skeleton"],
            result_dict["grasp_pose"],
            result_dict["obj_pose"],
            result_dict["obj_scale"],
            collision_mesh_id[result_dict["parallel_id_lst"]],
        )
        collision_plane = (
            collision_plane[result_dict["parallel_id_lst"]]
            if collision_plane is not None
            else None
        )
        query_shape = grasp_pose.shape[0]
        if self.coll_buffer_length < query_shape:
            self.coll_buffer_length = query_shape
            self.coll_buf = torch.ones([query_shape], dtype=bool, device=self.device)
        else:
            self.coll_buf[:] = 1

        wf_hsk = torch_transform_points(
            handframe_skeleton,
            torch_quaternion_to_matrix(grasp_pose[..., 3:]),
            grasp_pose[..., :3].view(-1, 1, 3),
        )
        if collision_plane is not None:
            pf_hsk = torch_inv_transform_points(
                wf_hsk,
                torch_quaternion_to_matrix(collision_plane[..., 3:]),
                collision_plane[..., :3].view(-1, 1, 3),
            )
            self.coll_buf[:query_shape] = (pf_hsk[..., -1] > 0.02).min(dim=-1)[0]

        of_hsk = torch_inv_transform_points(
            wf_hsk,
            torch_quaternion_to_matrix(obj_pose[..., 3:]),
            obj_pose[..., :3].view(-1, 1, 3),
            obj_scale.view(-1, 1, 3),
        )
        out_valid_idx = MeshLineSegCollision(
            of_hsk, collision_mesh_id, self.coll_buf[:query_shape]
        )
        return out_valid_idx

    def qp_based_filter(self, result_dict):
        obj_points, obj_normals, ext_wrench, ext_center = (
            result_dict["obj_cp"],
            result_dict["obj_cn"],
            result_dict["ext_wrench"],
            result_dict["ext_center"],
        )

        qp_cfg = self.filter_cfg["qp_based"]
        miu_coef = qp_cfg["miu_coef"]
        qp_error = torch.ones(
            obj_points.shape[0], dtype=torch.float32, device=self.device
        )
        if qp_cfg["batched"]:
            logging.debug(
                f"qp shape: {obj_points.shape} {self.filter_cfg['qp_based']['max_batch_size']}"
            )
            max_batch_size = qp_cfg["max_batch_size"] // obj_points.shape[-2]
            batch_num = math.ceil(obj_points.shape[0] / max_batch_size)
            for i in range(batch_num):
                start = i * max_batch_size
                end = min((i + 1) * max_batch_size, obj_points.shape[0])

                _, qp_error[start:end] = get_qp_error_batched(
                    obj_points[start:end],
                    obj_normals[start:end],
                    ext_wrench[start:end],
                    ext_center[start:end],
                    miu_coef,
                )
        else:
            single_solver = ContactQP(miu_coef)
            for i in range(obj_points.shape[0]):
                _, qp_error[i] = single_solver.solve(
                    obj_points[i].cpu().numpy(),
                    obj_normals[i].cpu().numpy(),
                    ext_wrench[i].cpu().numpy(),
                    ext_center[i].cpu().numpy(),
                )

        out_valid_idx = qp_error < qp_cfg["threshold"]
        return out_valid_idx

    def fps_based_filter(self, result_dict):
        obj_points = result_dict["obj_cp"]
        obj_pose = result_dict["obj_pose"]
        parallel_id_lst = result_dict["parallel_id_lst"]
        of_op = torch_inv_transform_points(
            obj_points,
            torch_quaternion_to_matrix(obj_pose[..., 3:]),
            obj_pose[..., :3].view(-1, 1, 3),
        )
        out_valid_idx = torch.zeros(of_op.shape[0], device=self.device).bool()
        of_single_op = {}
        corr_index = {}
        for i in range(parallel_id_lst.shape[0]):
            oi = str(parallel_id_lst[i].data.item())
            if oi not in of_single_op.keys():
                of_single_op[oi] = [of_op[i]]
                corr_index[oi] = [i]
            else:
                of_single_op[oi].append(of_op[i])
                corr_index[oi].append(i)

        for k, v in of_single_op.items():
            all_points = torch.stack(v, dim=0)

            rand_start = random.randint(0, all_points.shape[0] - 1)
            valid_points = all_points[rand_start].unsqueeze(0)
            out_valid_idx[corr_index[k][rand_start]] = True
            for i in range(min(len(all_points), self.filter_cfg.final_num - 1)):
                farthest_dist, farthest_id = (
                    (valid_points.unsqueeze(0) - all_points.unsqueeze(1))
                    .norm(dim=-1)
                    .mean(dim=-1)
                    .min(dim=-1)[0]
                    .max(dim=-1)
                )
                valid_points = torch.cat(
                    [valid_points, all_points[farthest_id].unsqueeze(0)], dim=0
                )
                out_valid_idx[corr_index[k][farthest_id]] = True
        return out_valid_idx


def task_syn_obj(configs):
    task_config = configs.task
    assert os.path.exists(
        os.path.join(configs.init_template_dir, configs.template_name + ".npy")
    )
    logging.info(f"Hand template name: {configs.template_name}")

    obj_loader = get_object_dataloader(task_config.object, configs.n_worker)
    matcher = ObjectAligner(task_config.device, **task_config.matcher)
    try:
        hand_library = HandTemplateLibrary(
            xml_path=configs.hand.xml_path,
            skeleton_path=configs.hand_skeleton_path,
            template_name=configs.template_name,
            init_template_dir=configs.init_template_dir,
            new_template_dir=configs.new_template_dir,
            max_data_buffer=configs.max_template_buffer,
            num_workers=configs.n_worker,
        )

        for eee in range(task_config.epoch):
            logging.info(f"Epoch {eee}")
            for obj_samples in obj_loader:
                # Check exist path
                continue_flag = False
                for scene_cfg in obj_samples["scene_cfg"]:
                    check_dir = os.path.join(
                        configs.init_dir,
                        configs.template_name,
                        scene_cfg["scene_id"],
                        f"{eee}**.npy",
                    )
                    if len(glob(check_dir)) > 0:
                        continue_flag = True
                        break
                if continue_flag:
                    logging.info("skip")
                    continue

                obj_samples = {
                    k: (
                        v.to(task_config.device, non_blocking=True)
                        if isinstance(v, torch.Tensor)
                        else v
                    )
                    for k, v in obj_samples.items()
                }

                collision_mesh_id = matcher.load_warp_mesh(
                    obj_samples["collision_mesh"]
                )

                hand_temp_dict = hand_library.get_batched_data(
                    obj_samples["sampled_rot"].shape[:2], task_config.device
                )

                result_dict = matcher.matching(
                    hand_temp_dict["nf_hc"],
                    hand_temp_dict["nf_hf_rot"],
                    hand_temp_dict["nf_hf_trans"],
                    hand_temp_dict["hf_hsk"],
                    hand_temp_dict["grasp_qpos"][..., 7:],
                    hand_temp_dict["evolution_num"],
                    obj_samples["sampled_rot"],
                    obj_samples["sampled_trans"],
                    obj_samples["obj_scale"],
                    obj_samples["wf_sof_pose"],
                    obj_samples["wf_ext_center"],
                    obj_samples["wf_ext_wrench"],
                    (
                        obj_samples["collision_plane"]
                        if "collision_plane" in obj_samples
                        else None
                    ),
                    collision_mesh_id,
                )
                succ_num = result_dict["grasp_pose"].shape[0]
                logging.info(f"Get {succ_num} valid")
                if succ_num == 0:
                    continue

                for k, v in result_dict.items():
                    result_dict[k] = v.cpu().numpy()

                for count, (
                    grasp_pose,
                    grasp_qpos,
                    hand_c,
                    hand_evolution_num,
                    obj_cp,
                    obj_cn,
                    obj_id,
                    ext_center,
                    ext_wrench,
                ) in enumerate(
                    zip(
                        result_dict["grasp_pose"],
                        result_dict["grasp_qpos"],
                        result_dict["hand_contacts"],
                        result_dict["hand_evolution_num"],
                        result_dict["obj_cp"],
                        result_dict["obj_cn"],
                        result_dict["parallel_id_lst"],
                        result_dict["ext_center"],
                        result_dict["ext_wrench"],
                    )
                ):

                    scene_cfg = obj_samples["scene_cfg"][obj_id]
                    scene_path = obj_samples["scene_path"][obj_id]
                    grasp_dir = os.path.join(
                        configs.init_dir,
                        hand_temp_dict["hand_template_name"],
                        scene_cfg["scene_id"],
                    )
                    os.makedirs(grasp_dir, exist_ok=True)

                    obj_worldframe_contacts = np.concatenate([obj_cp, -obj_cn], axis=-1)
                    np.save(
                        f"{grasp_dir}/{eee}_{count}_grasp.npy",
                        {
                            "evolution_num": np_array32([hand_evolution_num]),
                            "hand_type": configs.hand_name,
                            "hand_template_name": hand_temp_dict["hand_template_name"],
                            "grasp_qpos": np.concatenate([grasp_pose, grasp_qpos])[
                                None
                            ],
                            "hand_worldframe_contacts": hand_c,
                            "hand_contact_body_names": hand_temp_dict[
                                "hand_contact_body_names"
                            ],
                            "necessary_contact_body_names": hand_temp_dict[
                                "necessary_contact_body_names"
                            ],
                            "obj_worldframe_contacts": obj_worldframe_contacts,
                            "ext_center": ext_center,
                            "ext_wrench": ext_wrench,
                            "scene_path": scene_path,
                        },
                    )
        logging.info("Finish task syn_obj.")
    except BaseException as e:
        hand_library.stop()
        error_traceback = traceback.format_exc()
        logging.info(f"{error_traceback}")
    finally:
        hand_library.stop()

    return
