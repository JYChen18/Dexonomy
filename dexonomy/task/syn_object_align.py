import numpy as np
import torch
import random
import math
import os
import warp as wp
import logging

from dexonomy.util.warp_util import MeshQueryPoint, MeshLineSegCollision
from dexonomy.util.np_rot_util import np_array32
from dexonomy.util.torch_rot_util import (
    torch_normal_to_rot,
    torch_transform_points,
    torch_axis_angle_rotation,
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


def sample_init_poses(
    sampled_points,
    sampled_normals,
    init_inplane_num,
    scale_range,
):
    batch_size = sampled_normals.shape[0]
    sampled_rot_base = torch_normal_to_rot(sampled_normals).unsqueeze(-3)
    delta_rot = torch_axis_angle_rotation(
        "X",
        torch.linspace(-np.pi, np.pi, init_inplane_num, device=sampled_normals.device),
    ).view(1, 1, -1, 3, 3)
    sampled_rot = (sampled_rot_base @ delta_rot).view(batch_size, -1, 3, 3)
    sampled_trans = (
        sampled_points.view(batch_size, -1, 1, 3)
        .repeat(1, 1, init_inplane_num, 1)
        .view(batch_size, -1, 3)
    )

    sampled_scale = (
        torch.rand_like(sampled_trans[..., 0]).to(sampled_trans.device)
        * (scale_range[1] - scale_range[0])
        + scale_range[0]
    )
    return sampled_rot, sampled_trans, sampled_scale


class ObjectAligner:

    def __init__(
        self, device, opt_iter, loss_filter, coll_filter, qp_filter, fps_filter
    ):
        self.device = device
        self._wp_device = wp.torch.device_from_torch(device)
        self._wp_mesh_cache = {}
        self.buffer_shape = torch.Size([0])
        self.coll_buffer_length = 0

        self.loss_filter = loss_filter
        self.coll_filter = coll_filter
        self.qp_filter = qp_filter
        self.fps_filter = fps_filter
        self.opt_iter = opt_iter
        return

    def load_warp_mesh(self, tm_lst, obj_name_lst):
        wm_id_lst = []
        for tm_obj, obj_name in zip(tm_lst, obj_name_lst):
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
        hf_hsk,
        hf_ogd,
        nf_hf_rot,
        nf_hf_trans,
        grasp_qpos,
        hand_evolution_num,
        sampled_rot,
        sampled_trans,
        sampled_scale,
        of_ogc,
        swf_of_pose,
        wm_id_lst,
        scale_range,
        unified_obj_mass,
        has_floor_z0,
        override_ogd,
    ):

        opt_q = torch_matrix_to_quaternion(sampled_rot)
        opt_q.requires_grad_()
        opt_t = sampled_trans
        opt_t.requires_grad_()
        opt_s = sampled_scale
        opt_s.requires_grad_()

        b, h = opt_t.shape[:-1]
        n_points = nf_hc.shape[-2]

        if self.buffer_shape != torch.Size([b, h, n_points]):
            self.out_points = torch.zeros([b, h, n_points, 3], device=self.device)
            self.out_normals = torch.zeros([b, h, n_points, 3], device=self.device)
            self.adj_points = torch.zeros([b, h, n_points, 3], device=self.device)
            self.buffer_shape = torch.Size([b, h, n_points])

        optimizer = torch.optim.Adam(
            [
                {"params": opt_q, "lr": 1e-2},
                {"params": opt_t, "lr": 1e-2},
                {"params": opt_s, "lr": 1e-3},
            ]
        )

        for i in range(self.opt_iter):
            optimizer.zero_grad()
            opt_r = torch_quaternion_to_matrix(opt_q)
            of_hc = torch_transform_points(
                nf_hc,
                opt_r,
                opt_t.view(b, h, 1, 3),
                1 / opt_s.view(b, h, 1, 1),
            )

            of_op, of_on = MeshQueryPoint.apply(
                of_hc[..., :3].contiguous(),
                self.out_points,
                self.out_normals,
                self.adj_points,
                wm_id_lst,
            )

            loss_dist = ((opt_s.view(b, h, 1, 1) * (of_op - of_hc[..., :3])) ** 2).sum(
                dim=-1
            )  # [b,h,n]
            loss_normal = ((of_on - of_hc[..., 3:]) ** 2).sum(dim=-1)  # [b,h,n]
            loss = (1000 * loss_dist + loss_normal).mean(dim=-1).mean(dim=-1).sum()
            if i == self.opt_iter - 1:
                break

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                opt_s.clamp_(min=scale_range[0], max=scale_range[1])

        with torch.no_grad():
            if self.loss_filter.enable:
                loss_valid_idx = torch.where(
                    (
                        loss_normal.view(b * h, n_points).max(dim=-1)[0]
                        < self.loss_filter.normal_thre
                    )
                    & (
                        loss_dist.view(b * h, n_points).max(dim=-1)[0]
                        < self.loss_filter.pos_thre
                    )
                )[0]
            else:
                loss_valid_idx = torch.arange(b * h, device=self.device)
            logging.debug(
                f"Loss filtering remain {loss_valid_idx.shape[0]} out of {b * h}"
            )
            obj_id_lst = loss_valid_idx // h

            of_nf_quat = torch_normalize_vector(opt_q)
            of_nf_rot = torch_quaternion_to_matrix(of_nf_quat)
            of_nf_trans = opt_t.view(b, h, 1, 3)

            wf_of_rot = torch_quaternion_to_matrix(swf_of_pose[..., 3:]).view(
                b, 1, 3, 3
            )
            wf_of_trans = swf_of_pose[..., :3].view(b, 1, 1, 3) * opt_s.view(b, h, 1, 1)
            wf_of_scale = opt_s.view(b, h, 1, 1)
            wf_of_pose = torch.cat(
                [
                    wf_of_trans.view(b, h, 3),
                    swf_of_pose[..., 3:].view(b, 1, 4).expand(b, h, 4),
                ],
                dim=-1,
            )

            wf_nf_rot, wf_nf_trans = torch_pose_multiply(
                wf_of_rot, wf_of_trans, of_nf_rot, of_nf_trans * wf_of_scale
            )

            wf_hc = torch_transform_points(of_hc, wf_of_rot, wf_of_trans, wf_of_scale)
            wf_op = torch_transform_points(of_op, wf_of_rot, wf_of_trans, wf_of_scale)
            wf_on = torch_transform_points(of_on, wf_of_rot)
            wf_ogc = torch_transform_points(
                of_ogc.view(b, 1, 1, 3), wf_of_rot, wf_of_trans, wf_of_scale
            )
            wf_hf_rot, wf_hf_trans = torch_pose_multiply(
                wf_nf_rot,
                wf_nf_trans,
                nf_hf_rot,
                nf_hf_trans.unsqueeze(-2),
            )

            wf_ogd = torch_transform_points(
                hf_ogd.unsqueeze(-2), wf_hf_rot, wf_hf_trans
            )
            if has_floor_z0 and override_ogd:
                wf_ogd = wf_ogd * 0.0
                wf_ogd[..., 2] = -1.0

            result_dict = {
                "grasp_pose": torch.cat(
                    [wf_hf_trans.squeeze(-2), torch_matrix_to_quaternion(wf_hf_rot)],
                    dim=-1,
                ).view(b * h, 7)[loss_valid_idx],
                "hand_contacts": wf_hc.view(b * h, -1, 6)[loss_valid_idx],
                "grasp_qpos": grasp_qpos.view(b * h, -1)[loss_valid_idx],
                "hand_evolution_num": hand_evolution_num.view(b * h)[loss_valid_idx],
                "obj_cp": wf_op.view(b * h, -1, 3)[loss_valid_idx],
                "obj_cn": wf_on.view(b * h, -1, 3)[loss_valid_idx],
                "obj_scale": opt_s.view(b * h, 1)[loss_valid_idx],
                "wm_id_lst": wm_id_lst[obj_id_lst],
                "obj_id_lst": obj_id_lst,
                "obj_pose": wf_of_pose.view(b * h, 7)[loss_valid_idx],
                "obj_gd": wf_ogd.view(b * h, 6)[loss_valid_idx],
                "obj_gc": wf_ogc.view(b * h, 3)[loss_valid_idx],
            }

        remain_number = loss_valid_idx.shape[0]
        if remain_number == 0:
            return result_dict

        if self.coll_filter["enable"]:
            coll_valid_idx = self.collision_based_filter(
                hf_hsk.view(b * h, -1, 3)[loss_valid_idx],
                result_dict["grasp_pose"],
                result_dict["obj_pose"],
                result_dict["obj_scale"],
                result_dict["wm_id_lst"],
                has_floor_z0,
            )
            for k, v in result_dict.items():
                result_dict[k] = v[coll_valid_idx]
            logging.debug(
                f"Collision filtering remain {coll_valid_idx.sum()} out of {remain_number}"
            )
            remain_number = coll_valid_idx.sum()
            if remain_number == 0:
                return result_dict

        if self.qp_filter["enable"]:
            qp_valid_idx = self.qp_based_filter(
                result_dict["obj_cp"],
                result_dict["obj_cn"],
                result_dict["obj_gd"] * unified_obj_mass,
                result_dict["obj_gc"],
                self.qp_filter["force_closure"],
            )
            for k, v in result_dict.items():
                result_dict[k] = v[qp_valid_idx]
            logging.debug(
                f"QP filtering remain {qp_valid_idx.sum()} out of {remain_number}"
            )
            remain_number = qp_valid_idx.sum()
            if remain_number == 0:
                return result_dict

        if self.fps_filter["enable"]:
            fps_valid_idx = self.fps_based_filter(
                result_dict["obj_cp"],
                result_dict["obj_pose"],
                result_dict["obj_id_lst"],
                self.fps_filter["max_num"],
                self.fps_filter["min_dist"],
            )
            for k, v in result_dict.items():
                result_dict[k] = v[fps_valid_idx]
            logging.debug(
                f"FPS filtering remain {fps_valid_idx.sum()} out of {remain_number}"
            )

        for k, v in result_dict.items():
            result_dict[k] = v.detach().cpu().numpy()

        return result_dict

    @torch.no_grad()
    def collision_based_filter(
        self,
        handframe_skeleton: torch.Tensor,
        grasp_pose: torch.Tensor,
        obj_pose: torch.Tensor,
        obj_scale: torch.Tensor,
        wm_id_lst,
        has_floor_z0: bool,
    ):
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

        if has_floor_z0:
            self.coll_buf[:query_shape] = (wf_hsk[..., -1] > 0.02).min(dim=-1)[0]

        of_hsk = torch_inv_transform_points(
            wf_hsk,
            torch_quaternion_to_matrix(obj_pose[..., 3:]),
            obj_pose[..., :3].view(-1, 1, 3),
            obj_scale.view(-1, 1, 1),
        )
        out_valid_idx = MeshLineSegCollision(
            of_hsk,
            wm_id_lst,
            self.coll_buf[:query_shape],
        )
        return out_valid_idx

    @torch.no_grad()
    def qp_based_filter(
        self, obj_points, obj_normals, gravity, gravity_center, force_closure
    ):
        if force_closure:
            gravity *= 0.0
        miu_coef = self.qp_filter["miu_coef"]
        qp_error = torch.ones(
            obj_points.shape[0], dtype=torch.float32, device=self.device
        )
        if self.qp_filter["batched"]:
            # logging.info(
            #     f"qp shape: {obj_points.shape} {self.qp_filter['max_batch_size']}"
            # )
            max_batch_size = self.qp_filter["max_batch_size"] // obj_points.shape[-2]
            batch_num = math.ceil(obj_points.shape[0] / max_batch_size)
            for i in range(batch_num):
                start = i * max_batch_size
                end = min((i + 1) * max_batch_size, obj_points.shape[0])

                _, qp_error[start:end] = get_qp_error_batched(
                    obj_points[start:end],
                    obj_normals[start:end],
                    gravity[start:end],
                    gravity_center[start:end],
                    miu_coef,
                )
        else:
            single_solver = ContactQP(miu_coef)
            for i in range(obj_points.shape[0]):
                _, qp_error[i] = single_solver.solve(
                    obj_points[i].cpu().numpy(),
                    obj_normals[i].cpu().numpy(),
                    gravity[i].cpu().numpy(),
                    gravity_center[i].cpu().numpy(),
                )

        out_valid_idx = qp_error < self.qp_filter["threshold"]
        return out_valid_idx

    @torch.no_grad()
    def fps_based_filter(self, obj_points, obj_pose, obj_id_lst, max_num, min_dist):
        of_op = torch_inv_transform_points(
            obj_points,
            torch_quaternion_to_matrix(obj_pose[..., 3:]),
            obj_pose[..., :3].view(-1, 1, 3),
        )
        out_valid_idx = torch.zeros(of_op.shape[0], device=self.device).bool()
        of_single_op = {}
        corr_index = {}
        for i in range(obj_id_lst.shape[0]):
            oi = str(obj_id_lst[i].data.item())
            if oi not in of_single_op.keys():
                of_single_op[oi] = [of_op[i]]
                corr_index[oi] = [i]
            else:
                of_single_op[oi].append(of_op[i])
                corr_index[oi].append(i)

        bh, n_points, _ = obj_points.shape
        for k, v in of_single_op.items():
            all_points = torch.stack(v, dim=0)

            rand_start = random.randint(0, all_points.shape[0] - 1)
            valid_points = all_points[rand_start].unsqueeze(0)
            out_valid_idx[corr_index[k][rand_start]] = True
            for i in range(min(len(all_points), max_num - 1)):
                farthest_dist, farthest_id = (
                    (valid_points.unsqueeze(0) - all_points.unsqueeze(1))
                    .norm(dim=-1)
                    .mean(dim=-1)
                    .min(dim=-1)[0]
                    .max(dim=-1)
                )
                if farthest_dist < min_dist:
                    break
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
    logging.warning(f"Hand template name: {configs.template_name}")

    obj_loader = get_object_dataloader(task_config.object, configs.n_worker)
    hand_library = HandTemplateLibrary(
        xml_path=configs.hand.xml_path,
        skeleton_path=configs.hand_skeleton_path,
        template_name=configs.template_name,
        init_template_dir=configs.init_template_dir,
        new_template_dir=(
            configs.new_template_dir if configs.update_template is not False else None
        ),
        max_data_buffer=configs.max_template_buffer,
        num_workers=configs.n_worker,
    )
    matcher = ObjectAligner(task_config.device, **task_config.matcher)

    for eee in range(task_config.epoch):
        logging.warning(f"Epoch {eee}")
        for obj_samples in obj_loader:
            obj_samples = {
                k: (
                    v.to(task_config.device, non_blocking=True)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in obj_samples.items()
            }

            wm_id_lst = matcher.load_warp_mesh(
                obj_samples["tm_mesh"], obj_samples["obj_name"]
            )

            sampled_rot, sampled_trans, sampled_scale = sample_init_poses(
                obj_samples["sampled_points"],
                obj_samples["sampled_normals"],
                task_config.object.init_inplane_num,
                task_config.object.scale_range,
            )

            hand_temp_dict = hand_library.get_batched_data(
                sampled_rot.shape[:2], task_config.device
            )

            result_dict = matcher.matching(
                hand_temp_dict["nf_hc"],
                hand_temp_dict["hf_hsk"],
                hand_temp_dict["hf_ogd"],
                hand_temp_dict["nf_hf_rot"],
                hand_temp_dict["nf_hf_trans"],
                hand_temp_dict["grasp_qpos"][..., 7:],
                hand_temp_dict["evolution_num"],
                sampled_rot,
                sampled_trans,
                sampled_scale,
                obj_samples["of_ogc"],
                obj_samples["swf_of_pose"],
                wm_id_lst,
                task_config.object.scale_range,
                configs.obj_mass,
                configs.has_floor_z0,
                configs.override_ogd,
            )
            succ_num = result_dict["grasp_pose"].shape[0]
            logging.warning(f"Get {succ_num} valid")
            if succ_num == 0:
                continue

            for (
                grasp_pose,
                grasp_qpos,
                hand_c,
                hand_evolution_num,
                obj_cp,
                obj_cn,
                obj_scale,
                obj_pose,
                obj_id,
                obj_gd,
                obj_gc,
            ) in zip(
                result_dict["grasp_pose"],
                result_dict["grasp_qpos"],
                result_dict["hand_contacts"],
                result_dict["hand_evolution_num"],
                result_dict["obj_cp"],
                result_dict["obj_cn"],
                result_dict["obj_scale"],
                result_dict["obj_pose"],
                result_dict["obj_id_lst"],
                result_dict["obj_gd"],
                result_dict["obj_gc"],
            ):

                obj_name = obj_samples["obj_name"][obj_id]
                obj_path = obj_samples["obj_path"][obj_id]

                grasp_dir = os.path.join(
                    configs.init_dir, hand_temp_dict["hand_template_name"], obj_name
                )
                os.makedirs(grasp_dir, exist_ok=True)
                count = len(os.listdir(grasp_dir)) + 1

                obj_worldframe_contacts = np.concatenate([obj_cp, -obj_cn], axis=-1)
                np.save(
                    f"{grasp_dir}/{eee}_{count}.npy",
                    {
                        "evolution_num": np_array32([hand_evolution_num]),
                        "hand_type": configs.hand_name,
                        "hand_template_name": hand_temp_dict["hand_template_name"],
                        "grasp_qpos": np.concatenate([grasp_pose, grasp_qpos]),
                        "hand_worldframe_contacts": hand_c,
                        "hand_contact_body_names": hand_temp_dict[
                            "hand_contact_body_names"
                        ],
                        "necessary_contact_body_names": hand_temp_dict[
                            "necessary_contact_body_names"
                        ],
                        "obj_worldframe_contacts": obj_worldframe_contacts,
                        "obj_path": obj_path,
                        "obj_scale": obj_scale,
                        "obj_pose": obj_pose,
                        "obj_pose_id": obj_samples["obj_pose_id"][obj_id].cpu().numpy(),
                        "obj_gravity_center": obj_gc,
                        "obj_gravity_direction": obj_gd,
                    },
                )

    return
