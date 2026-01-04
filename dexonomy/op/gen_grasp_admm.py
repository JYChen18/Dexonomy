import os
import multiprocessing as mp
import logging
import glob
import numpy as np
import torch
import trimesh
import mujoco
import time
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from typing import List,Tuple

from dexonomy.sim import MuJoCo_OptEnv, MuJoCo_OptCfg, HandCfg
from dexonomy.qp.qp_single import ContactQP, ContactQPTorch
from dexonomy.util.np_util import np_array32
from dexonomy.util.file_util import load_scene_cfg, safe_wrapper

def load_obj_trimesh(data: dict):
    scene_cfg = load_scene_cfg(data["scene_path"])["scene"]
    assert len(scene_cfg) == 1, "Only single object is supported now."
    obj_name, obj_cfg = list(scene_cfg.items())[0]
    assert obj_cfg['type'] == 'rigid_object', "Only rigid object is supported now"
    obj_mesh_tri = trimesh.load_mesh(obj_cfg['file_path'])
    # scale and translate
    obj_mesh_tri.apply_scale(obj_cfg['scale'])
    obj_tran = obj_cfg['pose'][:3]
    obj_rot = obj_cfg['pose'][3:]
    obj_rot = R.from_quat([obj_rot[1], obj_rot[2], obj_rot[3], obj_rot[0]]).as_matrix()
    obj_homo = np.eye(4)
    obj_homo[:3, :3] = obj_rot
    obj_homo[:3, 3] = obj_tran
    obj_mesh_tri.apply_transform(obj_homo)
    
    vertices = torch.tensor(obj_mesh_tri.vertices)
    normals = torch.tensor(obj_mesh_tri.vertex_normals)
    faces = torch.LongTensor(obj_mesh_tri.faces)
    
    return obj_mesh_tri, vertices, normals, faces

def get_closest_triangles(mesh: trimesh.Trimesh, 
                          points: np.ndarray,
                          vertices: torch.Tensor = None, 
                          normals: torch.Tensor = None,
                          faces: torch.LongTensor = None
                          ):
    closest, dist, tri_ids = trimesh.proximity.closest_point(mesh, points)
    if vertices is None:
        return closest, dist

    tri_vert = vertices[faces[tri_ids]]
    tri_norm = normals[faces[tri_ids]]
    
    return closest, dist, tri_vert, tri_norm

def point_to_triangle(points, triangles, normals=None, eps=1e-20):
    """
    Batched shortest distance from points to triangles in 3D.
    Also returns the closest points on the triangles.
    Handles degenerate triangles robustly.

    Args:
        points: (N,3) tensor of points
        tris:   (N,3,3) tensor of triangles
        eps:    small value to avoid division by zero

    Returns:
        distances: (N,) tensor of distances
        closest_points: (N,3) tensor of closest points on triangles
    """
    # Extract vertices
    A, B, C = triangles.unbind(dim=1)  # each (N, 3)
    NA, NB, NC = None, None, None
    if normals is not None:
        NA, NB, NC = normals.unbind(dim=1)  # each (N, 3)

    # Compute edges
    AB = B - A
    AC = C - A
    BC = C - B

    # Vector from A to point
    AP = points - A

    # Compute barycentric coordinates
    dot00 = torch.sum(AB * AB, dim=1)
    dot01 = torch.sum(AB * AC, dim=1)
    dot02 = torch.sum(AB * AP, dim=1)
    dot11 = torch.sum(AC * AC, dim=1)
    dot12 = torch.sum(AC * AP, dim=1)

    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + eps)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    # Check if inside triangle
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)

    # Projection onto plane
    proj = A + u.unsqueeze(1) * AB + v.unsqueeze(1) * AC

    # Closest points on edges
    # Edge AB
    t_ab = torch.clamp(torch.sum(AP * AB, dim=1) / (dot00 + eps), 0, 1)
    p_ab = A + t_ab.unsqueeze(1) * AB

    # Edge AC
    t_ac = torch.clamp(torch.sum(AP * AC, dim=1) / (dot11 + eps), 0, 1)
    p_ac = A + t_ac.unsqueeze(1) * AC

    # Edge BC
    BP = points - B
    dot_bc = torch.sum(BC * BC, dim=1)
    t_bc = torch.clamp(torch.sum(BP * BC, dim=1) / (dot_bc + eps), 0, 1)
    p_bc = B + t_bc.unsqueeze(1) * BC

    # Compute distances
    dist_proj = torch.norm(points - proj, dim=1)
    dist_ab = torch.norm(points - p_ab, dim=1)
    dist_ac = torch.norm(points - p_ac, dim=1)
    dist_bc = torch.norm(points - p_bc, dim=1)

    # Find minimum distance for outside points
    dist_out, idx_out = torch.min(
        torch.stack([dist_ab, dist_ac, dist_bc], dim=1), dim=1
    )
    closest_out = torch.stack([p_ab, p_ac, p_bc], dim=1)[
        torch.arange(points.shape[0]), idx_out
    ]
    
    # compute u, v, w
    # u * AB + v * AC = w * A + u * B + v * C
    
    # Edge AB: (u, v) = (t_ab, 0)
    u_ab = t_ab
    v_ab = torch.zeros_like(t_ab)
    
    # Edge AC: (u, v) = (0, t_ac)
    u_ac = torch.zeros_like(t_ac)
    v_ac = t_ac
    
    # Edge BC: (u, v) = (1 - t_bc, t_bc)
    u_bc = 1 - t_bc
    v_bc = t_bc
    
    u_out = torch.stack([u_ab, u_ac, u_bc], dim=1)[
        torch.arange(points.shape[0]), idx_out
    ]
    v_out = torch.stack([v_ab, v_ac, v_bc], dim=1)[
        torch.arange(points.shape[0]), idx_out
    ]
    

    # Final results
    distances = torch.where(inside, dist_proj, dist_out)
    closest_points = torch.where(inside.unsqueeze(1), proj, closest_out)
    
    uv_coords = torch.stack([
        torch.where(inside, u, u_out),
        torch.where(inside, v, v_out)
    ], dim=1)
    
    if normals is not None:
        normals_intp = (1 - uv_coords[:, 0] - uv_coords[:, 1]).unsqueeze(1) * NA \
          + uv_coords[:, 0].unsqueeze(1) * NB + uv_coords[:, 1].unsqueeze(1) * NC
        normals_intp = normals_intp / (torch.norm(normals_intp, dim=1, keepdim=True) + eps)
        return distances, closest_points, normals_intp
    else:
        return distances, closest_points

def point_to_triangle_simple(points, triangles, normals=None, eps=1e-20):
    # Extract vertices
    A, B, C = triangles.unbind(dim=1)  # each (N, 3)
    NA, NB, NC = None, None, None
    if normals is not None:
        NA, NB, NC = normals.unbind(dim=1)  # each (N, 3)

    # Compute edges
    AB = B - A
    AC = C - A
    BC = C - B

    # Vector from A to point
    AP = points - A

    # Compute barycentric coordinates
    dot00 = torch.sum(AB * AB, dim=1)
    dot01 = torch.sum(AB * AC, dim=1)
    dot02 = torch.sum(AB * AP, dim=1)
    dot11 = torch.sum(AC * AC, dim=1)
    dot12 = torch.sum(AC * AP, dim=1)
    
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + eps)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    proj = A + u.unsqueeze(1) * AB + v.unsqueeze(1) * AC
    distance = torch.norm(points - proj, dim=-1, keepdim=True)

    if normals is not None:
        normals_intp = (1 - u - v).unsqueeze(1) * NA \
          + u.unsqueeze(1) * NB + v.unsqueeze(1) * NC
        normals_intp = normals_intp / (torch.norm(normals_intp, dim=1, keepdim=True) + eps)
        return distance, proj, normals_intp
    else:
        return distance, proj

def create_hand_trimesh(
    env: MuJoCo_OptEnv
):
    meshes = []
    for geom_id in range(env._model.ngeom):
        geom_type = env._model.geom_type[geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = env._model.geom_dataid[geom_id]
            vertex_start = env._model.mesh_vertadr[mesh_id]
            vertex_num   = env._model.mesh_vertnum[mesh_id]
            face_start   = env._model.mesh_faceadr[mesh_id]
            face_num     = env._model.mesh_facenum[mesh_id]
            vertices = env._model.mesh_vert[vertex_start : vertex_start + vertex_num]
            faces = env._model.mesh_face[face_start : face_start + face_num]
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            meshes.append(mesh)
        else:
            meshes.append(None)
    return meshes


def project_to_hand(
  pts: np.ndarray, # (N, 3)
  hand_cbody: List[str], # (N,)
  env: MuJoCo_OptEnv,
  meshes: List[trimesh.Trimesh],
  geom_filter: str = "all"
  ):
    q_world = np.zeros_like(pts)
    q_body = np.zeros_like(pts)
    for i in range(pts.shape[0]):
        body_id = env._model.body(env._cfg.hand_prefix + hand_cbody[i]).id
        b_r = env._data.xmat[body_id].reshape(3, 3)
        b_t = env._data.xpos[body_id]
        geom_ids = np.where(env._model.geom_bodyid == body_id)[0]
        if geom_ids.size == 0:
            raise RuntimeError(f"Body '{hand_cbody}' has no geoms")
        for g in geom_ids:
            is_visual = (env._model.geom_contype[g] == 0) and (env._model.geom_conaffinity[g] == 0)
            if geom_filter == "visual" and not is_visual:
                continue
            if geom_filter == "collision" and is_visual:
                continue
            g_id = g
            break
        else:
            raise RuntimeError("No geom left after filter")
        
        g_r = R.from_quat(env._model.geom_quat[g_id], scalar_first=True).as_matrix().reshape(3, 3)
        g_t = env._model.geom_pos[g_id]
        
        g_t = b_t + b_r @ g_t
        g_r = b_r @ g_r
        
        g_type = env._model.geom_type[g_id]
        g_size = env._model.geom_size[g_id]
        
        p_local = (pts[i] - g_t) @ g_r
        p_local = p_local.reshape(1, 3)
        if g_type == mujoco.mjtGeom.mjGEOM_BOX:
            h = g_size[:3]
            q_local = np.clip(p_local, -h, h)
            d = np.maximum(h - np.abs(q_local), 0.)
            k = np.argmin(d, axis=-1, keepdims=True) # which axis to project (N, 1)
            s = np.sign(np.take_along_axis(q_local, k, -1)) \
              * np.take_along_axis(h[None,...], k, -1) # project to +h/-h (N, 1)
            np.put_along_axis(q_local, k, s, -1)
            # n_local = np.zeros_like(q_local)
            # np.put_along_axis(n_local, k, np.sign(s), -1)
            # TODO: corner case
        elif g_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = g_size[0]
            d = np.linalg.norm(p_local, axis=-1, keepdims=True)
            q_local = p_local * (r / (d + 1e-9))
            # n_local = p_local / (d + 1e-9)
        elif g_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r, h = g_size[0], g_size[1]
            xy = p_local[:, :2]
            z = p_local[:, 2:3]
            dxy = np.linalg.norm(xy, axis=-1, keepdims=True)
            z_clipped = np.clip(z, -h, h)
            xy_clipped = np.where(dxy > r, xy / (dxy + 1e-9) * r, xy)
            q_local = np.concatenate([xy_clipped, z_clipped], axis=-1)
            dist_to_side = np.maximum(r - dxy, 0)
            dist_to_cap = np.maximum(h - np.abs(z), 0)
            dists = np.concatenate([dist_to_side, dist_to_cap], axis=-1)
            k = np.argmin(dists, axis=-1, keepdims=True) # 0: side, 1: cap
            cap_proj_z = np.sign(z) * h
            side_proj_xy = xy / (dxy + 1e-9) * r
            final_xy = np.where(k == 0, side_proj_xy, xy_clipped)
            final_z = np.where(k == 1, cap_proj_z, z_clipped)
            q_local = np.concatenate([final_xy, final_z], axis=-1)
        elif g_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            r, h = g_size[0], g_size[1]
            xy = p_local[:, :2]
            z = p_local[:, 2]
            to_side = np.abs(z) <= h
            # if to side
            dxy = np.linalg.norm(xy, axis=-1)
            xy_side = xy / (dxy + 1e-9)[...,None] * r
            q_side = np.concatenate([xy_side, z[...,None]], axis=-1)
            # if to cap
            z_clipped = np.clip(z, -h, h)
            dcap = np.sqrt((z - z_clipped)**2 + dxy**2)
            xy_cap = xy / (dcap[...,None] + 1e-9) * r
            z_cap = (z - z_clipped) / (dcap + 1e-9) * r + z_clipped
            q_cap = np.concatenate([xy_cap, z_cap[...,None]], axis=-1)
            q_local = np.where(to_side[..., None], q_side, q_cap)
        elif g_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = meshes[g_id]
            q_local, dist, _ = trimesh.proximity.closest_point(mesh, p_local)
        else:
            raise NotImplementedError(f"geom type {g_type} not supported yet")
            
        q_world[i] = (q_local @ g_r.T + g_t).reshape(3)
        q_body[i] = ((q_world[i] - b_t) @ b_r).reshape(3)
    return q_world, q_body



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
        ret = self.sim_env.check_qpos_limit(self.cfg.limit.thre)
        if not ret:
            logging.debug(f"{self.log_name} qpos limit")
        return ret

def init_worker():
    OmegaConf.register_new_resolver("hydra", lambda *args: None, replace=True)

@safe_wrapper
def _single_grasp(param):
    input_path, cfg = param[0], param[1]
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    if cfg.op.grasp.device == "cuda":
        gpu_id = param[2]
        if torch.cuda.is_available():
            torch.cuda._initialized = False
            torch.cuda.init()
            device = torch.device(f"cuda:{gpu_id}")
            logging.debug(f"Worker {os.getpid()}: GPU {gpu_id} ready, "
                        f"visible devices: {torch.cuda.device_count()}")
        else:
            torch.set_default_device("cpu")
            device = torch.device('cpu')
            logging.warning(f"Worker {os.getpid()}: CUDA not available, using CPU")
        torch.set_default_device(device)
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
    
    qpsolver = ContactQPTorch(grasp_cfg.opt_miu_coef)

    grasp_filter = GraspFilter(grasp_cfg.filter, sim_env, f"{input_path} grasp")
    pregrasp_filter = GraspFilter(
        pregrasp_cfg.filter, sim_env, f"{input_path} pregrasp"
    )

    # Generate grasp qpos
    hand_cbody, obj_cpn_w = grasp_data["hand_cbody"], grasp_data["obj_cpn_w"]
    obj_mesh, obj_vert, obj_norm, obj_face = load_obj_trimesh(grasp_data)
    hand_cpn_w = grasp_data["hand_cpn_w"]
    hand_cpn_b = sim_env.transform_cpn_w2b(hand_cbody, grasp_data["hand_cpn_w"])
    hand_meshes = create_hand_trimesh(sim_env)
    NC = len(hand_cpn_b)
    cp_o = torch.zeros((NC, 3), requires_grad=False)
    cp_h = torch.zeros((NC, 3), requires_grad=False)
    lam = torch.zeros((NC, 3), requires_grad=False)
    cp_o.copy_(torch.from_numpy(obj_cpn_w[:, :3]))
    cp_h.copy_(torch.from_numpy(hand_cpn_w[:, :3]))
    lam.zero_()
    rho = grasp_cfg.rho_admm
    alpha = grasp_cfg.lr_obj
    
    
    # time_step1 = 0.
    # time_step2 = 0.
    # time_forward = 0.
    # time_backward = 0.
    # res_all = []
    terminated = False
    for iter in range(grasp_cfg.step):
        cp = torch.zeros((NC, 3), requires_grad=True)
        with torch.no_grad(): cp[:] = cp_o
        # ADMM step 1
        # pre_time = time.time()
        _, _, tri_vert, tri_norm = get_closest_triangles(
            obj_mesh, cp.detach().cpu().numpy(), obj_vert, obj_norm, obj_face
        )
        for ii in range(grasp_cfg.step_obj):
            # pre_time_forward = time.time()
            cp.grad = None
            _, cp_o, cn_o = point_to_triangle_simple(
                cp, tri_vert, tri_norm
            )
            cn_o = - cn_o
            try:
                res = qpsolver.solve(
                    cp_o,
                    cn_o,
                    grasp_data['ext_wrench'],
                    grasp_data['ext_center'],
                )
                if res == None:
                    terminated = True
                    break
            except:
                terminated = True
                break
            # res_all.append(res.detach().item())
            # time_forward += time.time() - pre_time_forward
            # pre_time_backward = time.time()
            res.backward()
            with torch.no_grad():
                rho = grasp_cfg.rho_admm
                cp.grad += rho * (cp - cp_h - lam)
                # project grad to plane
                cp.grad -= torch.sum(cp.grad * cn_o, dim=-1, keepdim=True) * cn_o
                cp -= alpha * cp.grad
                closest, _, tri_vert, tri_norm = get_closest_triangles(obj_mesh, cp.detach().cpu().numpy(), obj_vert, obj_norm, obj_face)
                cp.copy_(torch.from_numpy(closest))
                cp_o[:] = cp
            # time_backward += time.time() - pre_time_backward
        # end_time = time.time()
        # time_step1 += end_time - pre_time
        if terminated: break
        # ADMM step 2
        # pre_time = time.time()
        with torch.no_grad():
            cp = cp_o - lam
            if not grasp_cfg.fix_hand_local_cp:
                _, tmp = project_to_hand(cp.detach().cpu().numpy(), hand_cbody, sim_env, hand_meshes, "visual")
                hand_cpn_b[:, :3] = tmp
        for ii in range(grasp_cfg.step_hand):
            curr_diff = sim_env.apply_contact_forces_no_normal(hand_cbody, hand_cpn_b, cp.detach().cpu().numpy())
            sim_env.step_sim(grasp_cfg.substep)
            if ii == 0 or np.max(prev_diff - curr_diff) > 1e-4:
                prev_diff = np.copy(curr_diff)
                continue
            else:
                prev_diff = np.copy(curr_diff)
            # if grasp_filter.early_stop():
            #     break
        # end_time = time.time()
        # time_step2 += end_time - pre_time
        # ADMM step 3
        with torch.no_grad():
            cpn_h = sim_env.transform_cpn_b2w(hand_cbody, hand_cpn_b)
            cp_h.copy_(torch.from_numpy(cpn_h[:, :3]))
            lam += cp_h - cp_o

    if terminated: # fallback
        sim_env.reset_qpos(grasp_data["grasp_qpos"][0])
        hand_cbody, obj_cpn_w = grasp_data["hand_cbody"], grasp_data["obj_cpn_w"]
        hand_cpn_b = sim_env.transform_cpn_w2b(hand_cbody, grasp_data["hand_cpn_w"])
        for ii in range(20):
            curr_diff = sim_env.apply_contact_forces(hand_cbody, hand_cpn_b, obj_cpn_w)
            sim_env.step_sim(10)
            if ii == 0 or np.max(prev_diff - curr_diff) > 1e-4:
                prev_diff = np.copy(curr_diff)
                continue
            else:
                prev_diff = np.copy(curr_diff)
            if grasp_filter.early_stop():
                break
    
    grasp_data["grasp_qpos"] = np_array32(sim_env.get_hand_qpos())[None]

    # print(f"Time step 1: {time_step1:.3f}, Time step 2: {time_step2:.3f}")
    # print(f"Time forward: {time_forward:.3f}, Time backward: {time_backward:.3f}")
    # print(input_path)
    # print(res_all[-1])

    succ_flag, ho_c = grasp_filter.forward(grasp_data)
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


def operate_grasp_admm(cfg: dict):
    n_worker = cfg.n_worker
    gpu_slots = []
    if cfg.op.grasp.device == "cuda":
        mp.set_start_method('spawn', force=True)
        n_gpu = len(cfg.op.grasp.gpus)
        process_per_gpu = cfg.op.grasp.process_per_gpu
        n_worker = n_gpu * process_per_gpu
        for gpu_id in cfg.op.grasp.gpus:
            gpu_slots.extend([gpu_id] * process_per_gpu)
    else:
        torch.set_default_device("cpu")
  
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

    cfg_dict = OmegaConf.to_container(cfg, resolve=False)
    if cfg.op.grasp.device == "cuda":
        param_lst = []
        for i, path in enumerate(input_path_lst):
            worker_idx = i % n_worker
            gpu_id = gpu_slots[worker_idx]
            param_lst.append((path, cfg_dict, gpu_id))
    else:
        param_lst = zip(input_path_lst, [cfg] * len(input_path_lst))
    if n_worker == 1 or cfg.debug_view:
        for ip in param_lst:
            _single_grasp(ip)
    else:
        initializer = init_worker if cfg.op.grasp.device == "cuda" else None
        with mp.Pool(processes=n_worker, initializer=initializer) as pool:
            jobs = pool.imap_unordered(_single_grasp, param_lst)
            results = list(jobs)
            # Log the processed paths
            write_mode = "a" if cfg.skip_done else "w"
            with open(cfg.log_path, write_mode) as f:
                f.write("\n".join(results) + "\n")

    logging.info(f"Finish grasp generation")

    return
