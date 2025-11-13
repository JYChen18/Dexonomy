from typing import Tuple, List
import os
import math
import numpy as np
import quaternion
import scipy
import logging
import warp as wp
import warp.sim
import warp.sim.render
import torch
import mujoco
import trimesh
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
from scipy.spatial.transform import Rotation as R
from dexonomy.util.file_util import load_scene_cfg
from dexonomy.util.np_util import np_normalize_vector, np_normal_to_rot

from utils.vis_util import points_to_obj, arrows_to_obj
from utils.warp_util import add_arr1D_wp, axbyz_arr1D_wp, axpy_arr1D_wp
from sim_mujoco import MuJoCo_OptQEnv, MuJoCo_OptQCfg, HandCfg
from utils.import_mjcf_custom import parse_mjcf

import hydra
from omegaconf import DictConfig
from utils.log_util import set_logging
from loader import load_instance

FIX_HAND_LOCAL_CP = True

torch.set_default_device("cpu")

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

def create_hand_trimesh(
    env: MuJoCo_OptQEnv
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
  env: MuJoCo_OptQEnv,
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
            k = np.argmin(d, axis=-1, keepdims=True) # which axis to project
            s = np.sign(np.take_along_axis(q_local, k, -1)) \
              * np.take_along_axis(h[None,...], k, -1) # project to +h/-h
            np.put_along_axis(q_local, k, s, -1)
        elif g_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = g_size[0]
            d = np.linalg.norm(p_local, axis=-1, keepdims=True)
            q_local = p_local * (r / (d + 1e-9))
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

def init_sim_mujoco(cfg: dict, grasp_data: dict):
    sim_env = MuJoCo_OptQEnv(
        hand_cfg=HandCfg(xml_path=cfg.hand.xml_path, freejoint=True),
        scene_cfg=load_scene_cfg(grasp_data["scene_path"]),
        sim_cfg=MuJoCo_OptQCfg(obj_margin=cfg.cdist),
        debug_render=cfg.debug_render,
        debug_view=cfg.debug_view,
    )
    sim_env.reset_qpos(grasp_data["grasp_qpos"][0])
    sim_env.set_obj_margin(cfg.cdist)
    return sim_env

wp.config.quiet = True
wp.init()

def qpos_mj2wp(qpos: np.ndarray):
    # input: qpos or grad_qpos, real part first
    ret = qpos.copy()
    # don't know why, but this works
    r1 = R.from_rotvec(-np.array([0, 1, 0]) * np.pi / 2)
    r2 = R.from_rotvec(np.array([0, 0, 1]) * np.pi)
    rq = (r2 * r1).as_quat(scalar_first=True)
    
    fq = quaternion.as_quat_array(qpos[3:7]) * quaternion.as_quat_array(rq)
    ret[3:7] = quaternion.as_float_array(fq)
    
    # move real part to the back
    w = ret[3]
    ret[3:6] = ret[4:7]
    ret[6] = w
    
    return ret

def qpos_wp2mj(qpos: np.ndarray):
    # input: qpos or grad_qpos, real part last
    ret = qpos.copy()
    # move real part to the front
    ret[4:7] = qpos[3:6]
    ret[3] = qpos[6]
    # don't know why, but this works
    r1 = R.from_rotvec(-np.array([0, 1, 0]) * np.pi / 2)
    r2 = R.from_rotvec(np.array([0, 0, 1]) * np.pi)
    rq = (r2 * r1).as_quat(scalar_first=True)
    
    fq = quaternion.as_quat_array(ret[3:7]) / quaternion.as_quat_array(rq)
    ret[3:7] = quaternion.as_float_array(fq)
    
    return ret

# test_qpos_mj = np.zeros(23)
# test_qpos_mj[3:7] = np.random.rand(4) * 1e-3
# test_qpos_wp = qpos_mj2wp(test_qpos_mj)
# test_qpos_mj2 = qpos_wp2mj(test_qpos_wp)
# print ("test mj2wp and wp2mj")
# print(test_qpos_mj[3:7], test_qpos_wp[3:7], test_qpos_mj2[3:7])

def init_sim_wp(cfg: dict, grasp_data: dict):
    builder = wp.sim.ModelBuilder()
    # hand
    parse_mjcf(
        cfg.hand.xml_path,
        builder,
        visual_classes=['palm_visual', 'base_visual', 'proximal_visual', 'medial_visual', 'distal_visual', 'fingertip_visual', 'thumbtip_visual', 'visual'],
        # collider_classes=["plastic_collision"],
        floating=True,
        density=1e5,  # NOTE: If density==1e6, the simluation will be unstable
        armature=0.01,
        stiffness=1,  # NOTE: If stiffness>=10, the simluation will be unstable
        damping=1,
        contact_ke=1,
        contact_kd=1,
        contact_kf=1,
        contact_mu=1.0,
        contact_restitution=0.0,
        # up_axis="Y",
        verbose=True
    )
    model = builder.finalize("cuda")
    model.requires_grad = True
    model.ground = False
    
    state = model.state()
    model.joint_q.requires_grad = True
    state.body_q.requires_grad = True
    
    logging.debug(f"Model has {model.body_count} bodies and {model.joint_count} joints.")
    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        joint_types = model.joint_type.numpy()
        joint_names = model.joint_name if hasattr(model, 'joint_name') else [f"joint_{i}" for i in range(model.joint_count)]
        for i in range(model.joint_count):
            joint_type = joint_types[i]
            joint_name = joint_names[i]

            if joint_type == wp.sim.JOINT_REVOLUTE:
                print(f"{joint_name}: JOINT_REVOLUTE")
            elif joint_type == wp.sim.JOINT_PRISMATIC:
                print(f"{joint_name}: JOINT_PRISMATIC")
            elif joint_type == wp.sim.JOINT_BALL:
                print(f"{joint_name}: JOINT_BALL")
            elif joint_type == wp.sim.JOINT_FIXED:
                print(f"{joint_name}: JOINT_FIXED")
            elif joint_type == wp.sim.JOINT_FREE:
                print(f"{joint_name}: JOINT_FREE")
            elif joint_type == wp.sim.JOINT_UNIVERSAL:
                print(f"{joint_name}: JOINT_UNIVERSAL")
            elif joint_type == wp.sim.JOINT_COMPOUND:
                print(f"{joint_name}: JOINT_COMPOUND")
            elif joint_type == wp.sim.JOINT_D6:
                print(f"{joint_name}: JOINT_D6")
            else:
                print(f"{joint_name}: UNKNOWN_JOINT_TYPE {joint_type}")
    
    # assign initial qpos
    qpos = grasp_data["grasp_qpos"][0]
    qpos_wp = qpos_mj2wp(qpos)
    
    model.joint_q.assign(qpos_wp)
    state.joint_q.assign(qpos_wp)

    wp.sim.eval_fk(model, model.joint_q, model.joint_qd, None, state)
    
    # load object mesh
    scene_cfg = load_scene_cfg(grasp_data["scene_path"])["scene"]
    assert len(scene_cfg) == 1, "Only single object is supported in warp sim."
    obj_name, obj_cfg = list(scene_cfg.items())[0]
    assert obj_cfg['type'] == 'rigid_object', "Only rigid object is supported in warp sim."
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
    obj_normals = obj_mesh_tri.vertex_normals.astype(np.float32)
    obj_mesh_verts = wp.array(
        data=obj_mesh_tri.vertices,
        dtype=wp.vec3,
    )
    obj_mesh_inds = wp.array(
        data=obj_mesh_tri.faces.flatten(),
        dtype=int,
    )
    obj_mesh_normals = wp.array(
        data=obj_normals,
        dtype=wp.vec3,
    )
    obj_mesh = wp.Mesh(points=obj_mesh_verts, 
                       indices=obj_mesh_inds, 
                       velocities=obj_mesh_normals)
    obj_mesh.refit()
    
    return model, state, obj_mesh

def get_body_mesh_map(model: wp.sim.Model):
    mesh_ids = []
    geos = model.shape_geo.source.numpy()
    for i in range(model.body_count):
        shapes = model.body_shapes[i]
        assert len(shapes) == 1, "Multiple shapes in one body is not supported."
        mesh_ids.append(geos[shapes[0]])
    mesh_ids = wp.array(data=mesh_ids, dtype=wp.uint64)
    
    return mesh_ids
  
# optimization target
class QPSingle(torch.autograd.Function):
    # on cpu
    @staticmethod
    def forward(ctx,
                points: torch.Tensor, # (N, 3)
                normals: torch.Tensor, # (N, 3)
                gravity: np.ndarray, # (6,)
                gravity_center: np.ndarray, # (6,)
                G_matrix: np.ndarray,
                E_matrix: np.ndarray,
                h_matrix: np.ndarray,
                solver_type: str,
                ):
        points_np = points.detach().cpu().numpy()
        normals_np = normals.detach().cpu().numpy()
        rot = np_normal_to_rot(normals_np)
        axis_0, axis_1, axis_2 = rot[..., 0], rot[..., 1], rot[..., 2]
        relative_pos_np = points_np - gravity_center[None]
        grasp_matrix = np.zeros((points_np.shape[0], 6, 6))
        grasp_matrix[:, :3, 0] = grasp_matrix[:, 3:, 3] = axis_0
        grasp_matrix[:, :3, 1] = grasp_matrix[:, 3:, 4] = axis_1
        grasp_matrix[:, :3, 2] = grasp_matrix[:, 3:, 5] = axis_2
        grasp_matrix[:, 3:, 0] = np.cross(relative_pos_np, axis_0, axis=-1)
        grasp_matrix[:, 3:, 1] = np.cross(relative_pos_np, axis_1, axis=-1)
        grasp_matrix[:, 3:, 2] = np.cross(relative_pos_np, axis_2, axis=-1)

        param2force = grasp_matrix @ E_matrix  # [n, 6, 6]
        flatten_param2force = np.transpose(param2force, (1, 0, 2)).reshape(6, -1)
        
        P_matrix = flatten_param2force.T @ flatten_param2force
        q_matrix = gravity @ flatten_param2force
        
        solution = solve_qp(
            P=scipy.sparse.csc_matrix(P_matrix),
            q=q_matrix,
            G=scipy.sparse.csc_matrix(G_matrix),
            h=h_matrix,
            solver=solver_type,
        )
        # print(solution)
        if solution is None:
            return None, 1.0
          
        solution = solution.reshape(-1, 6)
        contact_wrenches_np = (param2force @ solution[..., None]).squeeze(
            axis=-1
        )  # [n, 6]
        f_param_np = (E_matrix @ solution[..., None]).squeeze(axis=-1)  # [n, 6]
        f_param = torch.from_numpy(f_param_np).to(torch.float32).to(points.device)
        
        wrench_error = np.linalg.norm(np.sum(contact_wrenches_np, axis=0) + gravity)
        relative_pos = torch.from_numpy(relative_pos_np).to(torch.float32).to(points.device)
        contact_wrenches = torch.from_numpy(contact_wrenches_np).to(torch.float32).to(points.device)
        wrench_res = torch.sum(contact_wrenches, dim=0) + torch.from_numpy(gravity).to(torch.float32).to(points.device)  # (6,)
        
        ctx.save_for_backward(relative_pos, f_param, contact_wrenches, wrench_res)
        
        return torch.tensor(wrench_error, dtype=torch.float32, device=points.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        # gradient of squared wrench error * 0.5
        relative_pos, f_param, contact_wrenches, wrench_res = ctx.saved_tensors
        # grad to points
        grad_points = torch.cross(contact_wrenches[:, 3:], wrench_res[None, :3], dim=-1)
        # grad to normals
        grad_normals = (wrench_res[None, :3] - torch.cross(relative_pos, wrench_res[None, 3:], dim=-1)) * f_param[:, 0][:, None]
        return grad_points, grad_normals, None, None, None, None, None, None

class QPSingleSolver:
    def __init__(self, miu_coef, solver_type="clarabel"):
        self.miu_coef = miu_coef
        self.solver_type = solver_type
        self.num_contact = -1
        return
      
    def _build_constraint(self):
        """
        Build G matrix and h matrix for constraints Gx <= h,
        using soft contact model with pyramid discretization.

        """
        num_f_strength = self.num_contact * 6
        G_matrix = np.zeros((num_f_strength + self.num_contact + 1, num_f_strength))
        h_matrix = np.zeros((num_f_strength + self.num_contact + 1))

        # - force <= 0
        G_matrix[range(0, num_f_strength), range(0, num_f_strength)] = -1.0
        # - force <= - eps
        h_matrix[range(0, num_f_strength)] = - 1e-2

        # pressure <= 1
        for i in range(self.num_contact):
            G_matrix[num_f_strength + i, 6 * i : 6 * i + 6] = 1.0
        h_matrix[-self.num_contact - 1 : -1] = 1.0

        # - sum pressure <= -0.1
        G_matrix[-1, :] = -1.0
        # remove this constraint by commenting out the following line
        # h_matrix[-1] = -1.0

        # https://mujoco.readthedocs.io/en/stable/_images/contact_frame.svg
        E_matrix = np.zeros((self.num_contact, 6, 6))
        E_matrix[:, 0, :] = 1
        E_matrix[:, 1, 0] = E_matrix[:, 2, 2] = self.miu_coef[0]
        E_matrix[:, 1, 1] = E_matrix[:, 2, 3] = -self.miu_coef[0]
        E_matrix[:, 3, 4] = self.miu_coef[1]
        E_matrix[:, 3, 5] = -self.miu_coef[1]

        return G_matrix, h_matrix, E_matrix
    
    def set_num_contact(self, num_contact):
        if self.num_contact != num_contact:
            self.num_contact = num_contact
            self.G_matrix, self.h_matrix, self.E_matrix = self._build_constraint()
        self.num_contact = num_contact

    def solve(
        self,
        pos: torch.Tensor,  # (N, 3)
        normal: torch.Tensor,  # (N, 3)
        gravity,
        gravity_center,
    ):
        self.set_num_contact(pos.shape[0])
        wrench_error = QPSingle.apply(
            pos,
            normal,
            gravity,
            gravity_center,
            self.G_matrix,
            self.E_matrix,
            self.h_matrix,
            self.solver_type,
        )
        return wrench_error

def init_local_contact(data: dict,
                       env_mujoco: MuJoCo_OptQEnv,
                       model_wp: wp.sim.Model
                       ):
    hand_cbody, obj_cpn_w = data["hand_cbody"], data["obj_cpn_w"]
    hand_cpn_b_np = env_mujoco.transform_cpn_w2b(hand_cbody, data["hand_cpn_w"])
    hand_cpn_b = wp.array(hand_cpn_b_np, dtype=float)
    
    cbody_ids = []
    body_name_wp = model_wp.body_name
    for cb in hand_cbody:
        # cid = sim_env._model.body(sim_env._cfg.hand_prefix + cb).id
        cid = body_name_wp.index(cb)
        cbody_ids.append(cid)
    cbody_ids = wp.array(cbody_ids, dtype=wp.int32)
    
    return hand_cpn_b, cbody_ids, hand_cbody

@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    set_logging(cfg.verbose)
    data = load_instance(cfg)

    sim_env = init_sim_mujoco(cfg, data)
    hand_meshes = create_hand_trimesh(sim_env)
    obj_mesh, obj_vert, obj_norm, obj_face = load_obj_trimesh(data)
    model_wp, state_wp, obj_wp = init_sim_wp(cfg, data)
    hand_cpn_b, cbody_ids, hand_cbody = init_local_contact(data, sim_env, model_wp)
    hand_cpn_b_np = hand_cpn_b.numpy()
    
    NC = len(hand_cpn_b)
    
    cp_h_wp = wp.array(dtype=wp.vec3, shape=len(hand_cpn_b), requires_grad=True) # contact points on hand
    cp_h = torch.zeros((NC, 3), requires_grad=True)
    lam = torch.zeros((NC, 3), requires_grad=False)
    cp_o_cpu = torch.zeros((NC, 3), requires_grad=False)
    
    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        print("contact points from mujoco:")
        print(data["hand_cpn_w"][:, :3])
        print("contact points from warp:")
        print(cp_h_wp)

    qpsolver = QPSingleSolver(cfg.miu_coeff)

    with torch.no_grad():
        cpn_h = sim_env.transform_cpn_b2w(hand_cbody, hand_cpn_b_np)
        cp_h.copy_(torch.from_numpy(cpn_h[:, :3]))
        closest, *_ = get_closest_triangles(obj_mesh, cp_h.detach().numpy())
        cp_o_cpu.copy_(torch.from_numpy(closest))

    renderer = wp.sim.render.SimRenderer(model_wp, "opt.usd")
    renderer.begin_frame(0.0)
    renderer.render(state_wp)
    # render contact points cp_wp
    # renderer.render_points('contact_points', cp_h.numpy(), radius=5e-3, colors=(1.0, 0.0, 0.0))
    renderer.render_mesh('object', points=obj_wp.points.numpy(), indices=obj_wp.indices.numpy(), colors=(0.8, 0.8, 0.8))
    renderer.end_frame()
    qp_res = []

    for iter in range(cfg.num_step):
        # reset gradients
        # cp_tmp.grad = None
        cp_tmp = torch.zeros((NC, 3), requires_grad=True)
        # ADMM step 1: primal step, update cp_o
        # initial guess
        with torch.no_grad():
            cp_tmp[:] = cp_o_cpu
        # cp_tmp to closest point
        for ii in range(cfg.step_cp):
            cp_tmp.grad = None
            _, _, tri_vert, tri_norm = get_closest_triangles(
                obj_mesh, cp_tmp.detach().numpy(), obj_vert, obj_norm, obj_face
            )
            _, cp_o, cn_o = point_to_triangle(
                cp_tmp, tri_vert, tri_norm
            )
            cn_o = - cn_o # for QP
            # foward to torch for QP solving
            res = qpsolver.solve(
                cp_o,
                cn_o,
                data['ext_wrench'],
                data['ext_center'],
            )
            print(f"Step {iter}, primal step {ii}, QP wrench error: {res.item()}")
            res.backward()
            with torch.no_grad():
                rho = 1.
                cp_tmp.grad += rho * (cp_tmp - cp_h - lam)
                alpha = 1e-1 # fixed for now
                cp_tmp -= alpha * cp_tmp.grad
                # project to object
                closest, *_ = get_closest_triangles(obj_mesh, cp_tmp.detach().numpy())
                cp_tmp.copy_(torch.from_numpy(closest))
                # grad_step1_np = cp_tmp_wp.grad.numpy()
                cp_o_cpu[:] = cp_tmp
        
        # ADMM step 2: fix cp_o_wp, optimize q
        with torch.no_grad():
            cp_tmp = cp_o_cpu - lam
        # update hand_cpn_b
        if not FIX_HAND_LOCAL_CP:
            _, tmp = project_to_hand(cp_tmp.detach().numpy(), hand_cbody, sim_env, hand_meshes, "visual")
            hand_cpn_b_np[:, :3] = tmp
        for ii in range(cfg.step_mj):
            curr_diff = sim_env.apply_contact_forces(hand_cbody, hand_cpn_b_np, cp_tmp.detach().numpy())
            sim_env.step_sim(cfg.substep)
            if ii == 0 or np.max(prev_diff - curr_diff) > 1e-4:
                prev_diff = np.copy(curr_diff)
                continue
            else:
                prev_diff = np.copy(curr_diff)
                # break
        # update to warp
        qpos_mj = sim_env.get_hand_qpos()
        qpos_wp = qpos_mj2wp(qpos_mj)
        model_wp.joint_q.assign(qpos_wp)
        state_wp.joint_q.assign(qpos_wp)
        wp.sim.eval_fk(model_wp, model_wp.joint_q, model_wp.joint_qd, None, state_wp)
        # ADMM step 3: update lam
        with torch.no_grad():
            cpn_h = sim_env.transform_cpn_b2w(hand_cbody, hand_cpn_b_np)
            cp_h.copy_(torch.from_numpy(cpn_h[:, :3]))
            lam += cp_h - cp_o_cpu
        print(f"Step {iter}, lam norm: {lam.norm().item()}")
        qp_res.append(res.item())
        
        
        renderer.begin_frame(float(iter + 1) * 0.4)
        renderer.render(state_wp)
        renderer.render_points('contact_points', cp_o_cpu.numpy(), radius=5e-3, colors=(1.0, 0.0, 0.0))
        renderer.render_mesh('object', points=obj_wp.points.numpy(), indices=obj_wp.indices.numpy(), colors=(0.8, 0.8, 0.8))
        # # visualize gradient cp_h_wp.grad
        # cp_st_vis = cp_h_wp.numpy()
        # cp_ed_vis = cp_h_wp.numpy() - grad_step1_np * 5e0
        # # s0, e0, s1, e1, ...
        # cp_vis = np.zeros((len(cp_st_vis) * 2, 3), dtype=np.float32)
        # cp_vis[0::2, :] = cp_st_vis
        # cp_vis[1::2, :] = cp_ed_vis
        # cp_idx_vis = np.arange(len(cp_st_vis) * 2, dtype=np.uint32)
        # renderer.render_line_list('contact_point_grads', cp_vis, cp_idx_vis, radius=5e-3, color=(0.0, 1.0, 0.0))
        renderer.end_frame()
    renderer.save()
    
    plt.plot(qp_res)
    plt.xlabel("iter")
    plt.ylabel("qp res")
    plt.savefig("qp_res.png")

if __name__ == "__main__":
    main()