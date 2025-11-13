from typing import Tuple, List
import os
import math
import numpy as np
import quaternion
import scipy
import logging
import warp as wp
import warp.sim
import warp.sim.render as wpr
import torch
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

import mujoco
import mujoco.viewer

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

def create_trimeshes(
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

def project(
  pts: np.ndarray, # (N, 3)
  hand_cbody: str,
  env: MuJoCo_OptQEnv,
  meshes: List[trimesh.Trimesh],
  geom_filter: str = "all"
  ):
    body_id = env._model.body(env._cfg.hand_prefix + hand_cbody).id
    b_r = env._data.xmat[body_id].reshape(3, 3)
    b_t = env._data.xpos[body_id]
    
    geom_ids = np.where(env._model.geom_bodyid == body_id)[0]
    if geom_ids.size == 0:
        raise RuntimeError(f"Body '{hand_cbody}' has no geoms")
    print(geom_ids)
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
    
    p_local = (pts - g_t) @ g_r
    
    print(f"geom_id: {g_id}, geom_type: {g_type}, geom_size: {g_size}")

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
    q_world = q_local @ g_r.T + g_t
    return q_world, None
  
@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    set_logging(cfg.verbose)
    data = load_instance(cfg)

    sim_env = init_sim_mujoco(cfg, data)
    obj_mesh, obj_vert, obj_norm, obj_face = load_obj_trimesh(data)
    model_wp, state_wp, obj_wp = init_sim_wp(cfg, data)
    hand_cpn_b, cbody_ids, hand_cbody = init_local_contact(data, sim_env, model_wp)

    # sample random points in space
    NP = 100
    pts = np.random.normal(loc=np.zeros(3), scale=8e-2, size=(NP, 3)).astype(np.float32)
    cbodies = ['ff_proximal', 'ff_tip', 'mf_proximal', 'mf_tip', 'rf_proximal', 'rf_tip', 'th_medial', 'th_tip', 'palm', 'palm', 'palm']
    meshes = create_trimeshes(sim_env)
    out_pts, _ = project(pts, cbodies[1], sim_env, meshes, geom_filter="visual")
    
    OUT_USD = "test_proj.usd"
    renderer = wpr.SimRenderer(model_wp, OUT_USD)
    renderer.begin_frame(0.0)
    renderer.render(state_wp)
    renderer.render_points("original_pts", pts, radius=1e-3, colors=(0.2,0.2,0.2))
    renderer.render_points("proj_pts", out_pts, radius=3e-3, colors=(1.,0.2,0.2))
    # renderer.render_mesh('object', points=obj_vert.numpy(), indices=obj_face.numpy(), colors=(0.8, 0.8, 0.8))
    renderer.end_frame()
    renderer.save()
    print(f"可视化文件已保存到 {OUT_USD}")
    
if __name__ == "__main__":
    main()