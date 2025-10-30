import os
import math
import numpy as np
import scipy
import logging
import warp as wp
import warp.sim
import warp.sim.render
import torch
import trimesh
from qpsolvers import solve_qp
from scipy.spatial.transform import Rotation as R
from dexonomy.util.file_util import load_scene_cfg
from dexonomy.util.np_util import np_normalize_vector, np_normal_to_rot

from utils.vis_util import points_to_obj, arrows_to_obj
from sim_mujoco import MuJoCo_OptQEnv, MuJoCo_OptQCfg, HandCfg
from utils.import_mjcf_custom import parse_mjcf

import hydra
from omegaconf import DictConfig
from utils.log_util import set_logging
from loader import load_instance

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

def init_sim_warp(cfg: dict, grasp_data: dict):
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
    # move quaternion's w to the back
    w = qpos[3]
    qpos[3:6] = qpos[4:7]
    qpos[6] = w
    # don't know why, but this works
    r = R.from_quat(qpos[3:7])
    r1 = R.from_rotvec(-np.array([0, 1, 0]) * np.pi / 2)
    r2 = R.from_rotvec(np.array([0, 0, 1]) * np.pi)
    qpos[3:7] = (r * r2 * r1).as_quat()
    
    model.joint_q.assign(qpos)
    state.joint_q.assign(qpos)

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

@wp.func
def project_to_plane(p: wp.vec3, c: wp.vec3, n: wp.vec3) -> wp.vec3:
    """Project point p onto the plane with point c and normal n."""
    return p - wp.dot(p - c, n) * n

@wp.kernel
def get_closest_point(
    points: wp.array(dtype=wp.vec3),
    mesh_idx: wp.array(dtype=wp.uint64),
    single_batch_num: wp.int32,
    out_points: wp.array(dtype=wp.vec3),
    out_normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    bid = tid // single_batch_num

    point = points[tid]
    max_distance = 100.0
    collide_result = wp.mesh_query_point(mesh_idx[bid], point, max_distance)

    if collide_result.result:
        face_index = collide_result.face
        
        v0 = wp.mesh_get_point(mesh_idx[bid], face_index * 3 + 0)
        v1 = wp.mesh_get_point(mesh_idx[bid], face_index * 3 + 1)
        v2 = wp.mesh_get_point(mesh_idx[bid], face_index * 3 + 2)
        # Use the 'velocity' attribute to save vertex normal
        vn0 = wp.mesh_get_velocity(mesh_idx[bid], face_index * 3 + 0)
        vn1 = wp.mesh_get_velocity(mesh_idx[bid], face_index * 3 + 1)
        vn2 = wp.mesh_get_velocity(mesh_idx[bid], face_index * 3 + 2)
        u, v, w = collide_result.u, collide_result.v, 1.0 - collide_result.u - collide_result.v
        
        p = u * v0 + v * v1 + w * v2
        c0 = project_to_plane(p, v0, vn0)
        c1 = project_to_plane(p, v1, vn1)
        c2 = project_to_plane(p, v2, vn2)
        q = u * c0 + v * c1 + w * c2
        
        alpha = 0.75
        cl_pt = (1.0 - alpha) * p + alpha * q
        
        norm_intp = wp.normalize(u * vn0 + v * vn1 + w * vn2)
        sign = 1.
        if wp.dot(norm_intp, point - cl_pt) < 0.:
            sign = -1.
        norm = sign * (point - cl_pt)

        out_points[tid] = cl_pt
        # out_normals[tid] = - norm # negative for QP
        out_normals[tid] = - norm_intp
        # out_normals[tid] = - wp.normalize(point)
    return

@wp.kernel
def compute_cp_warp(
    hand_cpn_b: wp.array(dtype=float, ndim=2),
    cbody_ids: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    cp_warp: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    cp_b = wp.vec3(hand_cpn_b[tid, 0], hand_cpn_b[tid, 1], hand_cpn_b[tid, 2])
    id = cbody_ids[tid]
    cp_warp[tid] = wp.transform_point(body_q[id], cp_b)

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
                       model_warp: wp.sim.Model
                       ):
    hand_cbody, obj_cpn_w = data["hand_cbody"], data["obj_cpn_w"]
    hand_cpn_b_np = env_mujoco.transform_cpn_w2b(hand_cbody, data["hand_cpn_w"])
    hand_cpn_b = wp.array(hand_cpn_b_np, dtype=float)
    
    cbody_ids = []
    body_name_warp = model_warp.body_name
    for cb in hand_cbody:
        # cid = sim_env._model.body(sim_env._cfg.hand_prefix + cb).id
        cid = body_name_warp.index(cb)
        cbody_ids.append(cid)
    cbody_ids = wp.array(cbody_ids, dtype=wp.int32)
    
    return hand_cpn_b, cbody_ids
  
def build_tape_graph():
    tape = wp.Tape()
    wp.capture_begin()
    with tape:
        pass

@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    set_logging(cfg.verbose)
    data = load_instance(cfg)

    sim_env = init_sim_mujoco(cfg, data)
    model_warp, state_warp, obj_warp = init_sim_warp(cfg, data)
    hand_cpn_b, cbody_ids = init_local_contact(data, sim_env, model_warp)
    
    cp_h_warp = wp.array(dtype=wp.vec3, shape=len(hand_cpn_b), requires_grad=True) # contact points on hand
    cp_o_warp = wp.array(dtype=wp.vec3, shape=len(hand_cpn_b), requires_grad=True) # contact points on object
    cn_o_warp = wp.array(dtype=wp.vec3, shape=len(hand_cpn_b), requires_grad=True) # contact normals in world
    cp_o_torch = wp.to_torch(cp_o_warp)
    cn_o_torch = wp.to_torch(cn_o_warp)
    
    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        wp.launch(compute_cp_warp, 
                  dim=len(cbody_ids), 
                  inputs=(hand_cpn_b, cbody_ids, state_warp.body_q, cp_h_warp)
                  )
        print("contact points from mujoco:")
        print(data["hand_cpn_w"][:, :3])
        print("contact points from warp:")
        print(cp_h_warp)
    
    qpsolver = QPSingleSolver(cfg.miu_coeff)

    # single iteration begin
    # rest state if any
    #
    tape = wp.Tape()
    with tape:
    # forward kinematics
        wp.sim.eval_fk(model_warp, model_warp.joint_q, model_warp.joint_qd, None, state_warp)
        # compute contact points in warp
        wp.launch(compute_cp_warp, 
                  dim=len(cbody_ids), 
                  inputs=(hand_cpn_b, cbody_ids, state_warp.body_q, cp_h_warp),
                  )
        # compute closest points on object mesh
        wp.launch(get_closest_point,
                  dim=len(cp_h_warp),
                  inputs=(cp_h_warp, wp.array([obj_warp.id], dtype=wp.uint64), 
                          len(cp_h_warp), cp_o_warp, cn_o_warp),
                  )
    # foward to torch for QP solving
    res = qpsolver.solve(
        cp_o_torch,
        cn_o_torch,
        data['ext_wrench'],
        data['ext_center'],
    )
    print(res)
    res.backward()
    tape.backward(grads={cp_o_warp: cp_o_warp.grad, cn_o_warp: cn_o_warp.grad})
    print(f"grad_q: {model_warp.joint_q.grad}")
    # single iteration end
    
    
    renderer = wp.sim.render.SimRenderer(model_warp, "debug_warp.usd")
    # draw hand
    renderer.begin_frame(0.0)
    renderer.render(state_warp)
    # render contact points cp_warp
    renderer.render_points('contact_points', cp_h_warp.numpy(), radius=0.01, colors=(1.0, 0.0, 0.0))
    renderer.render_mesh('object', points=obj_warp.points.numpy(), indices=obj_warp.indices.numpy(), colors=(0.8, 0.8, 0.8))
    renderer.end_frame()
    renderer.save()   

if __name__ == "__main__":
    main()