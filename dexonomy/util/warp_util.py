import warp as wp
import torch
import trimesh
import numpy as np

from dexonomy.util.torch_util import (
    torch_inv_transform_points,
    torch_quaternion_to_matrix,
)


def init_warp(quiet=True):
    wp.config.quiet = quiet
    wp.init()
    return True


init_warp()


@wp.kernel
def get_mesh_lineseg_coll(
    line_seg: wp.array(dtype=wp.vec3),
    mesh_idx: wp.array(dtype=wp.uint64),
    num_lines: wp.int32,
    output_valid_idx: wp.array(dtype=bool),
):
    tid = wp.tid()
    if output_valid_idx[tid] == False:
        return

    m_id = mesh_idx[tid]
    for line_id in range(num_lines):
        start = line_seg[tid * num_lines * 2 + line_id * 2]
        end = line_seg[tid * num_lines * 2 + line_id * 2 + 1]
        delta = end - start
        max_t = wp.length(delta)
        dir = wp.normalize(delta)
        query_result = wp.mesh_query_ray(m_id, start, dir, max_t)
        if query_result.result:
            output_valid_idx[tid] = False
            break
    return


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
        cl_pt = wp.mesh_eval_position(
            mesh_idx[bid], collide_result.face, collide_result.u, collide_result.v
        )
        # Use the 'velocity' attribute to save vertex normal
        vn1 = wp.mesh_get_velocity(mesh_idx[bid], collide_result.face * 3 + 0)
        vn2 = wp.mesh_get_velocity(mesh_idx[bid], collide_result.face * 3 + 1)
        vn3 = wp.mesh_get_velocity(mesh_idx[bid], collide_result.face * 3 + 2)
        normal = wp.normalize(
            collide_result.u * vn1
            + collide_result.v * vn2
            + (1.0 - collide_result.u - collide_result.v) * vn3
        )
        # normal = wp.mesh_eval_face_normal(mesh_idx[bid], collide_result.face)
        out_points[tid] = cl_pt
        out_normals[tid] = -normal
    return


class MeshQueryPoint(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        points: torch.Tensor,
        out_points: torch.Tensor,
        out_normals: torch.Tensor,
        adj_points: torch.Tensor,
        mesh_idx: torch.Tensor,
    ):
        out_points[:] = 0
        out_normals[:] = 0

        b, h, n, _ = points.shape
        ctx.b, ctx.h, ctx.n, _ = points.shape
        ctx.save_for_backward(
            points,
            out_points,
            out_normals,
            adj_points,
            mesh_idx,
        )

        wp.launch(
            kernel=get_closest_point,
            dim=b * h * n,
            inputs=[
                wp.from_torch(points.detach().view(-1, 3), dtype=wp.vec3),
                wp.from_torch(mesh_idx, dtype=wp.uint64),
                h * n,
            ],
            outputs=[
                wp.from_torch(out_points.view(-1, 3), dtype=wp.vec3),
                wp.from_torch(out_normals.view(-1, 3), dtype=wp.vec3),
            ],
            stream=wp.stream_from_torch(points.device),
        )

        return out_points, out_normals

    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        (
            points,
            out_points,
            out_normals,
            adj_points,
            mesh_idx,
        ) = ctx.saved_tensors
        adj_points[:] = 0

        wp_adj_out_points = wp.from_torch(
            grad_output1.view(-1, 3).contiguous(), dtype=wp.vec3
        )
        wp_adj_out_normals = wp.from_torch(
            grad_output2.view(-1, 3).contiguous(), dtype=wp.vec3
        )
        wp_adj_points = wp.from_torch(adj_points.view(-1, 3), dtype=wp.vec3)
        wp.launch(
            kernel=get_closest_point,
            dim=ctx.b * ctx.h * ctx.n,
            inputs=[
                wp.from_torch(
                    points.detach().view(-1, 3), dtype=wp.vec3, grad=wp_adj_points
                ),
                wp.from_torch(mesh_idx, dtype=wp.uint64),
                ctx.h * ctx.n,
            ],
            outputs=[
                wp.from_torch(
                    out_points.view(-1, 3), dtype=wp.vec3, grad=wp_adj_out_points
                ),
                wp.from_torch(
                    out_normals.view(-1, 3), dtype=wp.vec3, grad=wp_adj_out_normals
                ),
            ],
            adj_inputs=[
                None,
                None,
                ctx.h * ctx.n,
            ],
            adj_outputs=[
                None,
                None,
            ],
            stream=wp.stream_from_torch(points.device),
            adjoint=True,
        )

        g_p = None
        if ctx.needs_input_grad[0]:
            g_p = adj_points
        return g_p, None, None, None, None


class WarpCollisionEnv:
    def __init__(self, device: torch.device):
        self.device = device
        self._wp_dev = wp.torch.device_from_torch(device)
        self._wp_cache = {}  # Mesh cache
        self._col_plane = None
        self._col_mesh = torch.ones(0, dtype=torch.int64, device=device)
        self._out_points = torch.zeros(0, dtype=torch.float32, device=device)
        self._out_normals = torch.zeros(0, dtype=torch.float32, device=device)
        self._adj_points = torch.zeros(0, dtype=torch.float32, device=device)
        return

    def load(
        self,
        col_plane: torch.Tensor | None,
        col_mesh: list[tuple[str, trimesh.Trimesh]],
    ):
        mesh_id = []
        for cm in col_mesh:
            obj_name, tm_obj = cm[0], cm[1]
            if obj_name not in self._wp_cache:
                v = wp.array(tm_obj.vertices, dtype=wp.vec3, device=self._wp_dev)
                f = wp.array(np.ravel(tm_obj.faces), dtype=int, device=self._wp_dev)
                vn = wp.array(tm_obj.vertex_normals, dtype=wp.vec3, device=self._wp_dev)
                self._wp_cache[obj_name] = wp.Mesh(points=v, indices=f, velocities=vn)
            mesh_id.append(self._wp_cache[obj_name].id)
        self._col_mesh = torch.tensor(mesh_id, dtype=torch.int64, device=self.device)
        self._col_plane = col_plane
        return

    def query_point(self, points: torch.Tensor):
        if self._out_points.shape != points.shape:
            self._out_points = torch.zeros(points.shape, device=self.device)
            self._out_normals = torch.zeros(points.shape, device=self.device)
            self._adj_points = torch.zeros(points.shape, device=self.device)

        obj_cp_o, obj_cn_o = MeshQueryPoint.apply(
            points.contiguous(),
            self._out_points,
            self._out_normals,
            self._adj_points,
            self._col_mesh,
        )
        return obj_cp_o, obj_cn_o

    def check_mesh(self, line_seg: torch.Tensor, parallel_id: torch.Tensor):
        b, n_lines = line_seg.shape[0], line_seg.shape[1] // 2
        out_valid_idx = torch.ones(b, dtype=torch.bool, device=self.device)
        wp.launch(
            kernel=get_mesh_lineseg_coll,
            dim=b,
            inputs=[
                wp.from_torch(line_seg.view(-1, 3), dtype=wp.vec3),
                wp.from_torch(self._col_mesh[parallel_id], dtype=wp.uint64),
                n_lines,
            ],
            outputs=[
                wp.from_torch(out_valid_idx, dtype=wp.bool),
            ],
            stream=wp.stream_from_torch(line_seg.device),
        )
        return out_valid_idx

    def check_plane(
        self, line_seg: torch.Tensor, parallel_id: torch.Tensor, plane_thre: float
    ):
        out_valid_idx = torch.ones(
            line_seg.shape[0], dtype=torch.bool, device=self.device
        )
        if self._col_plane is not None:
            col_plane = self._col_plane[parallel_id]
            pf_hsk = torch_inv_transform_points(
                line_seg,
                torch_quaternion_to_matrix(col_plane[..., 3:]),
                col_plane[..., :3].view(-1, 1, 3),
            )
            out_valid_idx = (pf_hsk[..., -1] > plane_thre).min(dim=-1)[0]
        return out_valid_idx
