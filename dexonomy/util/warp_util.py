import warp as wp
import torch


def init_warp(quiet=True):
    wp.config.quiet = quiet
    wp.init()
    return True


init_warp()


@wp.kernel
def get_mesh_lineseg_coll(
    line_segments: wp.array(dtype=wp.vec3),
    mesh_idx: wp.array(dtype=wp.uint64),
    num_lines: wp.int32,
    output_valid_idx: wp.array(dtype=bool),
):
    tid = wp.tid()
    if output_valid_idx[tid] == False:
        return

    m_id = mesh_idx[tid]
    for line_id in range(num_lines):
        start = line_segments[tid * num_lines * 2 + line_id * 2]
        end = line_segments[tid * num_lines * 2 + line_id * 2 + 1]
        delta = end - start
        max_t = wp.length(delta)
        dir = wp.normalize(delta)
        query_result = wp.mesh_query_ray(m_id, start, dir, max_t)
        if query_result.result:
            output_valid_idx[tid] = False
            break
    return


def MeshLineSegCollision(
    line_segments,
    mesh_idx,
    output_valid_idx,
):
    b = line_segments.shape[0]
    num_lines = line_segments.shape[1] // 2
    wp.launch(
        kernel=get_mesh_lineseg_coll,
        dim=b,
        inputs=[
            wp.from_torch(line_segments.view(-1, 3), dtype=wp.vec3),
            wp.from_torch(mesh_idx, dtype=wp.uint64),
            num_lines,
        ],
        outputs=[
            wp.from_torch(output_valid_idx, dtype=wp.bool),
        ],
        stream=wp.stream_from_torch(line_segments.device),
    )
    return output_valid_idx


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
