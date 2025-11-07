import numpy as np
import warp as wp
import trimesh
from utils.import_mjcf_custom import parse_mjcf   # 沿用你已有的 parse_mjcf
import warp.sim.render as wpr

# =====================用户区=========================
HAND_XML      = "/mnt/home/ruanliangwang/Dexonomy-private/assets/hand/allegro/right.xml"   # <— 换成自己的
N_RANDOM_PTS  = 3000                             # 撒点数量
OUT_USD       = "test_proj.usd"
# ====================================================

wp.init()

def build_hand_model(hand_xml):
    """返回 warp model 与对应 body 的 mesh_id（wp.uint64）"""
    builder = wp.sim.ModelBuilder()
    parse_mjcf(
        hand_xml, 
        builder,
        visual_classes=['palm_visual', 'base_visual', 'proximal_visual', 'medial_visual', 'distal_visual', 'fingertip_visual', 'thumbtip_visual', 'visual'],
        # visual_classes=['palm_collision', 'base_collision', 'proximal_collision', 'medial_collision', 'distal_collision', 'thumb_base_collision', 'thumb_proximal_collision', 'thumb_medial_collision', 'thumb_distal_collision', 'fingertip_collision', 'thumbtip_collision', 'collision'],
        # collider_classes=["plastic_collision"],
        parse_visuals_as_colliders=True,
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
    model.ground = False
    # 建立 body→mesh 映射
    mesh_ids = []
    geos = model.shape_geo.source.numpy()
    for i in range(model.body_count):
        shapes = model.body_shapes[i]
        print(shapes)
        assert len(shapes) == 1, "only 1 shape per body"
        mesh_ids.append(geos[shapes[0]])
    mesh_ids = wp.array(mesh_ids, dtype=wp.uint64)
    return model, mesh_ids

def random_points_around_body(model, body_id, n_pts, radius=0.05):
    """在 body 质心附近随机撒点"""
    com = model.body_com.numpy()[body_id]
    pts = np.random.normal(loc=com, scale=radius, size=(n_pts, 3)).astype(np.float32)
    return pts

@wp.kernel
def update_cpn_b_kernel(cp_w: wp.array(dtype=wp.vec3),
                        cp_b: wp.array(dtype=wp.vec3),
                        body_q: wp.array(dtype=wp.transform),
                        body_id: wp.int32,
                        mesh_id: wp.uint64):
    tid = wp.tid()
    # 世界系→局部系
    cp_local = wp.transform_point(wp.transform_inverse(body_q[body_id]), cp_w[tid])
    # z_hack = 2.8e-2
    z_hack = 4.4e-2
    # cp_local.z -= z_hack
    # 投影到局部 mesh
    max_dist = 100.0
    res = wp.mesh_query_point(mesh_id, cp_local, max_dist)
    if res.result:
        f = res.face
        v0 = wp.mesh_get_point(mesh_id, f*3+0)
        v1 = wp.mesh_get_point(mesh_id, f*3+1)
        v2 = wp.mesh_get_point(mesh_id, f*3+2)
        u,v = res.u, res.v
        w = 1.0-u-v
        proj = u*v0 + v*v1 + w*v2
        proj.z += z_hack
        cp_b[tid] = proj
    else:
        cp_b[tid] = cp_local   # 失败则保持原局部坐标



def main():
    model, mesh_ids = build_hand_model(HAND_XML)
    state = model.state()

    # 找到目标 body 的索引
    body_name_list = [model.body_name[i] for i in range(model.body_count)]
    
    body_id = 20
    mesh_id = mesh_ids.numpy()[body_id]

    # 随机撒点（世界系）
    pts_w_np = random_points_around_body(model, body_id, N_RANDOM_PTS, radius=0.08)
    pts_w = wp.array(pts_w_np, dtype=wp.vec3)

    # 分配局部坐标 buffer
    pts_b = wp.empty_like(pts_w)

    # 先把 hand 摆一个固定姿势（这里用初始 qpos）
    qpos = np.zeros(model.joint_coord_count)
    qpos[3] = 1.0   # 单位四元数
    model.joint_q.assign(qpos)
    state.joint_q.assign(qpos)
    wp.sim.eval_fk(model, model.joint_q, model.joint_qd, None, state)

    # 运行投影
    wp.launch(update_cpn_b_kernel,
              dim=N_RANDOM_PTS,
              inputs=[pts_w, pts_b, state.body_q, body_id, mesh_id])

    # 再把局部点变回世界系用于可视化
    @wp.kernel
    def local2world(cp_b: wp.array(dtype=wp.vec3),
                    cp_w_new: wp.array(dtype=wp.vec3),
                    body_q: wp.array(dtype=wp.transform),
                    body_id: wp.int32):
        tid = wp.tid()
        cp_w_new[tid] = wp.transform_point(body_q[body_id], cp_b[tid])

    pts_w_new = wp.empty_like(pts_w)
    wp.launch(local2world, dim=N_RANDOM_PTS,
              inputs=[pts_b, pts_w_new, state.body_q, body_id])

    # 可视化
    renderer = wpr.SimRenderer(model, OUT_USD)
    renderer.begin_frame(0.0)
    renderer.render(state)
    # # 原始点（红色）
    # renderer.render_points("original_pts", pts_w.numpy(), radius=3e-3, colors=(1,0.2,0.2))
    # 投影后点（绿色）
    renderer.render_points("proj_pts", pts_w_new.numpy(), radius=3e-3, colors=(0.2,1,0.2))
    renderer.end_frame()
    renderer.save()
    print(f"可视化文件已保存到 {OUT_USD}")

if __name__ == "__main__":
    main()