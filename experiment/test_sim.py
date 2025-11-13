import warp as wp
import warp.sim
import warp.sim.render
import numpy as np
import os
from utils.import_mjcf_custom import parse_mjcf

wp.init()

# ---------------------- 参数 ---------------------- #
XML          =  "/mnt/home/ruanliangwang/Dexonomy-private/assets/hand/allegro/right.xml"          # 任意 MJCF 手
SIM_DT       = 1.0/2000            # 2 kHz
RENDER_FPS   = 50
TOTAL_SEC    = 3.0
GRAVITY      = wp.vec3(0, -9.81, 0)
# ------------------------------------------------ #

def build_fixed_hand(xml_path):
    builder = wp.sim.ModelBuilder()
    parse_mjcf(
        xml_path,
        builder,
        visual_classes=['palm_visual', 'base_visual', 'proximal_visual', 'medial_visual', 'distal_visual', 'fingertip_visual', 'thumbtip_visual', 'visual'],
        # collider_classes=["plastic_collision"],
        floating=False,
        density=1e5,  # NOTE: If density==1e6, the simluation will be unstable
        armature=0.01,
        stiffness=0.,  # NOTE: If stiffness>=10, the simluation will be unstable
        damping=1,
        contact_ke=1,
        contact_kd=1,
        contact_kf=1,
        contact_mu=1.0,
        contact_restitution=0.0,
        # up_axis="Y",
        verbose=True
    )
    # 关键：打开自碰撞
    builder.enable_self_collisions = True
    builder.ground = False
    model = builder.finalize(requires_grad=False)
    model.gravity = GRAVITY
    model.ground = False
    
    
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

    return model

def main():
    model = build_fixed_hand(XML)
    state0 = model.state()
    state1 = model.state()

    # 给手指一点初始弯曲，看起来更明显
    q = state0.joint_q.numpy().copy()
    # q[7:] = 0.3           # 大致弯曲
    state0.joint_q.assign(q)
    state0.joint_qd.zero_()

    integrator = wp.sim.FeatherstoneIntegrator(model)
    renderer = wp.sim.render.SimRenderer(model, "hand_self_gravity.usd")

    n_frames = int(TOTAL_SEC * RENDER_FPS)
    dt = SIM_DT
    render_every = int(1.0 / (RENDER_FPS * dt))

    for i in range(n_frames * render_every):
        wp.sim.eval_fk(model, state0.joint_q, state0.joint_qd, None, state0)
        integrator.simulate(model, state0, state1, dt)
        state0, state1 = state1, state0   # swap

        if i % render_every == 0:
            frame = i // render_every
            renderer.begin_frame(frame * 1.0 / RENDER_FPS)
            renderer.render(state0)
            renderer.end_frame()
            print(f'frame {frame}/{n_frames}')

    renderer.save()
    print('USD 已保存为 hand_self_gravity.usd')

if __name__ == '__main__':
    main()