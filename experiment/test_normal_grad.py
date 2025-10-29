import warp as wp
import numpy as np
import trimesh

# seed
np.random.seed(0)

wp.init()

radius   = 1.8
sphere   = trimesh.creation.icosphere(subdivisions=3, radius=radius)
# apply a random rotation to the sphere to avoid singularities when compute gradient
random_rotation = trimesh.transformations.random_rotation_matrix()
sphere.apply_transform(random_rotation)
verts    = np.array(sphere.vertices, dtype=np.float32)
faces    = np.array(sphere.faces,    dtype=np.int32)
norms  = sphere.vertex_normals.astype(np.float32)
print(verts.shape, faces.shape)
verts_wp = wp.array(verts, dtype=wp.vec3)
faces_wp = wp.array(faces.flatten(), dtype=wp.int32)
norms_wp = wp.array(norms, dtype=wp.vec3)
mesh     = wp.Mesh(points=verts_wp, indices=faces_wp, velocities=norms_wp)

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
def compute_loss(
    norms: wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    n = norms[tid]
    vec = wp.normalize(wp.vec3(1., 2., 3.))
    l = wp.acos(wp.dot(n, vec))
    wp.atomic_add(loss, 0, l)
    return
    

R = 2.
points_np = np.array([[R, 0.0, 0.0], [0.0, R, 0.0], [0.0, 0.0, R]], dtype=np.float32)
points = wp.array(points_np, dtype=wp.vec3, requires_grad=True)

cl_points = wp.zeros(points.shape, dtype=wp.vec3, requires_grad=True)
out_normals = wp.zeros(points.shape, dtype=wp.vec3, requires_grad=True)
loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

tape = wp.Tape()

with tape:
    wp.launch(
        kernel=get_closest_point,
        dim=points.shape[0],
        inputs=[points, wp.array([mesh.id], dtype=wp.uint64), points.shape[0]],
        outputs=[cl_points, out_normals],
        device="cuda"
    )
    wp.launch(
        kernel=compute_loss,
        dim=points.shape[0],
        inputs=[out_normals, loss],
        device="cuda"
    )

loss.grad.fill_(1.0)
tape.backward(grads={loss: loss.grad})

print(cl_points.numpy())
print(out_normals.numpy())
grad_np = tape.gradients[points].numpy()
print(np.linalg.norm(grad_np, axis=-1))

zn = np.cross(np.array([1., 2., 3.])[None,...], out_normals.numpy())
zn = zn / np.linalg.norm(zn, axis=-1, keepdims=True)
print("###")
print(np.einsum('ij,ij->i', grad_np, zn))
print(np.einsum('ij,ij->i', grad_np, out_normals.numpy()))
