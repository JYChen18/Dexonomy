import numpy as np
import trimesh
import logging

# input a (n, 3) array, and a radius
# output an obj file, with spheres at each point
def points_to_obj(points, radius, obj_file):
    # one single sphere mesh
    logging.debug(f"points shape: {points.shape}")
    sph = trimesh.creation.icosphere(radius=radius)
    vertices = sph.vertices[None, :, :] + points[:, None, :]   # broadcasting
    faces = sph.faces[None, :, :] + (np.arange(len(points)) * len(sph.vertices))[:, None, None]
    mesh = trimesh.Trimesh(vertices=vertices.reshape(-1, 3),
                           faces=faces.reshape(-1, 3))
    # save to obj
    mesh.export(obj_file)

def arrows_to_obj(positions, directions, radius, length=None, obj_file='arrows.obj'):
    n = len(positions)
    if n == 0:
        return

    dirs = np.asarray(directions, dtype=float)

    if length is None:
        disp_vecs = dirs
        target_lens = np.linalg.norm(dirs, axis=1)
    else:
        dirs_unit = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        length = np.asarray(length, dtype=float)
        if length.ndim == 0:
            disp_vecs = dirs_unit * length
            target_lens = np.full(n, length)
        else:
            disp_vecs = dirs_unit * length.reshape(-1, 1)
            target_lens = length

    shaft_len = 0.75
    head_len  = 0.25
    total_locallen = shaft_len + head_len

    shaft = trimesh.creation.cylinder(radius=radius, height=shaft_len, sections=16)
    head  = trimesh.creation.cone(radius=radius*1.6, height=head_len, sections=16)
    shaft.apply_translation([0, 0, shaft_len/2])
    head.apply_translation([0, 0, shaft_len])
    arrow = trimesh.util.concatenate(shaft, head)

    scales = target_lens / total_locallen
    vertices = arrow.vertices.copy()
    vertices = vertices[None, :, :].repeat(n, axis=0)
    vertices[:, :, 2] *= scales[:, None]

    z_axis = np.array([0, 0, 1], dtype=float)
    R = rotation_matrix_batch(z_axis, disp_vecs)

    v0 = vertices
    v_rot = np.einsum('nij,nvj->nvi', R, v0)

    v_final = v_rot + positions[:, None, :]

    n_v_arrow = v0.shape[1]
    faces = arrow.faces[None, :, :] + (np.arange(n) * n_v_arrow)[:, None, None]
    mesh = trimesh.Trimesh(vertices=v_final.reshape(-1, 3),
                           faces=faces.reshape(-1, 3))
    mesh.export(obj_file)
    logging.debug(f"exported {n} arrows to {obj_file}")

def rotation_matrix_batch(from_vec, to_vec):
    from_vec = from_vec / np.linalg.norm(from_vec)
    to_vec   = to_vec / (np.linalg.norm(to_vec, axis=1, keepdims=True) + 1e-12)
    cross = np.cross(from_vec, to_vec)          # (n,3)
    dot   = np.einsum('nj,j->n', to_vec, from_vec)
    eps = 1e-8
    flip = dot < -1 + eps
    rot = np.empty((len(to_vec), 3, 3))

    norm_cross = np.linalg.norm(cross, axis=1)
    mask = ~flip & (norm_cross > eps)
    k = cross[mask] / norm_cross[mask, None]
    theta = np.arccos(dot[mask])
    K = skew_symmetric(k)
    sin_, cos_ = np.sin(theta), np.cos(theta)
    rot[mask] = np.eye(3) + sin_[:,None,None]*K + (1-cos_)[:,None,None]*K@K

    if np.any(flip):
        axis = np.array([1,0,0]) if abs(from_vec[0])<0.9 else np.array([0,1,0])
        axis = np.cross(from_vec, axis)
        axis = axis / np.linalg.norm(axis)
        K = skew_symmetric(axis[None])
        rot[flip] = np.eye(3) + 2*K@K
    rot[~mask & ~flip] = np.eye(3)
    return rot

def skew_symmetric(v):
    n = v.shape[0]
    S = np.zeros((n, 3, 3))
    S[:, 0, 1] = -v[:, 2]; S[:, 0, 2] =  v[:, 1]
    S[:, 1, 0] =  v[:, 2]; S[:, 1, 2] = -v[:, 0]
    S[:, 2, 0] = -v[:, 1]; S[:, 2, 1] =  v[:, 0]
    return S