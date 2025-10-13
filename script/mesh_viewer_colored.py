import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------- 1. 建轴 ----------
def get_scene_axis(mesh, elev=30, azim=-60, figsize=(8, 6), dpi=150):
    bbox = mesh.bounding_box
    center = bbox.centroid
    max_extent = max(bbox.extents) / 2.0
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(center[0] - max_extent, center[0] + max_extent)
    ax.set_ylim(center[1] - max_extent, center[1] + max_extent)
    ax.set_zlim(center[2] - max_extent, center[2] + max_extent)
    ax.view_init(elev=elev, azim=azim)
    return fig, ax

# ---------- 2. 真正“合并 + 按面染色” ----------
def render_merged(ax, meshes,
                  edgecolor=(0.25, 0.25, 0.25, 0.25),
                  linewidths=0.25):
    """
    meshes : list[trimesh.Trimesh]
    先 concatenate 成整体，但给每个面记“来源索引”，
    然后一次性画，facecolor 按面给，z-buffer 正确。
    """
    n = len(meshes)
    colors = sns.color_palette("husl", n)          # RGB
    # 1) 记录每个 mesh 的面数
    face_counts = np.array([m.faces.shape[0] for m in meshes])
    # 2) 合并
    merged = trimesh.util.concatenate(meshes)
    # 3) 建立“面 → 来源文件索引”
    face_src = np.repeat(np.arange(n), face_counts)   # (N_face,)
    # 4) 按面查颜色表
    face_colors = np.asarray(colors)[face_src]        # (N_face, 3)
    # 5) 稍微降一点饱和度/亮度，并加透明度，重叠更柔和
    face_colors = 0.85 * face_colors + 0.15
    face_colors = np.clip(face_colors, 0, 1)
    face_colors = np.column_stack([face_colors, np.full(face_colors.shape[0], 0.90)])  # RGBA
    # 6) 一次性画
    poly = Poly3DCollection(merged.vertices[merged.faces],
                            facecolor=face_colors,
                            edgecolor=edgecolor,
                            linewidths=linewidths)
    ax.add_collection3d(poly)

# ---------- 3. 加载 ----------
def merge_meshes(path_list):
    path_list = [p for p in path_list if os.path.isfile(p)]
    if not path_list:
        raise RuntimeError("No valid input files found.")
    return [trimesh.load(p, force='mesh') for p in path_list]

# ---------- 4. 多视角 ----------
def render_views(obj_paths, output_prefix, views=None):
    if views is None:
        views = [
            {'elev': 30, 'azim': -60, 'name': 'front'},
            {'elev': 89.9, 'azim': -90, 'name': 'top'},
            {'elev': 0, 'azim': -90, 'name': 'side'}
        ]
    meshes = merge_meshes(obj_paths)
    combined = trimesh.util.concatenate(meshes)   # 仅用来算 bbox/轴范围
    plt.ioff()
    for view in views:
        fig, ax = get_scene_axis(combined, elev=view['elev'], azim=view['azim'])
        render_merged(ax, meshes)                 # 关键：传列表
        out_name = f"{output_prefix}_{view['name']}.png"
        fig.savefig(out_name, bbox_inches='tight', pad_inches=0.02, transparent=True)
        plt.close(fig)
        print(f"saved {out_name}")

# ---------- 5. CLI ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_obj", nargs='+', help="Input OBJ file path(s)")
    parser.add_argument("-o", "--output", default="output", help="Output filename prefix")
    args = parser.parse_args()

    input_files = args.input_obj
    if len(input_files) == 1:
        f = input_files[0]
        if f.endswith("_hand.obj"):
            input_files = [f, f.replace("_hand.obj", "_obj.obj")]
        elif f.endswith("_obj.obj"):
            input_files = [f.replace("_obj.obj", "_hand.obj"), f]

    render_views(input_files, args.output)