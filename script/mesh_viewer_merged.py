import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------- 1. Create an empty 3-D axis ----------
def get_scene_axis(mesh, elev=30, azim=-60, figsize=(8, 6), dpi=150):
    """
    Build a 3-D axis with proper aspect, limits and camera angle
    based on *mesh* bounding box.  No geometry is added yet.
    """
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


# ---------- 2. Render a single merged mesh ----------
def render_merged(ax, merged_mesh,
                  facecolor=(0.7, 0.7, 0.7, 1.0),
                  edgecolor=(0.3, 0.3, 0.3, 0.3),
                  linewidths=0.3):
    """
    Add the already-merged mesh to *ax* as one Poly3DCollection.
    """
    poly = Poly3DCollection(merged_mesh.vertices[merged_mesh.faces],
                            facecolor=facecolor,
                            edgecolor=edgecolor,
                            linewidths=linewidths)
    ax.add_collection3d(poly)


# ---------- 3. Merge meshes ----------
def merge_meshes(path_list):
    """
    Load all existing paths and return a single Trimesh object.
    """
    path_list = [p for p in path_list if os.path.isfile(p)]
    if not path_list:
        raise RuntimeError("No valid input files found.")
    meshes = [trimesh.load(p, force='mesh') for p in path_list]
    return trimesh.util.concatenate(meshes)


# ---------- 4. Multi-view pipeline ----------
def render_views(obj_paths, output_prefix, views=None):
    if views is None:
        views = [
            {'elev': 30, 'azim': -60, 'name': 'front'},
            {'elev': 89.9, 'azim': -90, 'name': 'top'},
            {'elev': 0,  'azim': -90, 'name': 'side'}
        ]

    merged = merge_meshes(obj_paths)

    plt.ioff()
    for view in views:
        fig, ax = get_scene_axis(merged,
                                 elev=view['elev'],
                                 azim=view['azim'])
        render_merged(ax, merged)
        out_name = f"{output_prefix}_{view['name']}.png"
        fig.savefig(out_name,
                    bbox_inches='tight',
                    pad_inches=0.02,
                    transparent=True)
        plt.close(fig)
        print(f"saved {out_name}")


# ---------- 5. CLI ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_obj", nargs='+',
                        help="Input OBJ file path(s); multiple files supported")
    parser.add_argument("-o", "--output", default="output",
                        help="Output filename prefix")
    args = parser.parse_args()

    # Auto pair _hand.obj / _obj.obj if only one file given
    input_files = args.input_obj
    if len(input_files) == 1:
        file = input_files[0]
        if file.endswith("_hand.obj"):
            obj_file = file.replace("_hand.obj", "_obj.obj")
            input_files = [file, obj_file]
        elif file.endswith("_obj.obj"):
            hand_file = file.replace("_obj.obj", "_hand.obj")
            input_files = [hand_file, file]

    render_views(input_files, args.output)