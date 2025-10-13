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
    If multiple meshes are to be rendered, use the largest one for
    setting up the scene.
    """
    if isinstance(mesh, list):
        # if multiple meshes, use the largest one to set up the scene
        areas = [m.bounding_box.volume for m in mesh]
        mesh = mesh[np.argmax(areas)]
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


# ---------- 2. Render a single mesh into an axis ----------
def render_mesh(ax, mesh,
                facecolor=(0.7, 0.7, 0.7, 1.0),
                edgecolor=(0.3, 0.3, 0.3, 0.3),
                linewidths=0.3,
                **kwargs):
    """
    Add *mesh* to *ax* as a Poly3DCollection.
    Call repeatedly to overlay multiple meshes.
    """
    vertices = mesh.vertices
    faces = mesh.faces
    poly = Poly3DCollection(vertices[faces],
                            facecolor=facecolor,
                            edgecolor=edgecolor,
                            linewidths=linewidths,
                            **kwargs)
    ax.add_collection3d(poly)


# ---------- 3. Multi-view / multi-mesh pipeline ----------
def render_views(obj_paths, output_prefix, views=None):
    if views is None:
        views = [
            {'elev': 30, 'azim': -60, 'name': 'front'},
            {'elev': 89.9, 'azim': -90, 'name': 'top'},
            {'elev': 0, 'azim': -90, 'name': 'side'}
        ]

    if isinstance(obj_paths, str):
        obj_paths = [obj_paths]

    # skip mesh if not exists
    obj_paths = [p for p in obj_paths if os.path.exists(p)]
    meshes = [trimesh.load(p, force='mesh') for p in obj_paths]

    plt.ioff()
    for view in views:
        fig, ax = get_scene_axis(meshes,
                                 elev=view['elev'],
                                 azim=view['azim'])

        for mesh in meshes:
            render_mesh(ax, mesh,
                        facecolor=(0.7, 0.7, 0.7, 1.0),
                        edgecolor=(0.3, 0.3, 0.3, 0.3))

        out_name = f"{output_prefix}_{view['name']}.png"
        fig.savefig(out_name,
                    bbox_inches='tight',
                    pad_inches=0.02,
                    transparent=True)
        plt.close(fig)
        print(f"saved {out_name}")


# ---------- 4. Command-line interface ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_obj", nargs='+',
                        help="Input OBJ file path(s); multiple files supported")
    parser.add_argument("-o", "--output", default="output",
                        help="Output filename prefix")
    args = parser.parse_args()
    
    input_files = args.input_obj
    if len(input_files) == 1: # only one input file
        file = input_files[0]
        if file.endswith("_hand.obj"):
            obj_file = "_".join(file.split(".")[-2].split("_")[:-1]) + "_obj" + ".obj"
            input_files = [file, obj_file]
        elif file.endswith("_obj.obj"):
            hand_file = "_".join(file.split(".")[-2].split("_")[:-1]) + "_hand" + ".obj"
            input_files = [hand_file, file]

    render_views(input_files, args.output)