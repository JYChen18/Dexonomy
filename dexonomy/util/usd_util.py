# Standard Library
from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Union

# Third Party
import numpy as np
import trimesh

try:
    # Third Party
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    raise ImportError(
        "usd-core failed to import, install with pip install usd-core"
        + " NOTE: Do not install this if using with ISAAC SIM."
    )

from dexonomy.util.np_util import np_even_sample_points_on_sphere


def create_view_matrix(position, target):
    # position: the position of the camera
    # target: the position of the object
    # return: the view matrix in opengl format
    front = np.array(target) - np.array(position)
    front = front / np.linalg.norm(front)
    up = np.array([0, 0, 1.0])
    if np.linalg.norm(np.cross(front, up)) < 0.1:
        up = np.array([0, 1.0, 0.0])
    # while 1:
    #     up = np.random.rand(3)
    #     up /= np.linalg.norm(up)
    #     if np.linalg.norm(np.cross(front, up)) > 0.1:
    #         break
    up = up - np.dot(up, front) * front
    up = up / np.linalg.norm(up)
    side = np.cross(front, up)
    view_matrix = np.eye(4)
    view_matrix[3, :3] = np.array(position)
    view_matrix[2, :3] = -front
    view_matrix[1, :3] = up
    view_matrix[0, :3] = side
    return view_matrix


@dataclass
class Material:
    color: List
    name: str = "1"
    metallic: float = 0.0
    roughness: float = 0.5


def set_geom_mesh_attrs(mesh_geom: UsdGeom.Mesh, obs: trimesh.Trimesh):
    verts, faces = obs.vertices, obs.faces
    mesh_geom.CreatePointsAttr(verts)
    mesh_geom.CreateFaceVertexCountsAttr([3 for _ in range(len(faces))])
    mesh_geom.CreateFaceVertexIndicesAttr(np.ravel(faces).tolist())
    mesh_geom.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    a = UsdGeom.Xformable(mesh_geom)  #
    a.AddTranslateOp()
    a.AddOrientOp()
    a.AddScaleOp()


class UsdHelper:
    def __init__(
        self,
        name: str = "curobo_stage.usd",
        base_frame: str = "/world",
        timesteps: Optional[int] = None,
        dt: float = 0.02,
        interpolation_steps: float = 1,
    ) -> None:
        self.stage = Usd.Stage.CreateNew(name)
        UsdGeom.SetStageUpAxis(self.stage, "Z")
        UsdGeom.SetStageMetersPerUnit(self.stage, 1)
        UsdPhysics.SetStageKilogramsPerUnit(self.stage, 1)
        xform = self.stage.DefinePrim(base_frame, "Xform")
        self.stage.SetDefaultPrim(xform)
        self.dt = dt
        self.interpolation_steps = interpolation_steps
        if timesteps is not None:
            self.stage.SetStartTimeCode(0)
            self.stage.SetEndTimeCode((timesteps - 1) * self.interpolation_steps)
            self.stage.SetTimeCodesPerSecond((24))

        # camera_radius = 150
        # camera_lookat = np.array([100.0, 100, 100])
        # camera_pos = camera_lookat + camera_radius * np_even_sample_points_on_sphere(
        #     3, delta_angle=45
        # )
        # for i in range(len(camera_pos)):
        #     camera_view_matrix = create_view_matrix(camera_pos[i], camera_lookat)
        #     self.add_camera(str(i), camera_view_matrix)

    def add_camera(self, id, cam_view_m):
        camera_path = f"/Camera_{id}"
        camera_prim = self.stage.DefinePrim(camera_path, f"Camera")
        camera = UsdGeom.Camera(self.stage.GetPrimAtPath(camera_path))
        camera.AddTransformOp().Set(Gf.Matrix4d(cam_view_m))
        camera.GetFocalLengthAttr().Set(80.0)
        camera.GetHorizontalApertureAttr().Set(20)
        camera.GetVerticalApertureAttr().Set(20)

    def add_subroot(self, root="/world", sub_root="obstacles"):
        xform = self.stage.DefinePrim(os.path.join(root, sub_root), "Xform")

    def add_meshlst_to_stage(
        self,
        mesh_lst: List[trimesh.Trimesh],
        name_lst: List[str],
        pose_lst: List[List],
        vis_time_lst: List[tuple[float, float]],
        base_frame: str = "/world",
        obstacles_frame: str = "obstacles",
        material: Material | None = None,
    ):
        self.add_subroot(base_frame, obstacles_frame)
        full_path = os.path.join(base_frame, obstacles_frame)

        prim_path = [
            self.add_mesh_to_stage(
                m, name, full_path, visible_time=vt, material=material
            )
            for vt, name, m in zip(vis_time_lst, name_lst, mesh_lst)
        ]

        for i, i_val in enumerate(prim_path):
            curr_prim = self.stage.GetPrimAtPath(i_val)
            form = UsdGeom.Xformable(curr_prim).GetOrderedXformOps()

            for t, p in enumerate(pose_lst):
                position = Gf.Vec3f(p[i][0], p[i][1], p[i][2])
                quat = Gf.Quatf(p[i][3], *p[i][4:-1])
                scale = Gf.Vec3f(p[i][-1], p[i][-1], p[i][-1])
                real_t = (t + vis_time_lst[i][0]) * self.interpolation_steps
                form[0].Set(time=real_t, value=position)
                form[1].Set(time=real_t, value=quat)
                form[2].Set(time=real_t, value=scale)
        return

    def add_mesh_to_stage(
        self,
        mesh: trimesh.Trimesh,
        mesh_name: str,
        base_frame: str = "/world/obstacles",
        visible_time: tuple[float, float] | None = None,
        material: Material | None = None,
    ):
        root_path = os.path.join(
            base_frame,
            "o"
            + mesh_name.replace(".", "_")
            .replace("é", "e")
            .replace("+", "_")
            .replace(":", "_")
            .replace("-", "_"),
        )
        obj_geom = UsdGeom.Mesh.Get(self.stage, root_path)
        if not obj_geom:
            obj_geom = UsdGeom.Mesh.Define(self.stage, root_path)
            set_geom_mesh_attrs(obj_geom, mesh)
            obj_geom.GetVisibilityAttr().Set("invisible", Usd.TimeCode(0))

        if visible_time is not None:
            obj_geom.GetVisibilityAttr().Set("inherited", Usd.TimeCode(visible_time[0]))
            obj_geom.GetVisibilityAttr().Set("invisible", Usd.TimeCode(visible_time[1]))

        if material is not None:
            obj_prim = self.stage.GetPrimAtPath(root_path)
            self.add_material(
                "material_" + material.name,
                root_path,
                material.color,
                obj_prim,
                material.roughness,
                material.metallic,
            )

        return root_path

    def write_stage_to_file(self, file_path: str, flatten: bool = False):
        if flatten:
            usd_str = self.stage.Flatten().ExportToString()
        else:
            usd_str = self.stage.GetRootLayer().ExportToString()
        with open(file_path, "w") as f:
            f.write(usd_str)

    def add_material(
        self,
        material_name: str,
        object_path: str,
        color: List[float],
        obj_prim: Usd.Prim,
        material_roughness: float,
        material_metallic: float,
    ):
        mat_path = os.path.join(object_path, material_name)
        material_usd = UsdShade.Material.Define(self.stage, mat_path)
        pbrShader = UsdShade.Shader.Define(
            self.stage, os.path.join(mat_path, "PbrShader")
        )
        pbrShader.CreateIdAttr("UsdPreviewSurface")
        pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            material_roughness
        )
        pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
            material_metallic
        )
        pbrShader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(color[:3])
        )
        pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(color[:3])
        )
        pbrShader.CreateInput("baseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(color[:3])
        )

        pbrShader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(color[3])
        material_usd.CreateSurfaceOutput().ConnectToSource(
            pbrShader.ConnectableAPI(), "surface"
        )
        obj_prim.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
        UsdShade.MaterialBindingAPI(obj_prim).Bind(material_usd)
        return material_usd

    def save(self):
        self.stage.Save()
