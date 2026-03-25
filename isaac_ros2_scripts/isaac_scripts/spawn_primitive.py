"""Isaac Sim 上に物理付きプリミティブ（Cube, Sphere, Cylinder）を生成する。"""

import math
import omni.usd
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf


def main(
    shape: str = "cube",
    name: str = "cube_0",
    size: float = 0.04,
    scale_x: float = 0.0,
    scale_y: float = 0.0,
    scale_z: float = 0.0,
    x: float = 0.5,
    y: float = 0.0,
    z: float = 0.1,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    mass: float = 0.1,
    color: list = None,
    static_friction: float = 1.0,
    dynamic_friction: float = 1.0,
    restitution: float = 0.0,
):
    """Create a physics-enabled primitive on the Isaac Sim stage.

    Args:
        shape: "cube", "sphere", or "cylinder"
        name: Prim name (placed under /World/)
        size: Size in meters (edge length for cube, diameter for sphere/cylinder).
              Used as base size; overridden per-axis if scale_x/y/z are set.
        scale_x, scale_y, scale_z: Non-uniform scale. If all zero, uniform size is used.
              For cube: size=1.0 + scale_x=0.8,scale_y=0.6,scale_z=0.02 → 0.8m x 0.6m x 0.02m
        x, y, z: Position in meters
        roll, pitch, yaw: Rotation in radians
        mass: Mass in kg
        color: RGB color as [r, g, b] (0-1 range)
        static_friction: Static friction coefficient
        dynamic_friction: Dynamic friction coefficient
        restitution: Bounciness (0 = no bounce)
    """
    if color is None:
        color = [0.2, 0.6, 1.0]

    use_nonuniform = (scale_x > 0.0 or scale_y > 0.0 or scale_z > 0.0)

    stage = omni.usd.get_context().get_stage()
    prim_path = f"/World/{name}"

    # Create shape (unit size, scaled via xform if non-uniform)
    if shape == "cube":
        geom = UsdGeom.Cube.Define(stage, prim_path)
        geom.GetSizeAttr().Set(1.0 if use_nonuniform else size)
    elif shape == "sphere":
        geom = UsdGeom.Sphere.Define(stage, prim_path)
        geom.GetRadiusAttr().Set(0.5 if use_nonuniform else size / 2.0)
    elif shape == "cylinder":
        geom = UsdGeom.Cylinder.Define(stage, prim_path)
        geom.GetRadiusAttr().Set(0.5 if use_nonuniform else size / 2.0)
        geom.GetHeightAttr().Set(1.0 if use_nonuniform else size)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    prim = stage.GetPrimAtPath(prim_path)

    # Transform
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
    if use_nonuniform:
        xformable.AddScaleOp().Set(Gf.Vec3f(scale_x, scale_y, scale_z))
    xformable.AddRotateXYZOp().Set(Gf.Vec3f(
        math.degrees(roll), math.degrees(pitch), math.degrees(yaw),
    ))

    # Rigid body physics
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.GetMassAttr().Set(mass)

    # Collision
    UsdPhysics.CollisionAPI.Apply(prim)

    # Physics material with friction
    mat_path = f"{prim_path}/physics_material"
    UsdPhysics.MaterialAPI.Apply(stage.DefinePrim(mat_path, "Material"))
    mat_prim = stage.GetPrimAtPath(mat_path)
    physics_mat = UsdPhysics.MaterialAPI(mat_prim)
    physics_mat.CreateStaticFrictionAttr().Set(static_friction)
    physics_mat.CreateDynamicFrictionAttr().Set(dynamic_friction)
    physics_mat.CreateRestitutionAttr().Set(restitution)

    # Bind material to prim
    binding_api = UsdPhysics.MaterialAPI(prim)
    if not binding_api:
        binding_api = UsdPhysics.MaterialAPI.Apply(prim)
    rel = prim.CreateRelationship("physics:materialBinding", custom=False)
    if not rel:
        rel = prim.GetRelationship("physics:materialBinding")
    rel.SetTargets([Sdf.Path(mat_path)])

    # Color
    geom.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

    if use_nonuniform:
        print(f"[spawn_primitive] Created {shape} '{name}' at ({x}, {y}, {z}), "
              f"scale=({scale_x}, {scale_y}, {scale_z}), mass={mass}kg")
    else:
        print(f"[spawn_primitive] Created {shape} '{name}' at ({x}, {y}, {z}), "
              f"size={size}m, mass={mass}kg, friction={static_friction}")

    return prim
