# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""
URDF to USD converter module.

This module provides functionality to convert URDF files to USD format,
applying custom Isaac Sim attributes (isaac_drive_api, isaac_rigid_body, etc.)
during the conversion process.

The converted USD file can be used with both Isaac Sim and Isaac Lab.

Usage:
    # As a module (within Isaac Sim environment)
    from convert_urdf import convert_urdf_to_usd
    usd_path = convert_urdf_to_usd("/path/to/robot.urdf", "/path/to/output.usd")

    # As a standalone script (requires Isaac Sim to be running)
    python convert_urdf.py /path/to/robot.urdf /path/to/output.usd
"""

import os
import math
import xml.etree.ElementTree as ET
import omni
import omni.kit.commands
import omni.usd
from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema, Sdf, Usd
from isaacsim.core.prims import GeometryPrim
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.asset.importer.urdf import _urdf
from omni.physx import utils
from omni.physx.scripts import physicsUtils

DEFAULT_CONVEX_DECOMPOSITION_COLLISION_SHRINK_WARP = True
DEFAULT_CONVEX_DECOMPOSITION_COLLISION_MAX_CONVEX_HULLS = 32


def find_prim_path_by_name(start_stage, prim_name):
    """Recursively search for a prim by name and return its path."""
    for prim in start_stage.GetAllChildren():
        if prim.GetName() == prim_name:
            return prim.GetPath().pathString
        else:
            ret = find_prim_path_by_name(prim, prim_name)
            if ret is not None:
                return ret
    return None


def _apply_rigid_body_materials(stage, urdf_root, robot_name):
    """Apply isaac_rigid_body material properties from URDF to USD."""
    for child in urdf_root.findall('./material'):
        dynamic_friction = 0.0
        static_friction = 0.0
        restitution = 0.0
        rigid_body_list = child.findall('./isaac_rigid_body')
        if len(rigid_body_list) > 0:
            if "dynamic_friction" in rigid_body_list[0].attrib:
                dynamic_friction = float(rigid_body_list[0].attrib["dynamic_friction"])
            if "static_friction" in rigid_body_list[0].attrib:
                static_friction = float(rigid_body_list[0].attrib["static_friction"])
            if "restitution" in rigid_body_list[0].attrib:
                restitution = float(rigid_body_list[0].attrib["restitution"])
        material_path = "/" + robot_name + "/Looks/material_" + child.attrib["name"]
        utils.addRigidBodyMaterial(
            stage, material_path,
            density=None,
            staticFriction=static_friction,
            dynamicFriction=dynamic_friction,
            restitution=restitution
        )


def _apply_collision_materials(stage_handle, urdf_root, robot_name):
    """Apply collision materials and convex decomposition settings from URDF to USD."""
    for link in urdf_root.findall("./link"):
        joint_prim_path = find_prim_path_by_name(
            stage_handle.GetPrimAtPath("/" + robot_name),
            link.attrib["name"]
        )
        if joint_prim_path is None:
            continue

        collision_list = link.findall('./collision')
        if len(collision_list) == 0:
            continue

        prim = stage_handle.GetPrimAtPath(joint_prim_path + "/collisions")
        if not prim.IsValid():
            continue

        # Apply material to collision prim
        material_list = link.findall('./visual/material')
        if len(material_list) > 0:
            material_path = "/" + robot_name + "/Looks/material_" + material_list[0].attrib["name"]
            physicsUtils.add_physics_material_to_prim(stage_handle, prim, material_path)

        # Apply convex decomposition settings
        token_attr = prim.GetAttribute("physics:approximation")
        if token_attr.IsValid():
            token_value = token_attr.Get()
            if token_value == "convexDecomposition":
                physx_convexdecomp_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                physx_convexdecomp_api.GetShrinkWrapAttr().Set(DEFAULT_CONVEX_DECOMPOSITION_COLLISION_SHRINK_WARP)

                convex_decomposition_list = collision_list[0].findall('./convex_decomposition')
                if len(convex_decomposition_list) > 0:
                    max_hulls = int(convex_decomposition_list[0].attrib["max_convex_hulls"])
                    physx_convexdecomp_api.GetMaxConvexHullsAttr().Set(max_hulls)
                else:
                    physx_convexdecomp_api.GetMaxConvexHullsAttr().Set(
                        DEFAULT_CONVEX_DECOMPOSITION_COLLISION_MAX_CONVEX_HULLS
                    )


def _apply_drive_api_settings(stage_handle, urdf_root, robot_name):
    """Apply isaac_drive_api joint parameters from URDF to USD."""
    urdf_joints = []
    joint_types = []

    for child in urdf_root.findall('./joint'):
        if child.attrib["type"] == "continuous":
            urdf_joints.append(child)
            joint_types.append("angular")
        elif child.attrib["type"] == "revolute":
            urdf_joints.append(child)
            joint_types.append("angular")
        elif child.attrib["type"] == "prismatic":
            urdf_joints.append(child)
            joint_types.append("linear")

    for index, joint in enumerate(urdf_joints):
        joint_prim_path = find_prim_path_by_name(
            stage_handle.GetPrimAtPath("/" + robot_name),
            joint.attrib["name"]
        )
        if joint_prim_path is None:
            continue

        drive = UsdPhysics.DriveAPI.Get(
            stage_handle.GetPrimAtPath(joint_prim_path),
            joint_types[index]
        )
        joint_api = PhysxSchema.PhysxJointAPI(stage_handle.GetPrimAtPath(joint_prim_path))

        api_list = joint.findall('./isaac_drive_api')
        if len(api_list) > 0:
            if "damping" in api_list[0].attrib:
                drive.CreateDampingAttr().Set(float(api_list[0].attrib["damping"]))
            if "stiffness" in api_list[0].attrib:
                drive.CreateStiffnessAttr().Set(float(api_list[0].attrib["stiffness"]))
            if "joint_friction" in api_list[0].attrib:
                joint_api.CreateJointFrictionAttr().Set(float(api_list[0].attrib["joint_friction"]))


def import_urdf_to_stage(
    urdf_path: str,
    fixed_base: bool = False,
    merge_fixed_joints: bool = False,
    convex_decomp: bool = True,
    import_inertia_tensor: bool = True,
    self_collision: bool = False,
):
    """
    Import a URDF file directly into the current stage with custom Isaac Sim attributes.

    This function imports a URDF into the current stage and applies custom
    Isaac Sim attributes. Use this for runtime spawning in Isaac Sim.

    Args:
        urdf_path: Path to the input URDF file
        fixed_base: Whether to fix the robot base to the world
        merge_fixed_joints: Whether to merge fixed joints
        convex_decomp: Whether to use convex decomposition for collision meshes
        import_inertia_tensor: Whether to import inertia tensors from URDF
        self_collision: Whether to enable self-collision detection

    Returns:
        tuple: (stage_path, robot_name) - Path to the imported robot prim and robot name

    Raises:
        FileNotFoundError: If the URDF file does not exist
        RuntimeError: If the import fails
    """
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    # Configure URDF import
    urdf_interface = _urdf.acquire_urdf_interface()
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")

    import_config.merge_fixed_joints = merge_fixed_joints
    import_config.convex_decomp = convex_decomp
    import_config.import_inertia_tensor = import_inertia_tensor
    import_config.self_collision = self_collision
    import_config.fix_base = fixed_base
    import_config.default_drive_strength = 0.0
    import_config.default_position_drive_damping = 0.0
    import_config.distance_scale = 1

    # Import URDF
    status, stage_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
        get_articulation_root=True,
    )

    if not status:
        raise RuntimeError(f"Failed to import URDF: {urdf_path}")

    # Get stage handle
    stage_handle = omni.usd.get_context().get_stage()

    # Parse URDF to get robot name and custom attributes
    urdf_root = ET.parse(urdf_path).getroot()
    robot_name = None
    for child in urdf_root.iter("robot"):
        robot_name = child.attrib["name"]
        break

    if robot_name is None:
        raise RuntimeError("Could not find robot name in URDF")

    # Apply custom Isaac Sim attributes
    _apply_rigid_body_materials(stage_handle, urdf_root, robot_name)
    _apply_collision_materials(stage_handle, urdf_root, robot_name)
    _apply_drive_api_settings(stage_handle, urdf_root, robot_name)

    print(f"[convert_urdf] Successfully imported URDF to stage: {stage_path}")
    return stage_path, robot_name


def convert_urdf_to_usd(
    urdf_path: str,
    output_usd_path: str = None,
    fixed_base: bool = False,
    merge_fixed_joints: bool = False,
    convex_decomp: bool = True,
    import_inertia_tensor: bool = True,
    self_collision: bool = False,
) -> str:
    """
    Convert a URDF file to USD format with custom Isaac Sim attributes.

    This function imports a URDF file into a temporary stage, applies custom
    Isaac Sim attributes (isaac_drive_api, isaac_rigid_body, convex_decomposition),
    and exports the result as a USD file.

    For runtime spawning in Isaac Sim, use import_urdf_to_stage() instead.
    This function is intended for pre-conversion (e.g., for Isaac Lab).

    Args:
        urdf_path: Path to the input URDF file
        output_usd_path: Path for the output USD file. If None, generates path
                        based on URDF filename in the same directory.
        fixed_base: Whether to fix the robot base to the world
        merge_fixed_joints: Whether to merge fixed joints
        convex_decomp: Whether to use convex decomposition for collision meshes
        import_inertia_tensor: Whether to import inertia tensors from URDF
        self_collision: Whether to enable self-collision detection

    Returns:
        str: Path to the generated USD file

    Raises:
        FileNotFoundError: If the URDF file does not exist
        RuntimeError: If the conversion fails
    """
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    # Generate output path if not provided
    if output_usd_path is None:
        urdf_dir = os.path.dirname(urdf_path)
        urdf_basename = os.path.splitext(os.path.basename(urdf_path))[0]
        output_usd_path = os.path.join(urdf_dir, urdf_basename + ".usd")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_usd_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save current stage reference
    current_stage = omni.usd.get_context().get_stage()
    current_stage_url = None
    if current_stage:
        current_stage_url = current_stage.GetRootLayer().identifier

    # Create new stage for conversion
    omni.usd.get_context().new_stage()

    try:
        # Import URDF to the new stage
        stage_path, robot_name = import_urdf_to_stage(
            urdf_path=urdf_path,
            fixed_base=fixed_base,
            merge_fixed_joints=merge_fixed_joints,
            convex_decomp=convex_decomp,
            import_inertia_tensor=import_inertia_tensor,
            self_collision=self_collision,
        )

        # Export to USD file
        stage_handle = omni.usd.get_context().get_stage()
        stage_handle.Export(output_usd_path)
        print(f"[convert_urdf] Successfully converted URDF to USD: {output_usd_path}")

        return output_usd_path

    finally:
        # Restore original stage if it existed
        if current_stage_url:
            omni.usd.get_context().open_stage(current_stage_url)


def main(urdf_path: str, output_usd_path: str = None, fixed: bool = False) -> str:
    """
    Main entry point for URDF to USD conversion.

    Args:
        urdf_path: Path to the input URDF file
        output_usd_path: Optional path for the output USD file
        fixed: Whether to fix the robot base to the world

    Returns:
        str: Path to the generated USD file
    """
    return convert_urdf_to_usd(
        urdf_path=urdf_path,
        output_usd_path=output_usd_path,
        fixed_base=fixed,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python convert_urdf.py <urdf_path> [output_usd_path] [fixed]")
        sys.exit(1)

    urdf_path = sys.argv[1]
    output_usd_path = sys.argv[2] if len(sys.argv) > 2 else None
    fixed = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else False

    result_path = main(urdf_path, output_usd_path, fixed)
    print(f"Output: {result_path}")
