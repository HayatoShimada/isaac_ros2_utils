#!/usr/bin/env python3
# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Standalone script for converting URDF to USD using Isaac Sim (via Isaac Lab).

This script must be run with Isaac Lab's Python environment:
    /workspace/isaaclab/isaaclab.sh -p convert_urdf_standalone.py --urdf_path /path/to/robot.urdf

Or from the installed location:
    isaaclab -p $(ros2 pkg prefix isaac_ros2_scripts)/share/isaac_ros2_scripts/convert_urdf_standalone.py \
        --urdf_path /path/to/robot.urdf
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Convert URDF to USD using Isaac Sim")
    parser.add_argument("--urdf_path", required=True, help="Path to URDF file")
    parser.add_argument("--output_usd_path", default=None, help="Output USD file path")
    parser.add_argument("--fixed_base", action="store_true", help="Fix robot base to world")
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless")

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.urdf_path):
        print(f"[ERROR] URDF file not found: {args.urdf_path}")
        return 1

    # Determine output path
    if args.output_usd_path:
        output_usd_path = args.output_usd_path
    else:
        urdf_dir = os.path.dirname(args.urdf_path)
        urdf_basename = os.path.splitext(os.path.basename(args.urdf_path))[0]
        output_usd_path = os.path.join(urdf_dir, urdf_basename + ".usd")

    print(f"[convert_urdf_standalone] Converting URDF to USD:")
    print(f"  Input:  {args.urdf_path}")
    print(f"  Output: {output_usd_path}")
    print(f"  Fixed base: {args.fixed_base}")

    # Initialize Isaac Lab/Sim (required for URDF import)
    from isaaclab.app import AppLauncher

    # Create app launcher with headless mode
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    return_code = 0
    converted = False

    # Ensure output directory exists
    output_dir = os.path.dirname(output_usd_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Method 1: Try Isaac Lab's UrdfConverter
    if not converted:
        try:
            print("[convert_urdf_standalone] Trying Isaac Lab UrdfConverter...")
            from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

            converter_cfg = UrdfConverterCfg(
                asset_path=args.urdf_path,
                usd_dir=os.path.dirname(output_usd_path),
                usd_file_name=os.path.basename(output_usd_path),
                fix_base=args.fixed_base,
                make_instanceable=False,
            )

            converter = UrdfConverter(converter_cfg)
            actual_usd_path = converter.usd_path

            if os.path.exists(actual_usd_path):
                print(f"[convert_urdf_standalone] SUCCESS (UrdfConverter): {actual_usd_path}")
                # If output path is different, copy or note it
                if actual_usd_path != output_usd_path:
                    print(f"[convert_urdf_standalone] Note: USD created at {actual_usd_path}")
                    # Try to copy to expected location
                    import shutil
                    shutil.copy2(actual_usd_path, output_usd_path)
                    print(f"[convert_urdf_standalone] Copied to: {output_usd_path}")
                converted = True
                return_code = 0
            else:
                print(f"[convert_urdf_standalone] UrdfConverter completed but file not found")
        except Exception as e:
            print(f"[convert_urdf_standalone] UrdfConverter failed: {e}")
            import traceback
            traceback.print_exc()

    # Method 2: Try omni.kit.commands directly
    if not converted:
        try:
            print("[convert_urdf_standalone] Trying omni.kit.commands...")
            import omni
            import omni.kit.commands
            import omni.usd

            # Create new stage for conversion
            omni.usd.get_context().new_stage()

            # Configure URDF import
            status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
            if status:
                import_config.merge_fixed_joints = False
                import_config.convex_decomp = True
                import_config.import_inertia_tensor = True
                import_config.self_collision = False
                import_config.fix_base = args.fixed_base
                import_config.default_drive_strength = 0.0
                import_config.default_position_drive_damping = 0.0
                import_config.distance_scale = 1

                # Import URDF
                print(f"[convert_urdf_standalone] Importing URDF...")
                status, stage_path = omni.kit.commands.execute(
                    "URDFParseAndImportFile",
                    urdf_path=args.urdf_path,
                    import_config=import_config,
                    get_articulation_root=True,
                )

                if status:
                    print(f"[convert_urdf_standalone] URDF imported to stage path: {stage_path}")

                    # Export to USD file
                    stage_handle = omni.usd.get_context().get_stage()
                    stage_handle.Export(output_usd_path)

                    if os.path.exists(output_usd_path):
                        print(f"[convert_urdf_standalone] SUCCESS (omni.kit.commands): {output_usd_path}")
                        converted = True
                        return_code = 0
                    else:
                        print(f"[convert_urdf_standalone] USD file not created")
                else:
                    print(f"[convert_urdf_standalone] URDFParseAndImportFile failed")
            else:
                print(f"[convert_urdf_standalone] URDFCreateImportConfig failed")
        except Exception as e:
            print(f"[convert_urdf_standalone] omni.kit.commands failed: {e}")
            import traceback
            traceback.print_exc()

    if not converted:
        print(f"[convert_urdf_standalone] ERROR: All conversion methods failed")
        return_code = 1

    # Close the simulation
    simulation_app.close()

    return return_code


if __name__ == "__main__":
    sys.exit(main())
