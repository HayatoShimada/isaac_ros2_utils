#!/usr/bin/env python3
# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
ROS 2 node for preparing a robot for Isaac Lab.

This node:
1. Converts URDF to USD (using Isaac Lab UrdfConverter or REST API)
2. Generates ArticulationCfg Python file from URDF

Usage:
    ros2 run isaac_ros2_scripts prepare_robot_for_isaaclab --ros-args \
        -p urdf_path:=/path/to/robot.urdf \
        -p output_dir:=/path/to/output

Parameters:
    urdf_path: Path to the URDF file (required)
    output_dir: Directory for output files (default: same as URDF)
    fixed_base: Whether to fix the robot base (default: false)
    api_url: REST API URL for USD conversion (default: http://localhost:8081)
"""

import os
import sys
import json
import subprocess
import urllib.request
import urllib.error

# Add isaac_scripts to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
isaac_scripts_dir = os.path.join(os.path.dirname(script_dir), 'isaac_scripts')
if isaac_scripts_dir not in sys.path:
    sys.path.insert(0, isaac_scripts_dir)

# Also check share directory (for installed package)
share_dir = None
try:
    from ament_index_python.packages import get_package_share_directory
    share_dir = get_package_share_directory('isaac_ros2_scripts')
    if share_dir not in sys.path:
        sys.path.insert(0, share_dir)
except:
    pass

import rclpy
from rclpy.node import Node


class PrepareRobotForIsaacLabNode(Node):
    """ROS 2 node for preparing robot assets for Isaac Lab."""

    def __init__(self):
        super().__init__('prepare_robot_for_isaaclab')

        # Declare parameters
        self.declare_parameter('urdf_path', '')
        self.declare_parameter('output_dir', '')
        self.declare_parameter('fixed_base', False)
        self.declare_parameter('api_url', 'http://localhost:8081')

        urdf_path = self.get_parameter('urdf_path').get_parameter_value().string_value
        output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        fixed_base = self.get_parameter('fixed_base').get_parameter_value().bool_value
        api_url = self.get_parameter('api_url').get_parameter_value().string_value

        if not urdf_path:
            self.get_logger().error('urdf_path parameter is required')
            return

        if not os.path.exists(urdf_path):
            self.get_logger().error(f'URDF file not found: {urdf_path}')
            return

        # Determine output paths
        urdf_basename = os.path.splitext(os.path.basename(urdf_path))[0]

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            usd_path = os.path.join(output_dir, urdf_basename + ".usd")
            config_path = os.path.join(output_dir, urdf_basename + "_cfg.py")
        else:
            urdf_dir = os.path.dirname(urdf_path)
            usd_path = os.path.join(urdf_dir, urdf_basename + ".usd")
            config_path = os.path.join(urdf_dir, urdf_basename + "_cfg.py")

        self.get_logger().info(f'Preparing robot for Isaac Lab:')
        self.get_logger().info(f'  URDF: {urdf_path}')
        self.get_logger().info(f'  USD:  {usd_path}')
        self.get_logger().info(f'  Config: {config_path}')

        # Generate ArticulationCfg
        try:
            import generate_isaaclab_config
            code = generate_isaaclab_config.main(
                urdf_path=urdf_path,
                usd_path=usd_path,
                output_path=config_path,
            )
            self.get_logger().info(f'Generated ArticulationCfg: {config_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to generate ArticulationCfg: {e}')

        # Try to convert URDF to USD using multiple methods
        usd_converted = False

        # Method 1: Try Isaac Lab's UrdfConverter (in Isaac Lab environment)
        if not usd_converted:
            usd_converted = self._convert_with_isaaclab(urdf_path, usd_path, fixed_base)

        # Method 2: Try REST API (for external conversion)
        if not usd_converted:
            usd_converted = self._convert_with_rest_api(urdf_path, usd_path, fixed_base, api_url)

        # Method 3: Try Isaac Sim's convert_urdf module
        if not usd_converted:
            usd_converted = self._convert_with_isaac_sim(urdf_path, usd_path, fixed_base)

        if usd_converted:
            self.get_logger().info(f'Converted URDF to USD: {usd_path}')
        else:
            self.get_logger().warn('USD conversion not performed. Options:')
            self.get_logger().warn('  1. Run in Isaac Lab environment (isaaclab python)')
            self.get_logger().warn('  2. Start REST API server (ros2 launch isaac_diffbot_sim isaaclab_api.launch.py)')
            self.get_logger().warn('  3. Run in Isaac Sim environment')

        self.get_logger().info('Robot preparation complete')

    def _convert_with_isaaclab(self, urdf_path: str, usd_path: str, fixed_base: bool) -> bool:
        """Try to convert URDF to USD using Isaac Lab's standalone script."""
        # Find the standalone conversion script
        script_path = None

        # Check installed package location
        if share_dir is not None:
            candidate = os.path.join(share_dir, 'convert_urdf_standalone.py')
            if os.path.exists(candidate):
                script_path = candidate

        # Check source location
        if script_path is None:
            source_script = os.path.join(isaac_scripts_dir, 'convert_urdf_standalone.py')
            if os.path.exists(source_script):
                script_path = source_script

        if script_path is None:
            self.get_logger().debug('convert_urdf_standalone.py not found')
            return False

        # Check if isaaclab.sh exists (indicates Isaac Lab environment)
        isaaclab_sh = '/workspace/isaaclab/isaaclab.sh'
        if not os.path.exists(isaaclab_sh):
            self.get_logger().debug('Isaac Lab environment not found (isaaclab.sh missing)')
            return False

        self.get_logger().info('Using Isaac Lab standalone script for USD conversion...')

        # Build command
        cmd = [
            isaaclab_sh, '-p', script_path,
            '--urdf_path', urdf_path,
            '--output_usd_path', usd_path,
            '--headless',
        ]
        if fixed_base:
            cmd.append('--fixed_base')

        try:
            self.get_logger().info(f'Running: {" ".join(cmd)}')
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
            )

            # Log output
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    self.get_logger().info(f'[isaaclab] {line}')
            if result.stderr:
                for line in result.stderr.strip().split('\n'):
                    self.get_logger().warn(f'[isaaclab] {line}')

            if result.returncode == 0 and os.path.exists(usd_path):
                self.get_logger().info(f'Isaac Lab conversion completed: {usd_path}')
                return True
            else:
                self.get_logger().warn(f'Isaac Lab conversion failed with code {result.returncode}')
                return False

        except subprocess.TimeoutExpired:
            self.get_logger().warn('Isaac Lab conversion timed out')
            return False
        except Exception as e:
            self.get_logger().warn(f'Isaac Lab conversion failed: {e}')
            return False

    def _convert_with_rest_api(self, urdf_path: str, usd_path: str, fixed_base: bool, api_url: str) -> bool:
        """Try to convert URDF to USD using REST API."""
        try:
            self.get_logger().info(f'Trying REST API for USD conversion: {api_url}')

            # Prepare request data
            data = json.dumps({
                'urdf_path': urdf_path,
                'output_usd_path': usd_path,
                'fixed_base': fixed_base,
            }).encode('utf-8')

            # Make the request
            req = urllib.request.Request(
                f'{api_url}/convert_urdf',
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                if result.get('success'):
                    self.get_logger().info(f'REST API conversion successful: {result.get("message")}')
                    # Check if the USD file was actually created
                    if os.path.exists(usd_path):
                        return True
                    else:
                        self.get_logger().warn('REST API returned success but USD file not found')
                        return False
                else:
                    self.get_logger().warn(f'REST API conversion failed: {result.get("message")}')
                    return False
        except urllib.error.URLError as e:
            self.get_logger().debug(f'REST API not available: {e}')
            return False
        except Exception as e:
            self.get_logger().warn(f'REST API conversion failed: {e}')
            return False

    def _convert_with_isaac_sim(self, urdf_path: str, usd_path: str, fixed_base: bool) -> bool:
        """Try to convert URDF to USD using Isaac Sim's convert_urdf module."""
        try:
            import convert_urdf
            self.get_logger().info('Using Isaac Sim convert_urdf for USD conversion...')
            convert_urdf.convert_urdf_to_usd(
                urdf_path=urdf_path,
                output_usd_path=usd_path,
                fixed_base=fixed_base,
            )
            return True
        except ImportError:
            self.get_logger().debug('Isaac Sim convert_urdf not available')
            return False
        except Exception as e:
            self.get_logger().warn(f'Isaac Sim convert_urdf failed: {e}')
            return False


def main(args=None):
    rclpy.init(args=args)

    try:
        node = PrepareRobotForIsaacLabNode()
        # Just run once and exit
        rclpy.spin_once(node, timeout_sec=1.0)
        node.destroy_node()
    except Exception as e:
        print(f'Error: {e}')
        return 1
    finally:
        rclpy.shutdown()

    return 0


if __name__ == '__main__':
    sys.exit(main())
