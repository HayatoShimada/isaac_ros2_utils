#!/usr/bin/env python3
# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
ROS 2 entry point for Isaac Lab REST API server.

This script launches the Isaac Lab REST API server, which provides endpoints
for URDF conversion, ArticulationCfg generation, and training control.

Usage:
    ros2 run isaac_ros2_scripts isaaclab_api
    ros2 run isaac_ros2_scripts isaaclab_api --ros-args -p port:=8081

API Documentation:
    http://localhost:8081/docs (Swagger UI)
"""

import sys
import os

# Add isaac_scripts to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
isaac_scripts_dir = os.path.join(os.path.dirname(script_dir), 'isaac_scripts')
if isaac_scripts_dir not in sys.path:
    sys.path.insert(0, isaac_scripts_dir)

import rclpy
from rclpy.node import Node


class IsaacLabApiNode(Node):
    """ROS 2 node that hosts the Isaac Lab REST API server."""

    def __init__(self):
        super().__init__('isaaclab_api')

        # Declare parameters
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 8081)

        host = self.get_parameter('host').get_parameter_value().string_value
        port = self.get_parameter('port').get_parameter_value().integer_value

        self.get_logger().info(f'Starting Isaac Lab REST API on {host}:{port}')

        # Import and start the API server
        try:
            from isaaclab_rest_api import IsaacLabRestApi
            self.api_server = IsaacLabRestApi(host=host, port=port)
            self.api_server.start()
            self.get_logger().info(f'Isaac Lab REST API started')
            self.get_logger().info(f'API documentation: http://{host}:{port}/docs')
        except ImportError as e:
            self.get_logger().error(f'Failed to import isaaclab_rest_api: {e}')
            self.get_logger().error('Make sure FastAPI and uvicorn are installed')
            raise


def main(args=None):
    rclpy.init(args=args)

    try:
        node = IsaacLabApiNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
        return 1
    finally:
        rclpy.shutdown()

    return 0


if __name__ == '__main__':
    sys.exit(main())
