# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""
Isaac Lab ArticulationCfg generator from URDF.

This module generates Isaac Lab configuration files (ArticulationCfg) from URDF files,
extracting joint information from ros2_control tags.

Usage:
    python generate_isaaclab_config.py /path/to/robot.urdf /path/to/robot.usd --output robot_cfg.py
"""

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class JointConfig:
    """Configuration for a single joint extracted from URDF."""
    name: str
    joint_type: str  # revolute, continuous, prismatic, fixed
    command_interface: str  # position, velocity, effort
    state_interfaces: List[str] = field(default_factory=list)
    min_limit: Optional[float] = None
    max_limit: Optional[float] = None
    initial_value: float = 0.0
    effort_limit: Optional[float] = None
    velocity_limit: Optional[float] = None


@dataclass
class RobotConfig:
    """Robot configuration extracted from URDF."""
    name: str
    joints: List[JointConfig] = field(default_factory=list)


def parse_urdf(urdf_path: str) -> RobotConfig:
    """
    Parse URDF file to extract robot configuration.

    Args:
        urdf_path: Path to the URDF file

    Returns:
        RobotConfig containing robot name and joint configurations
    """
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Get robot name
    robot_name = root.attrib.get("name", "robot")

    # Parse joint definitions from <joint> tags
    joint_types = {}
    joint_limits = {}
    for joint_elem in root.findall(".//joint"):
        joint_name = joint_elem.attrib.get("name")
        joint_type = joint_elem.attrib.get("type", "fixed")
        joint_types[joint_name] = joint_type

        # Parse limits
        limit_elem = joint_elem.find("limit")
        if limit_elem is not None:
            joint_limits[joint_name] = {
                "lower": float(limit_elem.attrib.get("lower", 0)),
                "upper": float(limit_elem.attrib.get("upper", 0)),
                "effort": float(limit_elem.attrib.get("effort", 0)),
                "velocity": float(limit_elem.attrib.get("velocity", 0)),
            }

    # Parse ros2_control section
    joints = []
    ros2_control_elem = root.find(".//ros2_control")
    if ros2_control_elem is not None:
        for joint_elem in ros2_control_elem.findall("joint"):
            joint_name = joint_elem.attrib.get("name")
            joint_type = joint_types.get(joint_name, "revolute")

            # Get command interface
            command_interface = "position"  # default
            cmd_elem = joint_elem.find("command_interface")
            if cmd_elem is not None:
                command_interface = cmd_elem.attrib.get("name", "position")
                # Parse min/max from command_interface params
                min_param = cmd_elem.find("param[@name='min']")
                max_param = cmd_elem.find("param[@name='max']")

            # Get state interfaces
            state_interfaces = []
            initial_value = 0.0
            for state_elem in joint_elem.findall("state_interface"):
                state_name = state_elem.attrib.get("name")
                state_interfaces.append(state_name)

                # Get initial value if specified
                init_param = state_elem.find("param[@name='initial_value']")
                if init_param is not None and init_param.text:
                    try:
                        initial_value = float(init_param.text)
                    except ValueError:
                        pass

            # Get limits from joint definition
            limits = joint_limits.get(joint_name, {})

            joint_config = JointConfig(
                name=joint_name,
                joint_type=joint_type,
                command_interface=command_interface,
                state_interfaces=state_interfaces,
                min_limit=limits.get("lower"),
                max_limit=limits.get("upper"),
                initial_value=initial_value,
                effort_limit=limits.get("effort"),
                velocity_limit=limits.get("velocity"),
            )
            joints.append(joint_config)

    return RobotConfig(name=robot_name, joints=joints)


def _to_python_identifier(name: str) -> str:
    """Convert a name to a valid Python identifier."""
    # Replace invalid characters with underscores
    identifier = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure it doesn't start with a number
    if identifier and identifier[0].isdigit():
        identifier = '_' + identifier
    return identifier.upper()


def generate_articulation_cfg(
    robot_config: RobotConfig,
    usd_path: str,
    output_path: str = None,
    class_name: str = None,
) -> str:
    """
    Generate Isaac Lab ArticulationCfg Python code from robot configuration.

    Args:
        robot_config: Robot configuration from parse_urdf()
        usd_path: Path to the USD file (relative or absolute)
        output_path: Path to save the generated Python file
        class_name: Name for the configuration class (default: RobotName + "Cfg")

    Returns:
        Generated Python code as a string
    """
    if class_name is None:
        class_name = _to_python_identifier(robot_config.name) + "_CFG"

    # Group joints by command interface type
    position_joints = [j for j in robot_config.joints if j.command_interface == "position"]
    velocity_joints = [j for j in robot_config.joints if j.command_interface == "velocity"]
    effort_joints = [j for j in robot_config.joints if j.command_interface == "effort"]

    # Generate initial joint positions dict
    initial_positions = {}
    for joint in robot_config.joints:
        if joint.initial_value != 0.0:
            initial_positions[joint.name] = joint.initial_value

    # Build the Python code
    code = f'''"""
Isaac Lab configuration for {robot_config.name}.

This file was auto-generated from URDF by generate_isaaclab_config.py.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


{class_name} = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="{usd_path}",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={{
'''

    # Add initial joint positions
    for joint_name, value in initial_positions.items():
        code += f'            "{joint_name}": {value},\n'

    code += '''        }},
    ),
    actuators={
'''

    # Generate actuator configs based on command interface
    if position_joints:
        joint_names = [j.name for j in position_joints]
        # Use regex pattern if there's a common pattern
        if len(joint_names) > 1:
            joint_pattern = "|".join(joint_names)
            code += f'''        "position_actuators": ImplicitActuatorCfg(
            joint_names_expr="({joint_pattern})",
            stiffness=100.0,
            damping=10.0,
        ),
'''
        else:
            code += f'''        "position_actuators": ImplicitActuatorCfg(
            joint_names_expr="{joint_names[0]}",
            stiffness=100.0,
            damping=10.0,
        ),
'''

    if velocity_joints:
        joint_names = [j.name for j in velocity_joints]
        if len(joint_names) > 1:
            joint_pattern = "|".join(joint_names)
            code += f'''        "velocity_actuators": ImplicitActuatorCfg(
            joint_names_expr="({joint_pattern})",
            stiffness=0.0,
            damping=10.0,
        ),
'''
        else:
            code += f'''        "velocity_actuators": ImplicitActuatorCfg(
            joint_names_expr="{joint_names[0]}",
            stiffness=0.0,
            damping=10.0,
        ),
'''

    if effort_joints:
        joint_names = [j.name for j in effort_joints]
        if len(joint_names) > 1:
            joint_pattern = "|".join(joint_names)
            code += f'''        "effort_actuators": ImplicitActuatorCfg(
            joint_names_expr="({joint_pattern})",
            stiffness=0.0,
            damping=0.0,
        ),
'''
        else:
            code += f'''        "effort_actuators": ImplicitActuatorCfg(
            joint_names_expr="{joint_names[0]}",
            stiffness=0.0,
            damping=0.0,
        ),
'''

    code += '''    },
)
'''

    # Generate helper information as comments
    code += f'''

# Robot: {robot_config.name}
# Joints:
'''
    for joint in robot_config.joints:
        code += f'#   - {joint.name}: {joint.joint_type}, command={joint.command_interface}\n'

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(code)
        print(f"[generate_isaaclab_config] Generated: {output_path}")

    return code


def main(
    urdf_path: str,
    usd_path: str,
    output_path: str = None,
    class_name: str = None,
) -> str:
    """
    Main entry point for generating Isaac Lab configuration from URDF.

    Args:
        urdf_path: Path to the URDF file
        usd_path: Path to the USD file
        output_path: Path to save the generated Python file
        class_name: Name for the configuration class

    Returns:
        Generated Python code as a string
    """
    robot_config = parse_urdf(urdf_path)
    return generate_articulation_cfg(
        robot_config=robot_config,
        usd_path=usd_path,
        output_path=output_path,
        class_name=class_name,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Isaac Lab ArticulationCfg from URDF"
    )
    parser.add_argument("urdf_path", help="Path to the URDF file")
    parser.add_argument("usd_path", help="Path to the USD file")
    parser.add_argument(
        "--output", "-o",
        help="Output Python file path (default: <robot_name>_cfg.py)"
    )
    parser.add_argument(
        "--class-name", "-c",
        help="Configuration class name (default: <ROBOT_NAME>_CFG)"
    )

    args = parser.parse_args()

    output = args.output
    if output is None:
        basename = os.path.splitext(os.path.basename(args.urdf_path))[0]
        output = f"{basename}_cfg.py"

    code = main(
        urdf_path=args.urdf_path,
        usd_path=args.usd_path,
        output_path=output,
        class_name=args.class_name,
    )

    if not args.output:
        print(code)
