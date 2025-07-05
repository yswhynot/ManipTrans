#!/usr/bin/env python3
"""
Generate Inspire Hand URDF from configuration file.
This script creates a URDF with box geometries based on configurable dimensions.
"""

import yaml
import argparse
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, Any, List


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def calculate_inertia_tensor(width: float, length: float, height: float, mass: float) -> List[float]:
    """Calculate inertia tensor for a box with given dimensions and mass."""
    # For a box with dimensions (w, l, h) and mass m:
    # Ixx = m/12 * (l^2 + h^2)
    # Iyy = m/12 * (w^2 + h^2) 
    # Izz = m/12 * (w^2 + l^2)
    # Ixy = Ixz = Iyz = 0 (assuming uniform density and centered mass)
    
    w, l, h = width, length, height
    m = mass
    
    ixx = m / 12.0 * (l**2 + h**2)
    iyy = m / 12.0 * (w**2 + h**2)
    izz = m / 12.0 * (w**2 + l**2)
    
    return [ixx, 0.0, 0.0, iyy, 0.0, izz]


def create_element_with_attributes(tag: str, **kwargs) -> ET.Element:
    """Create an XML element with attributes."""
    element = ET.Element(tag)
    for key, value in kwargs.items():
        element.set(key, str(value))
    return element


def create_origin_element(xyz: List[float], rpy: List[float]) -> ET.Element:
    """Create an origin element with xyz and rpy attributes."""
    return create_element_with_attributes(
        'origin',
        xyz=f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}",
        rpy=f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"
    )


def create_inertia_element(inertia: List[float]) -> ET.Element:
    """Create an inertia element with tensor values."""
    return create_element_with_attributes(
        'inertia',
        ixx=f"{inertia[0]:.6e}",
        ixy=f"{inertia[1]:.6e}",
        ixz=f"{inertia[2]:.6e}",
        iyy=f"{inertia[3]:.6e}",
        iyz=f"{inertia[4]:.6e}",
        izz=f"{inertia[5]:.6e}"
    )


def create_link_element(link_name: str, box_config: Dict[str, Any], hand_side: str) -> ET.Element:
    """Create a link element with box geometry."""
    width = box_config['width']
    length = box_config['length']
    height = box_config['height']
    mass = box_config['mass']
    visual_origin = box_config['visual_origin']
    
    # Calculate inertia tensor
    inertia = calculate_inertia_tensor(width, length, height, mass)
    
    # Split visual origin into xyz and rpy
    visual_xyz = visual_origin[:3]
    visual_rpy = visual_origin[3:]
    
    # Create link element
    link = create_element_with_attributes('link', name=f"{hand_side}{link_name}")
    
    # Inertial element (centered at origin)
    inertial = ET.SubElement(link, 'inertial')
    ET.SubElement(inertial, 'origin', xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, 'mass', value=str(mass))
    inertial.append(create_inertia_element(inertia))
    
    # Visual element
    visual = ET.SubElement(link, 'visual')
    visual.append(create_origin_element(visual_xyz, visual_rpy))
    
    geometry = ET.SubElement(visual, 'geometry')
    ET.SubElement(geometry, 'box', size=f"{width} {length} {height}")
    
    material = ET.SubElement(visual, 'material', name="")
    ET.SubElement(material, 'color', rgba="0.1 0.1 0.1 1")
    
    # Collision element
    collision = ET.SubElement(link, 'collision')
    collision.append(create_origin_element(visual_xyz, visual_rpy))
    
    collision_geometry = ET.SubElement(collision, 'geometry')
    ET.SubElement(collision_geometry, 'box', size=f"{width} {length} {height}")
    
    return link


def create_tip_link_element(tip_name: str, radius: float, hand_side: str) -> ET.Element:
    """Create a tip link element with sphere geometry."""
    link = create_element_with_attributes('link', name=f"{hand_side}{tip_name}")
    
    visual = ET.SubElement(link, 'visual')
    ET.SubElement(visual, 'origin', xyz="0 0 0", rpy="0 0 0")
    
    geometry = ET.SubElement(visual, 'geometry')
    ET.SubElement(geometry, 'sphere', radius=str(radius))
    
    material = ET.SubElement(visual, 'material', name="green")
    ET.SubElement(material, 'color', rgba="0 1 0 1")
    
    return link


def create_joint_element(joint_name: str, joint_config: Dict[str, Any], parent_link: str, child_link: str, hand_side: str) -> ET.Element:
    """Create a joint element."""
    joint_type = joint_config.get('type', 'revolute')
    lower = joint_config['lower']
    upper = joint_config['upper']
    effort = joint_config['effort']
    velocity = joint_config['velocity']
    origin = joint_config['origin']
    axis = joint_config['axis']
    
    # Split origin into xyz and rpy
    origin_xyz = origin[:3]
    origin_rpy = origin[3:]
    
    # Create joint element
    joint = create_element_with_attributes('joint', name=f"{hand_side}{joint_name}", type=joint_type)
    
    joint.append(create_origin_element(origin_xyz, origin_rpy))
    ET.SubElement(joint, 'parent', link=f"{hand_side}{parent_link}")
    ET.SubElement(joint, 'child', link=f"{hand_side}{child_link}")
    ET.SubElement(joint, 'axis', xyz=f"{axis[0]:.0f} {axis[1]:.0f} {axis[2]:.0f}")
    ET.SubElement(joint, 'limit', lower=str(lower), upper=str(upper), effort=str(effort), velocity=str(velocity))
    
    # Add mimic joint if specified
    if 'mimic' in joint_config:
        mimic_joint = joint_config['mimic']
        multiplier = joint_config.get('mimic_multiplier', 1.0)
        ET.SubElement(joint, 'mimic', joint=f"{hand_side}{mimic_joint}", multiplier=str(multiplier), offset="0")
    
    return joint


def create_tip_joint_element(tip_name: str, parent_link: str, tip_origin: List[float], hand_side: str) -> ET.Element:
    """Create a tip joint element (fixed)."""
    tip_xyz = tip_origin[:3]
    tip_rpy = tip_origin[3:]
    
    joint = create_element_with_attributes('joint', name=f"{hand_side}{tip_name}_joint", type="fixed")
    ET.SubElement(joint, 'parent', link=f"{hand_side}{parent_link}")
    ET.SubElement(joint, 'child', link=f"{hand_side}{tip_name}")
    joint.append(create_origin_element(tip_xyz, tip_rpy))
    
    return joint


def generate_urdf(config: Dict[str, Any]) -> ET.Element:
    """Generate the complete URDF XML tree."""
    robot_name = config['robot_name']
    hand_side = config['hand_side']
    box_dimensions = config['box_dimensions']
    joint_properties = config['joint_properties']
    tip_radius = config['tip_radius']
    tip_origins = config['tip_origins']
    
    # Create root robot element
    robot = create_element_with_attributes('robot', name=robot_name)
    
    # Define the hand structure
    hand_structure = [
        # Base link
        ('hand_base_link', 'hand_base'),
        
        # Thumb chain
        ('thumb_proximal_base', 'thumb_proximal_base'),
        ('thumb_proximal', 'thumb_proximal'),
        ('thumb_intermediate', 'thumb_intermediate'),
        ('thumb_distal', 'thumb_distal'),
        ('thumb_tip', None),  # Tip link
        
        # Index finger
        ('index_proximal', 'finger_proximal'),
        ('index_intermediate', 'index_intermediate'),
        ('index_tip', None),  # Tip link
        
        # Middle finger
        ('middle_proximal', 'finger_proximal'),
        ('middle_intermediate', 'middle_intermediate'),
        ('middle_tip', None),  # Tip link
        
        # Ring finger
        ('ring_proximal', 'finger_proximal'),
        ('ring_intermediate', 'ring_intermediate'),
        ('ring_tip', None),  # Tip link
        
        # Pinky finger
        ('pinky_proximal', 'finger_proximal'),
        ('pinky_intermediate', 'pinky_intermediate'),
        ('pinky_tip', None),  # Tip link
    ]
    
    # Create all links
    for link_name, box_type in hand_structure:
        if box_type is None:
            # This is a tip link
            robot.append(create_tip_link_element(link_name, tip_radius, hand_side))
        else:
            # This is a regular link with box geometry
            robot.append(create_link_element(link_name, box_dimensions[box_type], hand_side))
    
    # Define joint connections
    joint_connections = [
        # Thumb chain
        ('thumb_proximal_yaw', 'hand_base_link', 'thumb_proximal_base'),
        ('thumb_proximal_pitch', 'thumb_proximal_base', 'thumb_proximal'),
        ('thumb_intermediate', 'thumb_proximal', 'thumb_intermediate'),
        ('thumb_distal', 'thumb_intermediate', 'thumb_distal'),
        ('thumb_tip', 'thumb_distal', 'thumb_tip'),
        
        # Index finger
        ('index_proximal', 'hand_base_link', 'index_proximal'),
        ('index_intermediate', 'index_proximal', 'index_intermediate'),
        ('index_tip', 'index_intermediate', 'index_tip'),
        
        # Middle finger
        ('middle_proximal', 'hand_base_link', 'middle_proximal'),
        ('middle_intermediate', 'middle_proximal', 'middle_intermediate'),
        ('middle_tip', 'middle_intermediate', 'middle_tip'),
        
        # Ring finger
        ('ring_proximal', 'hand_base_link', 'ring_proximal'),
        ('ring_intermediate', 'ring_proximal', 'ring_intermediate'),
        ('ring_tip', 'ring_intermediate', 'ring_tip'),
        
        # Pinky finger
        ('pinky_proximal', 'hand_base_link', 'pinky_proximal'),
        ('pinky_intermediate', 'pinky_proximal', 'pinky_intermediate'),
        ('pinky_tip', 'pinky_intermediate', 'pinky_tip'),
    ]
    
    # Create all joints
    for joint_name, parent_link, child_link in joint_connections:
        if joint_name.endswith('_tip'):
            # This is a tip joint
            tip_key = joint_name.replace('_joint', '')
            robot.append(create_tip_joint_element(tip_key, parent_link, tip_origins[tip_key], hand_side))
        else:
            # This is a regular joint
            joint_config = joint_properties[joint_name]
            joint_config['type'] = 'revolute'
            robot.append(create_joint_element(joint_name, joint_config, parent_link, child_link, hand_side))
    
    return robot


def prettify_xml(element: ET.Element) -> str:
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(element, 'unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def main():
    parser = argparse.ArgumentParser(description='Generate Inspire Hand URDF from configuration')
    parser.add_argument('--config', type=str, default='inspire_hand_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='inspire_hand_generated.urdf',
                       help='Output URDF file path')
    parser.add_argument('--hand-side', type=str, choices=['R_', 'L_'], default=None,
                       help='Override hand side from config (R_ for right, L_ for left)')
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Configuration file {args.config} not found!")
        return
    
    config = load_config(args.config)
    
    # Override hand side if specified
    if args.hand_side:
        config['hand_side'] = args.hand_side
    
    # Generate URDF
    robot_element = generate_urdf(config)
    
    # Convert to pretty XML string
    urdf_content = prettify_xml(robot_element)
    
    # Write to file
    with open(args.output, 'w') as f:
        f.write(urdf_content)
    
    print(f"Generated URDF saved to {args.output}")
    print(f"Hand side: {config['hand_side']}")
    print(f"Robot name: {config['robot_name']}")


if __name__ == '__main__':
    main() 