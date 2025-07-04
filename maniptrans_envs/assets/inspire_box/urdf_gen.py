#!/usr/bin/env python3
"""
URDF Generator for Inspire Hand with Box Geometries
Replaces all STL mesh geometries with configurable box dimensions.
Keeps original sphere geometry for tip links.
"""

import xml.etree.ElementTree as ET
import os
import copy
import numpy as np
import argparse
from scipy.spatial.transform import Rotation

def create_box_geometry(size):
    """Create a box geometry element with given size."""
    geometry = ET.Element('geometry')
    box = ET.SubElement(geometry, 'box')
    box.set('size', size)
    return geometry

def rotation_matrix_from_rpy(roll, pitch, yaw):
    r = Rotation.from_euler('xyz', [roll, pitch, yaw])
    return r.as_matrix() 

def get_translate_then_rotate_origin(translation, rotation_rpy):
    """Convert translate-then-rotate to URDF's rotate-then-translate format."""
    # Convert rpy to rotation matrix
    roll, pitch, yaw = rotation_rpy
    R = rotation_matrix_from_rpy(roll, pitch, yaw)
    
    # Calculate the equivalent translation
    t_prime = R @ np.array(translation)
    
    return t_prime.tolist(), rotation_rpy

def replace_mesh_with_box(element, box_size, offset=None, rpy=None):
    """Replace mesh geometry with box geometry in a visual or collision element."""
    geometry = element.find('geometry')
    if geometry is not None:
        mesh = geometry.find('mesh')
        if mesh is not None:
            # Remove the mesh element
            geometry.remove(mesh)
            # Add box element
            box = ET.SubElement(geometry, 'box')
            # Convert list to space-separated string
            if isinstance(box_size, list):
                box.set('size', ' '.join(map(str, box_size)))
            else:
                box.set('size', box_size)
    

    if rpy is not None and rpy != [0, 0, 0]:
        origin = element.find('origin')
        if origin is None:
            origin = ET.SubElement(element, 'origin')
            origin.set('xyz', '0 0 0')
            origin.set('rpy', '0 0 0')
        origin.set('rpy', ' '.join(map(lambda x: f'{x:.4f}', rpy)))

        offset, rpy = get_translate_then_rotate_origin(offset, rpy)

    # Update origin if offset is provided
    if offset is not None and offset != [0, 0, 0]:
        origin = element.find('origin')
        if origin is None:
            origin = ET.SubElement(element, 'origin')
            origin.set('xyz', '0 0 0')
            origin.set('rpy', '0 0 0')
        
        origin.set('xyz', ' '.join(map(lambda x: f'{x:.4f}', offset)))

def replace_sphere_with_box(element, box_size):
    """Replace sphere geometry with box geometry in a visual element."""
    geometry = element.find('geometry')
    if geometry is not None:
        sphere = geometry.find('sphere')
        if sphere is not None:
            # change sphere size to match box size
            sphere.set('radius', str(1.2 * box_size[0] / 2))

def parse_xyz(xyz_str):
    """Parse xyz string to list of floats."""
    if xyz_str is None:
        return [0, 0, 0]
    return [float(x) for x in xyz_str.split()]

def calculate_joint_distance(joint):
    """Calculate the distance from joint origin to child link."""
    origin = joint.find('origin')
    if origin is not None:
        xyz = np.array(parse_xyz(origin.get('xyz')))
        return np.linalg.norm(xyz)
    return 0.0

def get_link_box_size_and_offset(link_name, joints, box_config):
    """Calculate box size and origin offset for a link based on joint distances."""
    if 'base_link' in link_name:
        offset = (-np.array(box_config['palm'])/2)
        offset[2] = 0
        return box_config['palm'], offset.tolist()
    
    # Find joints that connect to this link
    connected_joints = []
    for joint in joints:
        parent_link = joint.find('parent')
        if parent_link is not None and parent_link.get('link') == link_name:
            connected_joints.append(joint)
    
    if not connected_joints:
        return box_config['finger'], [0, 0, 0]
    
    # Calculate total length and offset from joint distances
    total_length = 0.0
    offset = [0, 0, 0]
    
    for joint in connected_joints:
        origin = joint.find('origin')
        if origin is not None:
            xyz = parse_xyz(origin.get('xyz'))
            distance = np.linalg.norm(xyz)
            total_length += distance
            
            # Calculate offset as half the joint distance
            # offset = (direction * distance / 2).tolist()
            offset = - np.array(box_config['finger'])/2
            offset[1] = -distance / 2

            offset = offset.tolist()
    
    # Use calculated length or default
    if total_length > 0:
        # Box dimensions: width x length x height
        # Length is the joint distance, width and height are from config
        width, height = box_config['finger'][0], box_config['finger'][2]
        return [width, total_length, height], offset
    else:
        return box_config['finger'], [0, 0, 0]

def get_link_rpy(link_name, joints):
    """Get the orientation of a link based on its joint direction."""
    for joint in joints:
        parent_link = joint.find('parent')
        if parent_link is not None and parent_link.get('link') == link_name:
            xyz = np.array(parse_xyz(joint.find('origin').get('xyz')))
            if np.linalg.norm(xyz) == 0:
                return [0, 0, 0]
            
            # Normalize the direction vector
            direction = xyz / np.linalg.norm(xyz)
            
            # Calculate yaw (rotation around Z-axis)
            yaw = np.arctan2(direction[1], direction[0]) + np.pi/2
            
            # Calculate pitch (rotation around Y-axis)
            pitch = np.arcsin(-direction[2])
            
            # Calculate roll (rotation around X-axis) - usually 0 for finger joints
            roll = 0.0
            
            return [roll, pitch, yaw]
    return [0, 0, 0]

def generate_urdf_with_boxes(input_urdf_path, output_urdf_path, box_config):
    """
    Generate URDF with box geometries replacing all meshes and spheres.
    
    Args:
        input_urdf_path: Path to the original URDF file
        output_urdf_path: Path to save the generated URDF
        box_config: list of box sizes for different link types
    """
    
    # Parse the original URDF
    tree = ET.parse(input_urdf_path)
    root = tree.getroot()
    
    # Update robot name
    root.set('name', root.get('name', 'h1') + '_box')
    
    # Remove mujoco section if present
    mujoco = root.find('mujoco')
    if mujoco is not None:
        root.remove(mujoco)
    
    # Get all joints for distance calculations
    joints = root.findall('joint')
    
    # Process each link
    for link in root.findall('link'):
        link_name = link.get('name', '')
        
        # Calculate box size and offset based on joint distances
        box_size, offset = get_link_box_size_and_offset(link_name, joints, box_config)
        rpy = [0, 0, 0]
        if 'thumb' in link_name:
            rpy = get_link_rpy(link_name, joints)
        
        # Replace visual geometry
        visual = link.find('visual')
        if visual is not None:
            replace_mesh_with_box(visual, box_size, offset, rpy)
            # Keep original sphere for tips
            if 'tip' in link_name:
                replace_sphere_with_box(visual, box_size)
            else:
                replace_mesh_with_box(visual, box_size)
        
        # Replace collision geometry
        collision = link.find('collision')
        if collision is not None:
            replace_mesh_with_box(collision, box_size, offset, rpy)
            # Keep original sphere for tips
            if 'tip' in link_name:
                replace_sphere_with_box(collision, box_size)
            else:
                replace_mesh_with_box(collision, box_size)

    # replace joint origin
    for joint in joints:
        joint_name = joint.get('name', '')
        if ('intermediate' in joint_name) and ('thumb' not in joint_name):
            origin = joint.find('origin')
            if origin is not None:
                xyz = parse_xyz(origin.get('xyz'))
                origin.set('xyz', f'0 {xyz[1]} 0')
        elif 'tip' in joint_name:
            origin = joint.find('origin')
            if origin is not None:
                xyz = parse_xyz(origin.get('xyz'))
                origin.set('xyz', f'{-box_config["finger"][0] / 2} {xyz[1]} {-box_config["finger"][2] / 2}')
    
    # Write the modified URDF
    tree.write(output_urdf_path, encoding='utf-8', xml_declaration=True)
    print(f"Generated URDF with box geometries: {output_urdf_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate URDF with box geometries')
    parser.add_argument('input_urdf', help='Input URDF file path')
    parser.add_argument('output_urdf', help='Output URDF file path')
    parser.add_argument('--palm-size', default='0.1 0.05 0.08', 
                       help='Box size for palm/base link (default: 0.1 0.05 0.08)')
    parser.add_argument('--finger-size', default='0.01 0.05 0.01', 
                       help='Box size for finger segments (default: 0.01 0.05 0.01)')
    
    args = parser.parse_args()
    
    # Box configuration - convert string arguments to lists
    box_config = {
        'palm': [float(x) for x in args.palm_size.split()],
        'finger': [float(x) for x in args.finger_size.split()]
    }
    
    # Generate the URDF
    generate_urdf_with_boxes(args.input_urdf, args.output_urdf, box_config)

if __name__ == "__main__":
    # Example usage without command line arguments
    import sys
    if len(sys.argv) == 1:
        # Default configuration
        box_config = {
            'palm': [0.01, 0.1, 0.05],
            'finger': [0.01, 0.05, 0.01],
        }
        
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        input_path = os.path.join(curr_dir, "inspire_hand_left.urdf")
        output_path = os.path.join(curr_dir, "inspire_hand_left_box.urdf")
        
        generate_urdf_with_boxes(input_path, output_path, box_config)
    else:
        main()
