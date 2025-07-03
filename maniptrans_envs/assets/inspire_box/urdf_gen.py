#!/usr/bin/env python3
"""
URDF Generator for Inspire Hand with Box Geometries
Replaces all STL mesh geometries with configurable box dimensions.
Keeps original sphere geometry for tip links.
"""

import xml.etree.ElementTree as ET
import os
import copy
import argparse

def create_box_geometry(size):
    """Create a box geometry element with given size."""
    geometry = ET.Element('geometry')
    box = ET.SubElement(geometry, 'box')
    box.set('size', size)
    return geometry

def replace_mesh_with_box(element, box_size):
    """Replace mesh geometry with box geometry in a visual or collision element."""
    geometry = element.find('geometry')
    if geometry is not None:
        mesh = geometry.find('mesh')
        if mesh is not None:
            # Remove the mesh element
            geometry.remove(mesh)
            # Add box element
            box = ET.SubElement(geometry, 'box')
            box.set('size', box_size)

def replace_sphere_with_box(element, box_size):
    """Replace sphere geometry with box geometry in a visual element."""
    geometry = element.find('geometry')
    if geometry is not None:
        sphere = geometry.find('sphere')
        if sphere is not None:
            # Keep the original sphere for tips
            pass

def parse_xyz(xyz_str):
    """Parse xyz string to list of floats."""
    if xyz_str is None:
        return [0, 0, 0]
    return [float(x) for x in xyz_str.split()]

def calculate_joint_distance(joint):
    """Calculate the distance from joint origin to child link."""
    origin = joint.find('origin')
    if origin is not None:
        xyz = parse_xyz(origin.get('xyz'))
        return (xyz[0]**2 + xyz[1]**2 + xyz[2]**2)**0.5
    return 0.0

def get_link_box_size(link_name, joints, box_config):
    """Calculate box size for a link based on joint distances."""
    if 'base_link' in link_name:
        return box_config['palm']
    
    # Find joints that connect to this link
    connected_joints = []
    for joint in joints:
        child_link = joint.find('child')
        if child_link is not None and child_link.get('link') == link_name:
            connected_joints.append(joint)
    
    if not connected_joints:
        return box_config['finger']
    
    # Calculate total length from joint distances
    total_length = 0.0
    for joint in connected_joints:
        total_length += calculate_joint_distance(joint)
    
    # Use calculated length or default
    if total_length > 0:
        # Box dimensions: length x width x height
        # Length is the joint distance, width and height are from config
        width, height = box_config['finger'].split()[1], box_config['finger'].split()[2]
        return f"{total_length:.4f} {width} {height}"
    else:
        return box_config['finger']

def generate_urdf_with_boxes(input_urdf_path, output_urdf_path, box_config):
    """
    Generate URDF with box geometries replacing all meshes and spheres.
    
    Args:
        input_urdf_path: Path to the original URDF file
        output_urdf_path: Path to save the generated URDF
        box_config: Dictionary with box sizes for different link types
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
        
        # Calculate box size based on joint distances
        box_size = get_link_box_size(link_name, joints, box_config)
        
        # Replace visual geometry
        visual = link.find('visual')
        if visual is not None:
            replace_mesh_with_box(visual, box_size)
            # Keep original sphere for tips
            if 'tip' not in link_name:
                replace_sphere_with_box(visual, box_size)
        
        # Replace collision geometry
        collision = link.find('collision')
        if collision is not None:
            replace_mesh_with_box(collision, box_size)
            # Keep original sphere for tips
            if 'tip' not in link_name:
                replace_sphere_with_box(collision, box_size)
    
    # Write the modified URDF
    tree.write(output_urdf_path, encoding='utf-8', xml_declaration=True)
    print(f"Generated URDF with box geometries: {output_urdf_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate URDF with box geometries')
    parser.add_argument('input_urdf', help='Input URDF file path')
    parser.add_argument('output_urdf', help='Output URDF file path')
    parser.add_argument('--palm-size', default='0.1 0.05 0.03', 
                       help='Box size for palm/base link (default: 0.1 0.05 0.03)')
    parser.add_argument('--finger-size', default='0.05 0.02 0.01', 
                       help='Box size for finger segments (default: 0.05 0.02 0.01)')
    parser.add_argument('--tip-size', default='0.01 0.01 0.01', 
                       help='Box size for finger tips (default: 0.01 0.01 0.01)')
    
    args = parser.parse_args()
    
    # Box configuration
    box_config = {
        'palm': args.palm_size,
        'finger': args.finger_size,
    }
    
    # Generate the URDF
    generate_urdf_with_boxes(args.input_urdf, args.output_urdf, box_config)

if __name__ == "__main__":
    # Example usage without command line arguments
    import sys
    if len(sys.argv) == 1:
        # Default configuration
        box_config = {
            'palm': '0.1 0.05 0.08',
            'finger': '0.01 0.05 0.01', 
        }
        
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        input_path = os.path.join(curr_dir, "inspire_hand_left.urdf")
        output_path = os.path.join(curr_dir, "inspire_hand_left_box.urdf")
        
        generate_urdf_with_boxes(input_path, output_path, box_config)
    else:
        main() 