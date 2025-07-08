#!/usr/bin/env python3
"""
Script to generate URDF files for each COACD-processed object
"""

import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

def create_urdf_from_template(object_name, mesh_filename, template_path, output_path):
    """
    Create a URDF file for an object based on the template
    """
    try:
        # Read the template file and clean it
        with open(template_path, 'r') as f:
            template_content = f.read().strip()
        
        # Parse the cleaned template
        root = ET.fromstring(template_content)
        
        # Update the robot name
        root.set('name', object_name)
        
        # Find all mesh elements and update the filename
        for mesh in root.findall('.//mesh'):
            mesh.set('filename', mesh_filename)
        
        # Create the output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the URDF file with proper formatting
        xml_str = ET.tostring(root, encoding='unicode')
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ")
        
        # Remove extra blank lines
        lines = pretty_xml.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip() or line.startswith('<?xml'):
                cleaned_lines.append(line)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(cleaned_lines))
        
        print(f"  ✓ Generated URDF: {output_path}")
        
    except ET.ParseError as e:
        print(f"  ✗ XML parsing error: {e}")
        # Create a simple URDF as fallback
        create_simple_urdf(object_name, mesh_filename, output_path)
    except Exception as e:
        print(f"  ✗ Error: {e}")
        # Create a simple URDF as fallback
        create_simple_urdf(object_name, mesh_filename, output_path)

def create_simple_urdf(object_name, mesh_filename, output_path):
    """
    Create a simple URDF file as fallback
    """
    urdf_content = f'''<?xml version="1.0"?>
<robot name="{object_name}">
  <material name="obj_color">
    <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <link name="base">
    <visual>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{mesh_filename}" scale="1 1 1"/>
      </geometry>
      <material name="obj_color"/>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{mesh_filename}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>'''
    
    # Create the output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(urdf_content)
    
    print(f"  ✓ Generated simple URDF: {output_path}")

def generate_urdfs_for_all_objects():
    """
    Generate URDF files for all COACD-processed objects
    """
    # Define paths
    base_dir = Path(__file__).parent
    coacd_dir = base_dir / "coacd_object_preview" / "align_ds"
    urdf_output_dir = base_dir / "coacd_object_preview" / "align_ds"
    template_path = base_dir / "obj_urdf_example.urdf"
    
    # Check if template exists
    if not template_path.exists():
        print(f"Warning: Template file not found at {template_path}")
        print("Will use built-in simple template instead.")
        template_path = None
    
    # Create output directory
    urdf_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all COACD-processed directories
    coacd_dirs = [d for d in coacd_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(coacd_dirs)} COACD-processed object directories")
    
    # Process each directory
    for i, obj_dir in enumerate(coacd_dirs, 1):
        obj_name = obj_dir.name
        
        # Look for the processed mesh file (prefer OBJ, fallback to PLY)
        mesh_files = list(obj_dir.glob("*.obj"))
        if not mesh_files:
            mesh_files = list(obj_dir.glob("*.ply"))
        
        if not mesh_files:
            print(f"[{i}/{len(coacd_dirs)}] No mesh files found in {obj_name}")
            continue
        
        # Use the first mesh file found
        mesh_file = mesh_files[0]
        
        print(f"[{i}/{len(coacd_dirs)}] Processing: {obj_name}")
        
        # Create output path for URDF
        urdf_output_path = urdf_output_dir / obj_name / "scan.urdf"
        
        try:
            if template_path and template_path.exists():
                # Create URDF file using template
                create_urdf_from_template(
                    object_name=obj_name,
                    mesh_filename=str(mesh_file.name),  # Just the filename, not full path
                    template_path=template_path,
                    output_path=urdf_output_path
                )
            else:
                # Create simple URDF file
                create_simple_urdf(
                    object_name=obj_name,
                    mesh_filename=str(mesh_file.name),
                    output_path=urdf_output_path
                )
            
        except Exception as e:
            print(f"  ✗ Failed to generate URDF for {obj_name}")
            print(f"    Error: {e}")
    
    print(f"\nURDF generation complete! Generated URDFs for {len(coacd_dirs)} objects.")
    print(f"URDF files saved to: {urdf_output_dir}")

if __name__ == "__main__":
    generate_urdfs_for_all_objects() 