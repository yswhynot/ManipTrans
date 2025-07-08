#!/usr/bin/env python3
"""
Script to convert PLY files to OBJ format for URDF compatibility
"""

import os
import sys
from pathlib import Path
import trimesh

def convert_ply_to_obj():
    """
    Convert all PLY files in coacd_object_preview to OBJ format
    """
    # Define paths
    base_dir = Path(__file__).parent
    coacd_dir = base_dir / "coacd_object_preview" / "align_ds"
    
    # Find all PLY files
    ply_files = list(coacd_dir.rglob("*.ply"))
    
    print(f"Found {len(ply_files)} PLY files to convert")
    
    # Convert each file
    for i, ply_file in enumerate(ply_files, 1):
        # Create OBJ file path (same directory, different extension)
        obj_file = ply_file.with_suffix('.obj')
        
        print(f"[{i}/{len(ply_files)}] Converting: {ply_file.name}")
        
        try:
            # Load PLY file
            mesh = trimesh.load(str(ply_file))
            
            # Export as OBJ
            mesh.export(str(obj_file))
            
            print(f"  ✓ Converted: {obj_file.name}")
            
        except Exception as e:
            print(f"  ✗ Failed to convert {ply_file.name}: {e}")
    
    print(f"\nConversion complete! Converted {len(ply_files)} files.")

if __name__ == "__main__":
    convert_ply_to_obj() 