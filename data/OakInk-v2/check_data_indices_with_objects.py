#!/usr/bin/env python3
"""
Script to check which objects each data index uses and verify if mesh files exist
"""

import os
import pickle
import json
import argparse
from pathlib import Path

def check_data_indices_with_objects(show_only_completed=False):
    """
    Check which objects each data index uses and verify mesh file existence
    """
    # Define paths
    base_dir = Path(__file__).parent
    anno_dir = base_dir / "anno_preview"
    program_dir = base_dir / "program" / "program_info"
    object_dir = base_dir / "object_preview" / "align_ds"
    coacd_dir = base_dir / "coacd_object_preview" / "align_ds"
    urdf_dir = base_dir / "coacd_object_preview" / "align_ds"
    
    if not anno_dir.exists():
        print(f"Annotation directory not found: {anno_dir}")
        return
    
    # Get all annotation files
    anno_files = list(anno_dir.glob("*.pkl"))
    anno_files.sort()
    
    print(f"Found {len(anno_files)} annotation files")
    print("\nChecking data indices with their objects:")
    print("=" * 80)
    
    valid_indices = []
    
    # Process each annotation file
    for i, anno_file in enumerate(anno_files, 1):
        filename = anno_file.name
        base_filename = filename.replace(".pkl", "")
        
        # Extract the hash part
        if "++seq__" in filename and "__" in filename.split("++seq__")[1]:
            hash_part = filename.split("++seq__")[1].split("__")[0]
            five_digit_hash = hash_part[:5]
        else:
            print(f"{i:3d}. [ERROR] Could not parse: {filename}")
            continue
        
        try:
            # Load annotation data
            with open(anno_file, 'rb') as f:
                anno_data = pickle.load(f)
            
            # Get object list
            object_list = anno_data.get("obj_list", [])
            
            # Check program info
            program_file = program_dir / f"{base_filename}.json"
            stages_info = {}
            if program_file.exists():
                with open(program_file, 'r') as f:
                    program_info = json.load(f)
                    for k, v in program_info.items():
                        seg_pair_def = eval(k)
                        stages_info[seg_pair_def] = v
            
            # Check mesh files for each object (silently)
            mesh_status = []
            for obj in object_list:
                # Check for any mesh files in the object directory
                obj_dir = object_dir / obj
                coacd_obj_dir = coacd_dir / obj
                urdf_dir_obj = urdf_dir / obj
                
                # Find original PLY files
                original_ply_files = list(obj_dir.glob("*.ply"))
                original_ply_exists = len(original_ply_files) > 0
                
                # Find COACD OBJ files
                coacd_obj_files = list(coacd_obj_dir.glob("*.obj"))
                coacd_obj_exists = len(coacd_obj_files) > 0
                
                # Find COACD PLY files
                coacd_ply_files = list(coacd_obj_dir.glob("*.ply"))
                coacd_ply_exists = len(coacd_ply_files) > 0
                
                # Find URDF files
                urdf_files = list(urdf_dir_obj.glob("*.urdf"))
                urdf_exists = len(urdf_files) > 0
                
                status = {
                    'obj': obj,
                    'original_ply': original_ply_exists,
                    'coacd_obj': coacd_obj_exists,
                    'coacd_ply': coacd_ply_exists,
                    'urdf': urdf_exists
                }
                mesh_status.append(status)
            
            # Check if ALL objects have required files
            all_objects_ready = all(
                status['coacd_obj'] and status['urdf'] 
                for status in mesh_status
            )
            
            if all_objects_ready:
                valid_indices.append(five_digit_hash)
                print(f"{i:3d}. {five_digit_hash} -> {filename}")
                print(f"     Objects: {object_list}")
                
                # Show details for completed sequences
                for obj in object_list:
                    obj_status = next(s for s in mesh_status if s['obj'] == obj)
                    print(f"       {obj}: ✓ READY")
                
                # Show stages info
                if stages_info:
                    print(f"     Stages: {len(stages_info)} stages available")
                    for stage_idx, (stage_range, stage_info) in enumerate(stages_info.items()):
                        stage_objects = stage_info.get("obj_list_rh", [])
                        print(f"       Stage {stage_idx}: {stage_objects}")
                
                print()
            else:
                # Show simple line for incomplete sequences
                print(f"  {five_digit_hash}: incomplete")
            
        except Exception as e:
            print(f"\n{i:3d}. {five_digit_hash} -> {filename}")
            print(f"     [ERROR] Could not load: {e}")
            print()
    
    print("\n" + "=" * 80)
    print(f"Summary: {len(valid_indices)} indices are ready for use")
    print("\nReady indices:")
    for idx in valid_indices:
        print(f"  {idx}")
    
    print(f"\nTo use a ready index, run:")
    print(f"python main/dataset/mano2dexhand.py --data_idx <hash>@<stage> --side right --dexhand inspire --headless --iter 7000")
    print(f"\nExample:")
    if valid_indices:
        print(f"python main/dataset/mano2dexhand.py --data_idx {valid_indices[0]}@0 --side right --dexhand inspire --headless --iter 7000")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check data indices with their objects and file status")
    parser.add_argument("--completed-only", action="store_true", 
                       help="Show only completed indices (all files exist)")
    args = parser.parse_args()
    
    check_data_indices_with_objects(show_only_completed=args.completed_only) 