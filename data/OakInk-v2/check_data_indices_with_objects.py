#!/usr/bin/env python3
"""
Script to check which objects each data index uses and verify if mesh files exist
"""

import os
import pickle
import json
from pathlib import Path

def check_data_indices_with_objects():
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
            
            print(f"{i:3d}. {five_digit_hash} -> {filename}")
            print(f"     Objects: {object_list}")
            
            # Check mesh files for each object
            mesh_status = []
            for obj in object_list:
                obj_mesh_path = object_dir / obj / "scan.obj"
                obj_ply_path = object_dir / obj / "scan.ply"
                coacd_obj_path = coacd_dir / obj / "scan.obj"
                coacd_ply_path = coacd_dir / obj / "scan.ply"
                urdf_path = urdf_dir / obj / "scan.urdf"
                
                status = {
                    'obj': obj,
                    'original_obj': obj_mesh_path.exists(),
                    'original_ply': obj_ply_path.exists(),
                    'coacd_obj': coacd_obj_path.exists(),
                    'coacd_ply': coacd_ply_path.exists(),
                    'urdf': urdf_path.exists()
                }
                mesh_status.append(status)
                
                print(f"       {obj}:")
                print(f"         Original OBJ: {'✓' if status['original_obj'] else '✗'}")
                print(f"         Original PLY: {'✓' if status['original_ply'] else '✗'}")
                print(f"         COACD OBJ: {'✓' if status['coacd_obj'] else '✗'}")
                print(f"         COACD PLY: {'✓' if status['coacd_ply'] else '✗'}")
                print(f"         URDF: {'✓' if status['urdf'] else '✗'}")
            
            # Check if all required files exist
            all_files_exist = all(
                status['coacd_obj'] and status['urdf'] 
                for status in mesh_status
            )
            
            if all_files_exist:
                valid_indices.append(five_digit_hash)
                print(f"     Status: ✓ READY (all files exist)")
            else:
                print(f"     Status: ✗ INCOMPLETE (missing files)")
            
            # Show stages info
            if stages_info:
                print(f"     Stages: {len(stages_info)} stages available")
                for stage_idx, (stage_range, stage_info) in enumerate(stages_info.items()):
                    stage_objects = stage_info.get("obj_list_rh", [])
                    print(f"       Stage {stage_idx}: {stage_objects}")
            
            print()
            
        except Exception as e:
            print(f"{i:3d}. {five_digit_hash} -> {filename}")
            print(f"     [ERROR] Could not load: {e}")
            print()
    
    print("=" * 80)
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
    check_data_indices_with_objects() 