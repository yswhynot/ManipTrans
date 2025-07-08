#!/usr/bin/env python3
"""
Script to list all available data indices from the annotation files
"""

import os
from pathlib import Path

def list_available_indices():
    """
    List all available data indices from the annotation files
    """
    # Define paths
    base_dir = Path(__file__).parent
    anno_dir = base_dir / "anno_preview"
    
    if not anno_dir.exists():
        print(f"Annotation directory not found: {anno_dir}")
        return
    
    # Get all annotation files
    anno_files = list(anno_dir.glob("*.pkl"))
    anno_files.sort()
    
    print(f"Found {len(anno_files)} annotation files")
    print("\nAvailable data indices (first 5 digits of hash):")
    print("=" * 50)
    
    # Extract and display the 5-digit hashes
    for i, anno_file in enumerate(anno_files, 1):
        filename = anno_file.name
        # Extract the hash part (between ++seq__ and __)
        if "++seq__" in filename and "__" in filename.split("++seq__")[1]:
            hash_part = filename.split("++seq__")[1].split("__")[0]
            five_digit_hash = hash_part[:5]
            print(f"{i:3d}. {five_digit_hash} -> {filename}")
        else:
            print(f"{i:3d}. [ERROR] Could not parse: {filename}")
    
    print("\nTo use a specific data index, run:")
    print("python main/dataset/mano2dexhand.py --data_idx <five_digit_hash>@<stage> --side right --dexhand inspire --headless --iter 7000")
    print("\nExample:")
    print("python main/dataset/mano2dexhand.py --data_idx 667a8@0 --side right --dexhand inspire --headless --iter 7000")

if __name__ == "__main__":
    list_available_indices() 