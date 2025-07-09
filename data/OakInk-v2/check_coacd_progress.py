#!/usr/bin/env python3

import os
import glob
from pathlib import Path
from collections import defaultdict

def find_ply_files(directory):
    """Find all .ply files in directory and subdirectories"""
    ply_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.ply'):
                ply_files.append(os.path.join(root, file))
    return ply_files

def check_coacd_progress(input_dir, output_dir):
    """Check progress of COACD conversion"""
    
    print(f"Checking COACD conversion progress...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Find all input .ply files
    input_files = find_ply_files(input_dir)
    print(f"Found {len(input_files)} input .ply files")
    
    # Find all output .ply files
    output_files = find_ply_files(output_dir)
    print(f"Found {len(output_files)} output .ply files")
    
    # Find .done and .failed files
    done_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.done'):
                done_files.append(os.path.join(root, file))
    
    print(f"Found {len(done_files)} .done files")
    
    # Calculate progress
    total_input = len(input_files)
    total_processed = len(done_files)
    total_successful = len(done_files)
    
    if total_input > 0:
        progress_percent = (total_processed / total_input) * 100
        success_rate = (total_successful / total_processed) * 100 if total_processed > 0 else 0
        
        print(f"\nProgress Summary:")
        print(f"  Total input files: {total_input}")
        print(f"  Total processed: {total_processed} ({progress_percent:.1f}%)")
        print(f"  Successful: {total_successful}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Remaining: {total_input - total_processed}")
        
        # Show progress bar
        print(f"\nProgress: [{'█' * int(progress_percent/5)}{'░' * (20 - int(progress_percent/5))}] {progress_percent:.1f}%")
        
    else:
        print("No input files found!")
    
    # Show some examples of failed files if any
    if failed_files:
        print(f"\nExamples of failed files (showing first 5):")
        for failed_file in failed_files[:5]:
            # Convert .failed to original input path for reference
            failed_file = failed_file.replace('.failed', '')
            print(f"  {failed_file}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")
    
    return {
        'total_input': total_input,
        'total_processed': total_processed,
        'total_successful': total_successful,
        'total_failed': total_failed,
        'progress_percent': progress_percent if total_input > 0 else 0,
        'success_rate': success_rate if total_processed > 0 else 0
    }

if __name__ == "__main__":
    # Default paths based on the command pattern
    input_dir = "object_preview/align_ds"
    output_dir = "coacd_object_preview/align_ds"
    
    # Check if directories exist
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist!")
        exit(1)
    
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist!")
        exit(1)
    
    # Check progress
    stats = check_coacd_progress(input_dir, output_dir)
    
    # Exit with appropriate code
    if stats['total_input'] == 0:
        exit(1)
    elif stats['progress_percent'] == 100:
        print("\n✅ Conversion completed!")
        exit(0)
    else:
        print(f"\n⏳ Conversion in progress... {stats['progress_percent']:.1f}% complete")
        exit(0) 