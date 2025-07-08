#!/bin/bash

# Script to process all objects in the align_ds directory using coacd_process.py

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Debug: print the paths
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Current working directory: $(pwd)"

# Define paths
INPUT_DIR="$SCRIPT_DIR/object_preview/align_ds"
OUTPUT_DIR="$SCRIPT_DIR/coacd_object_preview/align_ds"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Find all .ply files in the input directory
echo "Finding .ply files..."
PLY_FILES=($(find "$INPUT_DIR" -name "*.ply" -type f))
TOTAL_FILES=${#PLY_FILES[@]}

echo "Found $TOTAL_FILES .ply files to process"

# Process each file
for i in "${!PLY_FILES[@]}"; do
    PLY_FILE="${PLY_FILES[$i]}"
    
    # Get relative path from input_dir
    REL_PATH="${PLY_FILE#$INPUT_DIR/}"
    
    # Create corresponding output path
    OUTPUT_FILE="$OUTPUT_DIR/$REL_PATH"
    
    # Create output directory if it doesn't exist
    OUTPUT_FILE_DIR="$(dirname "$OUTPUT_FILE")"
    mkdir -p "$OUTPUT_FILE_DIR"
    
    echo "[$((i+1))/$TOTAL_FILES] Processing: $REL_PATH"
    
    # Run the coacd_process command
    cd "$PROJECT_ROOT"
    if python maniptrans_envs/lib/utils/coacd_process.py \
        -i "$PLY_FILE" \
        -o "$OUTPUT_FILE" \
        --max-convex-hull 32 \
        --seed 1 \
        -mi 2000 \
        -md 5 \
        -t 0.07; then
        echo "  ✓ Successfully processed: $REL_PATH"
    else
        echo "  ✗ Failed to process: $REL_PATH"
    fi
done

echo ""
echo "Processing complete! Processed $TOTAL_FILES files." 