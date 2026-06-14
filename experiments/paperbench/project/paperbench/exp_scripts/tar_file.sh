#!/bin/bash

# extract_and_identify.sh

DEFAULT_TARGET_DIR="/disk/disk_20T/luoyujie/preparedness/project/temp"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <tarball_path> [target_dir]"
    exit 1
fi

TARBALL_PATH="$1"
TARGET_DIR="${2:-$DEFAULT_TARGET_DIR}"

# --- 提取 task name (e.g., mechanistic-understanding) ---
# Step 1: Get the directory containing the .tar.gz file
TAR_DIR=$(dirname "$TARBALL_PATH")

# Step 2: Get the parent directory name (which is like "mechanistic-understanding_xxxx-xxxx-...")
PARENT_DIR_NAME=$(basename "$TAR_DIR")

# Step 3: Extract prefix before first underscore
TASK_NAME=$(echo "$PARENT_DIR_NAME" | cut -d'_' -f1)

echo "Detected task name: $TASK_NAME"

# --- Proceed to extract ---
if [ ! -f "$TARBALL_PATH" ]; then
    echo "Error: Tarball not found: $TARBALL_PATH"
    exit 1
fi

mkdir -p "$TARGET_DIR"

echo "Extracting to: $TARGET_DIR"
tar -xzf "$TARBALL_PATH" -C "$TARGET_DIR"

if [ $? -eq 0 ]; then
    echo "✅ Extraction completed. Task: $TASK_NAME"
else
    echo "❌ Extraction failed."
    exit 1
fi
