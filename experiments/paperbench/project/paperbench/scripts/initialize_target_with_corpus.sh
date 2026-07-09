#!/bin/bash

# Initialize target papers with guide.json AND collect related papers to KG
# Usage: ./initialize_target_with_corpus.sh <split_name> [max_parallel_jobs] [mode]
# Examples:
#   ./initialize_target_with_corpus.sh debug
#   ./initialize_target_with_corpus.sh dev 10 node-only
#   ./initialize_target_with_corpus.sh all 5 corpus-only
#
# Modes:
#   both        - Generate guide.json + collect corpus + generate nodes (default)
#   node-only   - Only generate guide.json for target papers
#   corpus-only - Only collect corpus (requires existing guide.json)

set -euo pipefail

# Parse arguments with better flexibility
# Usage:
#   ./script.sh debug                          (split only, use defaults)
#   ./script.sh debug 10                       (split + parallel jobs)
#   ./script.sh debug corpus-only              (split + mode)
#   ./script.sh debug 5 node-only              (split + parallel + mode)

SPLIT_NAME="debug"
MAX_PARALLEL_JOBS=3
COLLECT_CORPUS="both"  # both, node-only, corpus-only

# Parse positional arguments
if [[ $# -gt 0 ]]; then
    SPLIT_NAME="$1"
fi

if [[ $# -gt 1 ]]; then
    arg2="$2"
    if [[ "$arg2" =~ ^[0-9]+$ ]]; then
        # Second arg is a number - treat as max parallel jobs
        MAX_PARALLEL_JOBS="$arg2"
        # Check if there's a third arg for mode
        if [[ $# -gt 2 ]]; then
            arg3="$3"
            if [[ "$arg3" =~ ^(both|node-only|corpus-only)$ ]]; then
                COLLECT_CORPUS="$arg3"
            else
                echo "❌ Invalid mode: $arg3"
                echo "Valid modes: both, node-only, corpus-only"
                exit 1
            fi
        fi
    elif [[ "$arg2" =~ ^(both|node-only|corpus-only)$ ]]; then
        # Second arg is a mode - treat as collect mode
        COLLECT_CORPUS="$arg2"
    else
        echo "❌ Unknown argument: $arg2"
        echo "Usage: $0 <split> [max_parallel_jobs] [mode]"
        echo "Modes: both, node-only, corpus-only"
        exit 1
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
XKG_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

# Paths
SPLITS_DIR="$PROJECT_ROOT/experiments/splits"
DATA_DIR="$PROJECT_ROOT/data/papers"
LOG_DIR="$PROJECT_ROOT/logs/initialize_target_${SPLIT_NAME}_$(date +%Y%m%d_%H%M%S)"

# Create directories
mkdir -p "$LOG_DIR"

# Load environment
ENV_FILE="$PROJECT_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "⚠ .env not found, using system environment variables"
fi

# Activate conda environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    if conda info --envs | grep -q "xkg"; then
        echo "Activating xkg environment..."
        conda activate xkg || true
    fi
fi

# Validate split file
SPLIT_FILE="$SPLITS_DIR/${SPLIT_NAME}.txt"
if [ ! -f "$SPLIT_FILE" ]; then
    echo "❌ Split file not found: $SPLIT_FILE"
    echo "Available splits: $(ls "$SPLITS_DIR"/*.txt 2>/dev/null | xargs -I{} basename {} .txt | tr '\n' ' ')"
    exit 1
fi

echo "======================================================================"
echo "xKG Target Paper Initialization"
echo "Split: $SPLIT_NAME"
echo "Max parallel jobs: $MAX_PARALLEL_JOBS"
echo "Mode: $COLLECT_CORPUS"
echo "Log directory: $LOG_DIR"
echo "KG output: $XKG_ROOT/xKG/storage/kg"
echo "======================================================================"

# Extract paper title from paper.json (handles both "title" and "paper_title" fields)
get_paper_title() {
    local paper_dir="$1"
    local paper_json="$paper_dir/paper.json"
    if [ ! -f "$paper_json" ]; then
        echo ""
        return
    fi
    python3 -c "
import json
with open('$paper_json', 'r') as f:
    d = json.load(f)
title = d.get('paper_title') or d.get('title') or ''
print(title)
"
}

# Process each paper
process_paper() {
    local paper_id="$1"
    local paper_dir="$DATA_DIR/$paper_id"
    local log_file="$LOG_DIR/${paper_id}.log"

    if [ ! -d "$paper_dir" ]; then
        echo "❌ Paper directory not found: $paper_dir" | tee "$log_file"
        return 1
    fi

    echo "Processing paper: $paper_id" | tee "$log_file"

    # Extract real paper title from paper.json
    local paper_title
    paper_title=$(get_paper_title "$paper_dir")
    if [ -z "$paper_title" ]; then
        echo "❌ Cannot extract paper title from $paper_dir/paper.json" | tee -a "$log_file"
        return 1
    fi
    echo "  Title: $paper_title" >> "$log_file"

    # Step 1: Generate guide.json (paper-only, no code fetch)
    if [ "$COLLECT_CORPUS" != "corpus-only" ]; then
        echo "  [1/2] Generating guide.json..." >> "$log_file"

        # Use python -m so module resolution works from any directory
        python -m xKG.scripts.generate_node \
            --title "$paper_title" \
            --output-dir "$paper_dir" \
            --output-name "guide.json" \
            --fetch-code false \
            >> "$log_file" 2>&1

        if [ $? -ne 0 ]; then
            echo "✗ Failed to generate guide.json for $paper_id" | tee -a "$log_file"
            return 1
        fi
        echo "✓ guide.json generated" >> "$log_file"
    fi

    # Step 2: Collect related papers and generate nodes to KG
    if [ "$COLLECT_CORPUS" != "node-only" ]; then
        echo "  [2/2] Collecting related papers and generating nodes..." >> "$log_file"

        # Check if guide.json exists (either generated above or pre-existing)
        guide_json="$paper_dir/guide.json"
        if [ ! -f "$guide_json" ]; then
            echo "⚠ guide.json not found, skipping corpus collection" >> "$log_file"
            return 0
        fi

        # --generate-node true: also generate KG nodes for collected papers
        # --fetch-code false: don't download code repos (paper-only mode)
        # Output goes to config kg_path (default: xKG/storage/kg)
        python -m xKG.scripts.collect_corpus \
            --node "$guide_json" \
            --max-papers 10 \
            --generate-node true \
            >> "$log_file" 2>&1

        if [ $? -ne 0 ]; then
            echo "⚠ Failed to collect corpus for $paper_id (non-critical)" >> "$log_file"
            # Don't return 1 - this is non-blocking
        else
            echo "✓ Corpus collected and nodes generated" >> "$log_file"
        fi
    fi

    echo "✓ Completed: $paper_id" | tee -a "$log_file"
    return 0
}

# Export function for parallel execution
export -f process_paper get_paper_title
export SCRIPT_DIR LOG_DIR DATA_DIR XKG_ROOT COLLECT_CORPUS

echo ""
echo "Starting paper initialization..."
echo ""

failed_count=0
success_count=0

# Read split file and process each paper
while IFS= read -r paper_id || [ -n "$paper_id" ]; do
    # Skip empty lines and comments
    [[ -z "$paper_id" || "$paper_id" =~ ^# ]] && continue

    # Control parallel jobs
    while (( $(jobs -p | wc -l) >= MAX_PARALLEL_JOBS )); do
        sleep 1
    done

    # Run in background
    process_paper "$paper_id" &

done < "$SPLIT_FILE"

# Wait for all background jobs
wait

# Count results
for log_file in "$LOG_DIR"/*.log; do
    [ -f "$log_file" ] || continue
    if grep -q "Completed:" "$log_file"; then
        ((success_count++)) || true
    else
        ((failed_count++)) || true
    fi
done

# Summary
echo ""
echo "======================================================================"
echo "✅ Initialization Complete"
echo "  Success: $success_count"
echo "  Failed: $failed_count"
echo "  KG Path: $XKG_ROOT/xKG/storage/kg"
echo "======================================================================"

# Exit with error if any failed
[ $failed_count -eq 0 ] && exit 0 || exit 1
