#!/bin/bash

# Script functionality:
# 1. Read the number of runs per paper from command line.
# 2. Read the 'split.txt' file in the same directory to get the paper ID list.
# 3. For each paper in the list, execute the reproduction process in parallel (this version uses guide.json).
# 4. Use a task pool to control concurrency and prevent system overload.
# 5. Redirect each task's output to an independent log file for debugging.

# --- Safety settings ---
# -e: Exit immediately if any command fails
# -u: Exit if an undefined variable is used
# -o pipefail: Return failure if any command in a pipeline fails
set -euo pipefail

# --- Configuration area ---

# Agent ID
AGENT_ID="paper2code"

# Set the maximum number of parallel jobs. Adjust based on your CPU cores, memory, and API rate limits.
MAX_PARALLEL_JOBS=5

# Read API configuration from .env file
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Error: .env file not found: $ENV_FILE"
    exit 1
fi

# Model configuration
GPT_VERSION="${1:-o3-mini}"
EVAL_MODEL="${2:-o3-mini}"

# Fixed number of runs per paper
NUM_RUNS_PER_PAPER=1
# Get the script's directory (for relative path calculation)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Split file name (passed as parameter, default to 'debug')
SPLIT_NAME="${3:-debug}"
SPLIT_FILE="${SCRIPT_DIR}/../../experiments/splits/${SPLIT_NAME}.txt"
RUN_TIMESTAMP=$(date -u +%Y-%m-%dT%H-%M-%S-UTC)
RUNS_DIR="${SCRIPT_DIR}/../../runs/${RUN_TIMESTAMP}_${AGENT_ID}"

LOG_DIR="${RUNS_DIR}/logs" # Log file directory

if [ ! -f "$SPLIT_FILE" ]; then
    echo "Error: Split file not found at: $SPLIT_FILE"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"
echo "All task logs will be saved to '${LOG_DIR}' directory."

# --- Core processing function ---
# Encapsulate all logic for processing a single paper into this function
process_paper() {
    local PAPER_ID=$1
    local run_num=$2

    # --- Set variables for this run ---
    local RUN_ID=$(date +%Y%m%d_%H%M%S) # Use YYYYMMDD_HHMMSS as unique ID
    local REPO_INDEX="${PAPER_ID}-${RUN_ID}"

    echo "========================================================================"
    echo ">> Starting run: ${run_num}/${NUM_RUNS_PER_PAPER}"
    echo ">> Paper ID    : ${PAPER_ID}"
    echo ">> Run ID      : ${RUN_ID}"
    echo ">> Repo index  : ${REPO_INDEX}"
    echo "========================================================================"

    # --- Define paths ---
    local BASE_DATA_DIR="data/papers"
    local BASE_RUN_DIR="${RUNS_DIR}"

    local PDF_JSON_PATH="${BASE_DATA_DIR}/${PAPER_ID}/paper.json"
    local PDF_JSON_CLEANED_PATH="${BASE_DATA_DIR}/${PAPER_ID}/paper_cleaned.json"
    local GUIDE_PATH="${BASE_DATA_DIR}/${PAPER_ID}/guide.json" # Guide path
    local OUTPUT_DIR="${BASE_RUN_DIR}/${PAPER_ID}"
    local OUTPUT_REPO_DIR="${OUTPUT_DIR}_repo"

    # --- Create output directories ---
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_REPO_DIR"

    # --- Activate Conda environment and execute ---
    # !! Note: conda activate must be inside the function because each subshell needs independent activation
    echo "------- Switching to 'xkg' environment -------"
    eval "$(conda shell.bash hook)"
    conda activate xkg

    echo "------- 1. Preprocessing PDF JSON -------"
    python paperbench/agents/paper2code/codes/0_pdf_process.py \
        --input_json_path "${PDF_JSON_PATH}" \
        --output_json_path "${PDF_JSON_CLEANED_PATH}"

    echo "------- 2. PaperCoder - Planning (Planning with Guide) -------"
    python paperbench/agents/paper2code/codes/1_planning_with_knowledge.py \
        --paper_name "$REPO_INDEX" \
        --gpt_version "${GPT_VERSION}" \
        --pdf_json_path "${PDF_JSON_CLEANED_PATH}" \
        --guide_json_path "${GUIDE_PATH}" \
        --output_dir "${OUTPUT_DIR}"

    echo "------- 3. PaperCoder - Extract Config -------"
    python paperbench/agents/paper2code/codes/1.1_extract_config.py \
        --paper_name "$REPO_INDEX" \
        --output_dir "${OUTPUT_DIR}"

    # Copy config file
    cp -rp "${OUTPUT_DIR}/planning_config.yaml" "${OUTPUT_REPO_DIR}/config.yaml"

    echo "------- 4. PaperCoder - Analyzing (Analyzing with Guide) -------"
    python paperbench/agents/paper2code/codes/2_analyzing_with_knowledge.py \
        --paper_name "$REPO_INDEX" \
        --gpt_version "${GPT_VERSION}" \
        --pdf_json_path "${PDF_JSON_CLEANED_PATH}" \
        --guide_json_path "${GUIDE_PATH}" \
        --output_dir "${OUTPUT_DIR}"

    echo "------- 5. PaperCoder - Coding (Coding with guide) -------"
    python paperbench/agents/paper2code/codes/3_coding_with_knowledge.py \
        --paper_name "$REPO_INDEX" \
        --gpt_version "${GPT_VERSION}" \
        --pdf_json_path "${PDF_JSON_CLEANED_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --output_repo_dir "${OUTPUT_REPO_DIR}" \
        --guide_json_path "${GUIDE_PATH}" \

    # --- Switch environment for evaluation ---
    echo "------- Switching to 'paperbench' environment for evaluation -------"
    eval "$(conda shell.bash hook)"
    conda activate paperbench

    local SUBMISSION_PATH="$OUTPUT_REPO_DIR"
    local SAVE_PATH="${SUBMISSION_PATH}/evaluation"

    echo "------- 6. Running evaluation -------"
    python paperbench/scripts/run_judge.py \
         --submission-path "$SUBMISSION_PATH" \
         --paper-id "$PAPER_ID" \
         --judge simple \
         --model "$EVAL_MODEL" \
         --out-dir "$SAVE_PATH" \
         --reasoning-effort high \
         --code-only

    echo ">> Completed run: ${REPO_INDEX}"
    echo "------------------------------------------------------------------------"
}

# --- Main loop and task dispatching ---
# Export function so that subshells can access it via `bash -c`
export -f process_paper
# Export environment variables to ensure subshells can access them
export OPENAI_API_KEY
export OPENAI_BASE_URL
export GPT_VERSION
export EVAL_MODEL
export AGENT_ID
export RUNS_DIR

# Outer loop: control the number of runs for each paper
for (( run_num=1; run_num<=NUM_RUNS_PER_PAPER; run_num++ )); do
    # Inner loop: read split.txt and iterate through each paper
    while IFS= read -r PAPER_ID || [ -n "$PAPER_ID" ]; do
        # Skip empty lines
        if [ -z "$PAPER_ID" ]; then
            continue
        fi

        # When the number of active background tasks reaches the limit, wait for any to complete
        # `jobs -p` lists all background task PIDs
        while (( $(jobs -p | wc -l) >= MAX_PARALLEL_JOBS )); do
            sleep 1
        done

        # Start background task and redirect stdout and stderr to log file
        echo "Dispatching new task for ${PAPER_ID} (run ${run_num})..."
        # Using bash -c ensures execution in a new subshell with convenient parameter passing and redirection
        bash -c "process_paper '$PAPER_ID' '$run_num'" > "${LOG_DIR}/${PAPER_ID}_run${run_num}.log" 2>&1 &

    done < "$SPLIT_FILE"
done

# Wait for all remaining background tasks to complete
echo "All tasks dispatched, waiting for remaining tasks to complete..."
wait
echo "All tasks completed."
