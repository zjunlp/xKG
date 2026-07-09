#!/bin/bash

MAX_PARALLEL_JOBS=5

ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
    export OPENAI_API_KEY=$(grep '^OPENAI_API_KEY=' "$ENV_FILE" | cut -d'=' -f2-)
    export OPENAI_BASE_URL=$(grep '^OPENAI_BASE_URL=' "$ENV_FILE" | cut -d'=' -f2-)
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
RUNS_DIR="${SCRIPT_DIR}/../../runs/${RUN_TIMESTAMP}_paper2code"

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
    # local REPO_INDEX="mechanistic-understanding-20250929_005617"

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
    echo "------- 1. Preprocessing PDF JSON -------"
    python paperbench/agents/paper2code/codes/0_pdf_process.py \
        --input_json_path "${PDF_JSON_PATH}" \
        --output_json_path "${PDF_JSON_CLEANED_PATH}"

    echo "------- 2. PaperCoder - Planning -------"
    python paperbench/agents/paper2code/codes/1_planning.py \
        --paper_name "$REPO_INDEX" \
        --gpt_version "${GPT_VERSION}" \
        --pdf_json_path "${PDF_JSON_CLEANED_PATH}" \
        --output_dir "${OUTPUT_DIR}"

    echo "------- 3. PaperCoder - Extract Config -------"
    python paperbench/agents/paper2code/codes/1.1_extract_config.py \
        --paper_name "$REPO_INDEX" \
        --output_dir "${OUTPUT_DIR}"

    # Copy config file
    cp -rp "${OUTPUT_DIR}/planning_config.yaml" "${OUTPUT_REPO_DIR}/config.yaml"

    echo "------- 4. PaperCoder - Analyzing -------"
    python paperbench/agents/paper2code/codes/2_analyzing.py \
        --paper_name "$REPO_INDEX" \
        --gpt_version "${GPT_VERSION}" \
        --pdf_json_path "${PDF_JSON_CLEANED_PATH}" \
        --output_dir "${OUTPUT_DIR}"

    echo "------- 5. PaperCoder - Coding -------"
    python paperbench/agents/paper2code/codes/3_coding.py \
        --paper_name "$REPO_INDEX" \
        --gpt_version "${GPT_VERSION}" \
        --pdf_json_path "${PDF_JSON_CLEANED_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --output_repo_dir "${OUTPUT_REPO_DIR}"

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
