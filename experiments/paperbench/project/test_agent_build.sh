#!/bin/bash

set -x

echo "=============== Testing Agent Build Fix ==============="

# Get the paperbench project directory
PAPERBENCH_DIR="$(cd "$(dirname "$0")/paperbench" && pwd)"
cd "${PAPERBENCH_DIR}"

echo "Working directory: $(pwd)"
echo "Agent directory exists: $([ -d ./paperbench/agents/aisi-basic-agent/ ] && echo 'YES' || echo 'NO')"

# Try to build the image
NEW_IMAGE="aisi-basic-agent-basic-test:latest"

echo ""
echo "Building Docker image: ${NEW_IMAGE}"
echo "Build context: $(pwd)"
echo ""

# Build with a timeout (60 seconds) to avoid infinite hangs
# Note: On macOS, use `gtimeout` from brew if available, otherwise use Python's timeout
build_timeout=60
(
    docker build -t "${NEW_IMAGE}" -f - . <<'EOF'
FROM aisi-basic-agent

ENV HF_ENDPOINT="https://hf-mirror.com"

COPY ./paperbench/agents/aisi-basic-agent/ /home/agent/
EOF
) &
BUILD_PID=$!

# Wait with timeout
sleep $build_timeout &
SLEEP_PID=$!
wait -n BUILD_PID SLEEP_PID 2>/dev/null
BUILD_STATUS=$?

if [ $BUILD_STATUS -eq 0 ]; then
    kill $BUILD_PID 2>/dev/null || true
    echo ""
    echo "✓ Docker build completed successfully"
    docker images | grep aisi-basic-agent-basic-test
else
    kill $BUILD_PID 2>/dev/null || true
    echo ""
    echo "✗ Docker build failed or timed out"
fi

echo ""
echo "=============== END TEST ==============="
