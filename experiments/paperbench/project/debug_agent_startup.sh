#!/bin/bash

set -x  # Print all commands

echo "=============== DEBUG: Agent Startup Investigation ==============="

cd /Users/luoyujie/Documents/Code/xKG/experiments/paperbench/project/paperbench

# Step 1: Check if base image exists
echo ""
echo "[STEP 1] Checking if base image 'aisi-basic-agent' exists..."
docker images | grep aisi-basic-agent || echo "Base image not found!"

# Step 2: Check Docker daemon
echo ""
echo "[STEP 2] Checking Docker daemon status..."
docker ps | head -5 || echo "Docker connection failed!"

# Step 3: Try to build the image with verbose output
echo ""
echo "[STEP 3] Attempting to build aisi-basic-agent-basic:latest image..."
echo "Current directory: $(pwd)"
echo "Files in current directory:"
ls -la | head -20

# Try to build with verbose output and timeout
timeout 60 docker build -t "aisi-basic-agent-basic:latest" -f - . <<'EOF'
FROM aisi-basic-agent

ENV HF_ENDPOINT="https://hf-mirror.com"

COPY ./paperbench/agents/aisi-basic-agent/ /home/agent/
EOF

BUILD_EXIT=$?
echo ""
echo "[STEP 3 RESULT] Docker build exit code: $BUILD_EXIT"

if [ $BUILD_EXIT -eq 0 ]; then
    echo "✓ Docker image build successful"
    docker images | grep aisi-basic-agent-basic
else
    echo "✗ Docker image build failed or timed out"
fi

# Step 4: Check if agent directory exists
echo ""
echo "[STEP 4] Checking agent directory..."
if [ -d "./paperbench/agents/aisi-basic-agent/" ]; then
    echo "✓ Agent directory found"
    ls -la ./paperbench/agents/aisi-basic-agent/ | head -20
else
    echo "✗ Agent directory not found!"
fi

# Step 5: Check for Dockerfile
echo ""
echo "[STEP 5] Checking agent Dockerfile..."
if [ -f "./paperbench/agents/aisi-basic-agent/Dockerfile" ]; then
    echo "✓ Dockerfile found"
    head -20 ./paperbench/agents/aisi-basic-agent/Dockerfile
else
    echo "✗ Dockerfile not found!"
fi

echo ""
echo "=============== END DEBUG ==============="
