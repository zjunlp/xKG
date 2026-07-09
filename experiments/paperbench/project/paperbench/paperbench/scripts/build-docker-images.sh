#!/bin/bash

# Auto-detect platform: Mac ARM64 or Linux x86_64
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="linux/arm64"
else
    PLATFORM="linux/amd64"
fi

echo "Building Docker images for platform: $PLATFORM"

XKG_ROOT="$(cd "$(dirname "$0")/../../../../../../" && pwd)"

docker build --platform=$PLATFORM -t pb-env -f paperbench/agents/Dockerfile.base paperbench/agents/
docker build --platform=$PLATFORM -t dummy paperbench/agents/dummy/
docker build --platform=$PLATFORM -t aisi-basic-agent paperbench/agents/aisi-basic-agent/
docker build --platform=$PLATFORM -f paperbench/grader.Dockerfile -t pb-grader .
docker build --platform=$PLATFORM -f paperbench/reproducer.Dockerfile -t pb-reproducer .
docker build --platform=$PLATFORM -t aisi-basic-knowledge-agent \
    -f "${XKG_ROOT}/experiments/paperbench/project/paperbench/paperbench/agents/aisi-basic-agent/Dockerfile.knowledge" \
    "$XKG_ROOT"