#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 自动检测平台
PLATFORM=$(uname -m)
if [ "$PLATFORM" = "arm64" ] || [ "$PLATFORM" = "aarch64" ]; then
    BUILD_PLATFORM="linux/arm64"
else
    BUILD_PLATFORM="linux/amd64"
fi

echo "[INFO] Detected platform: $PLATFORM → $BUILD_PLATFORM"
echo "[INFO] Building xkg-coderunner:latest..."

docker build \
    --platform=$BUILD_PLATFORM \
    -f "$SCRIPT_DIR/Dockerfile.coderunner" \
    -t xkg-coderunner:latest \
    "$PROJECT_ROOT"

echo "[INFO] ✓ Build complete: xkg-coderunner:latest"
docker images | grep xkg-coderunner | head -1
