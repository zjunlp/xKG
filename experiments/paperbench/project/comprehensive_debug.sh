#!/bin/bash

set -e

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  Comprehensive Agent Startup Debug Suite               ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Get paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPERBENCH_DIR="$(cd "$SCRIPT_DIR/paperbench" && pwd)"

echo "Script directory: $SCRIPT_DIR"
echo "Paperbench directory: $PAPERBENCH_DIR"
echo ""

# ============================================================
# TEST 1: Docker daemon and images
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: Docker daemon and images"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if docker ps > /dev/null 2>&1; then
    echo "✓ Docker daemon is running"
else
    echo "✗ Docker daemon is NOT running"
    echo "  Please start Docker and try again"
    exit 1
fi

if docker images | grep -q "aisi-basic-agent:latest"; then
    echo "✓ Base image 'aisi-basic-agent:latest' exists"
else
    echo "✗ Base image 'aisi-basic-agent:latest' NOT found"
    echo "  Available images:"
    docker images | grep aisi-basic-agent || echo "  (none found)"
    exit 1
fi

echo ""

# ============================================================
# TEST 2: Check agent directory structure
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: Agent directory structure"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$PAPERBENCH_DIR"
AGENT_DIR="./paperbench/agents/aisi-basic-agent"

if [ ! -d "$AGENT_DIR" ]; then
    echo "✗ Agent directory NOT found at: $AGENT_DIR"
    exit 1
fi
echo "✓ Agent directory exists: $AGENT_DIR"

for file in "start.sh" "start.py" "Dockerfile"; do
    if [ ! -f "$AGENT_DIR/$file" ]; then
        echo "✗ Missing file: $file"
        exit 1
    fi
    echo "✓ Found: $file"
done

echo ""

# ============================================================
# TEST 3: Docker build (with timeout)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 3: Docker image build (timeout: 120 seconds)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TEST_IMAGE="aisi-basic-agent-basic-test:$(date +%s)"
echo "Test image: $TEST_IMAGE"
echo "Current directory: $(pwd)"
echo ""

# Use Python for cross-platform timeout
python3 << 'PYTHON_EOF'
import subprocess
import sys
import signal
import os

def timeout_handler(signum, frame):
    raise TimeoutError("Docker build timed out after 120 seconds")

# Set up signal handler for timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(120)

try:
    # Read the Dockerfile heredoc
    dockerfile_content = '''FROM aisi-basic-agent

ENV HF_ENDPOINT="https://hf-mirror.com"

COPY ./paperbench/agents/aisi-basic-agent/ /home/agent/
'''

    # Run docker build
    print("Starting Docker build...")
    process = subprocess.Popen(
        ['docker', 'build', '-t', os.environ.get('TEST_IMAGE', 'test:latest'), '-f', '-', '.'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # Line buffered
    )

    # Stream output
    for line in process.stdout:
        print(line, end='')
        sys.stdout.flush()

    returncode = process.wait()
    signal.alarm(0)  # Disable alarm

    if returncode == 0:
        print("\n✓ Docker build completed successfully")
        sys.exit(0)
    else:
        print(f"\n✗ Docker build failed with exit code: {returncode}")
        sys.exit(1)

except TimeoutError:
    print("\n✗ Docker build TIMED OUT after 120 seconds")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Docker build FAILED: {e}")
    sys.exit(1)
PYTHON_EOF

BUILD_EXIT=$?

if [ $BUILD_EXIT -eq 0 ]; then
    echo ""
    echo "Verifying built image:"
    docker images | grep aisi-basic-agent-basic-test || echo "  (image not found in docker images)"
else
    echo ""
    echo "✗ Docker build failed or timed out"
    exit 1
fi

echo ""

# ============================================================
# TEST 4: Container startup
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 4: Container startup (timeout: 30 seconds)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

CONTAINER_NAME="test-agent-$(date +%s)"
echo "Container name: $CONTAINER_NAME"
echo "Image: aisi-basic-agent-basic-test:*"
echo ""

# Use Python timeout for container startup
python3 << PYTHON_EOF
import subprocess
import sys
import time

try:
    print("Starting container...")

    # Get the image ID from docker images
    result = subprocess.run(
        ['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}', 'aisi-basic-agent-basic-test'],
        capture_output=True,
        text=True,
        timeout=5
    )

    images = result.stdout.strip().split('\n')
    if not images or images[0] == '':
        print("✗ No test image found")
        sys.exit(1)

    test_image = images[0]
    print(f"Using image: {test_image}")

    # Start container with timeout
    start_time = time.time()
    try:
        container = subprocess.run(
            ['docker', 'run', '-d', '--rm', test_image, 'sleep', '60'],
            capture_output=True,
            text=True,
            timeout=30
        )
    except subprocess.TimeoutExpired:
        print("✗ Container start TIMED OUT")
        sys.exit(1)

    if container.returncode != 0:
        print(f"✗ Container start FAILED: {container.stderr}")
        sys.exit(1)

    container_id = container.stdout.strip()
    elapsed = time.time() - start_time

    print(f"✓ Container started in {elapsed:.1f} seconds")
    print(f"  Container ID: {container_id}")

    # Verify container is running
    check = subprocess.run(
        ['docker', 'ps', '--filter', f'id={container_id}'],
        capture_output=True,
        text=True,
        timeout=5
    )

    if container_id in check.stdout:
        print("✓ Container is running")
    else:
        print("✗ Container is not running")
        sys.exit(1)

    # Test shell command
    print("\nTesting shell command in container...")
    exec_result = subprocess.run(
        ['docker', 'exec', container_id, 'echo', 'Hello from container'],
        capture_output=True,
        text=True,
        timeout=10
    )

    if exec_result.returncode == 0:
        print(f"✓ Shell command executed: {exec_result.stdout.strip()}")
    else:
        print(f"✗ Shell command failed: {exec_result.stderr}")
        sys.exit(1)

    # Cleanup
    subprocess.run(['docker', 'kill', container_id], capture_output=True)

except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
PYTHON_EOF

TEST4_EXIT=$?
[ $TEST4_EXIT -eq 0 ] && echo "" || exit 1

# ============================================================
# Summary
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ All tests PASSED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Environment is ready for agent execution!"
echo ""
echo "Next steps:"
echo "1. Run: ./scripts/aisi-basic/run.sh"
echo "2. Monitor logs in: outputs/"
echo ""
