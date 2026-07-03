# Agent Startup Dead Loop - Debug & Fix Guide

## Problem Summary

When running the agent startup script (`scripts/aisi-basic/run.sh`), the system gets stuck with:
```
{'component': 'paperbench.nano.eval', 'event': 'Attempting to start a cluster instance. This may take a while...', 'timestamp': '2026-07-03T17:53:48.889634Z', 'level': 'info'}
```

And repeats in a loop without progressing, eventually timing out.

## Root Causes Identified

### 1. **Docker Build Context Issue (PRIMARY)**

The `run.sh` script doesn't change to the correct working directory before running `docker build`. This causes the `COPY` command in the Dockerfile to fail silently or hang looking for the source files.

**Location**: `scripts/aisi-basic/run.sh`, line 24

**Problem**:
```bash
docker build -t "${NEW_IMAGE}" -f - . <<'EOF'  # Build context is CURRENT directory
...
COPY ./paperbench/agents/aisi-basic-agent/ /home/agent/  # Looking for file relative to build context
EOF
```

**Fix**: Change to `PAPERBENCH_DIR` before building

```bash
cd "${PAPERBENCH_DIR}"  # Change to correct directory
docker build -t "${NEW_IMAGE}" -f - . <<'EOF'  # Now build context is correct
```

### 2. **Missing Health Check During Cluster Startup (SECONDARY)**

The Alcatraz cluster startup includes a `wait_for_health()` check that loops for 60 seconds when no HEALTHCHECK is defined. However, this is correctly set to `wait_for_health=False` in the config (eval.py:730), so it should be a non-issue.

**Status**: Already handled correctly by setting `wait_for_health=False`

### 3. **Agent Execution Doesn't Require Jupyter (INFO)**

The `AlcatrazComputerInterface` has optional Jupyter kernel support that's only triggered by the `execute()` method, which is used for grading/reproduction, NOT for agent shell execution. The agent uses `send_shell_command()` which works directly via Docker exec.

**Status**: Not a blocker for agent execution, only for task grading

## Solution: What's Fixed

### Fixed in `scripts/aisi-basic/run.sh`:

```bash
#!/bin/bash

set -e

# 获取脚本所在目录的绝对路径（paperbench目录）
PAPERBENCH_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# ... parameters ...

# Change to PAPERBENCH_DIR before building Docker image
# This ensures COPY commands in the Dockerfile work correctly
cd "${PAPERBENCH_DIR}"

echo "Building Docker image from directory: $(pwd)"
echo "Image name: ${NEW_IMAGE}"

docker build -t "${NEW_IMAGE}" -f - . <<'EOF'
FROM aisi-basic-agent

ENV HF_ENDPOINT="https://hf-mirror.com"

COPY ./paperbench/agents/aisi-basic-agent/ /home/agent/
EOF
```

## How to Debug Agent Startup

### Option 1: Run Comprehensive Diagnostic

```bash
cd /Users/luoyujie/Documents/Code/xKG/experiments/paperbench/project
chmod +x comprehensive_debug.sh
./comprehensive_debug.sh
```

This script tests:
- Docker daemon connectivity
- Base image availability
- Agent directory structure
- Docker image building (with 120-second timeout)
- Container startup (with 30-second timeout)
- Shell command execution in container

### Option 2: Step-by-Step Manual Debugging

```bash
cd /Users/luoyujie/Documents/Code/xKG/experiments/paperbench/project/paperbench

# Step 1: Verify base image
docker images | grep aisi-basic-agent

# Step 2: Build test image
docker build -t aisi-test:latest -f - . <<'EOF'
FROM aisi-basic-agent
ENV HF_ENDPOINT="https://hf-mirror.com"
COPY ./paperbench/agents/aisi-basic-agent/ /home/agent/
EOF

# Step 3: Start container and test
CONTAINER=$(docker run -d aisi-test:latest sleep 300)
docker exec $CONTAINER echo "Hello from container"
docker kill $CONTAINER
```

### Option 3: Monitor Docker During Agent Run

In a separate terminal while the agent is running:

```bash
# Watch Docker containers
watch -n 1 'docker ps'

# Watch container logs (replace CONTAINER_ID)
docker logs -f CONTAINER_ID

# Check disk space
df -h
```

## Expected Success Indicators

### ✓ Docker Build Phase
- `Sending build context to Docker daemon` message appears
- Build completes without hanging
- `Successfully tagged aisi-basic-agent-basic:latest` message appears

### ✓ Cluster Startup Phase
- Container starts within 5-10 seconds
- `docker ps` shows the running container

### ✓ Agent Execution Phase
- `agent.log` file appears in the output directory
- Contains output from `inspect_ai` task execution
- Agent makes progress (LLM calls, tool usage, etc.)

## Troubleshooting Checklist

- [ ] Docker daemon is running: `docker ps`
- [ ] Base image exists: `docker images | grep aisi-basic-agent:latest`
- [ ] Agent directory exists: `ls -la ./paperbench/agents/aisi-basic-agent/`
- [ ] No docker build processes hanging: `ps aux | grep docker`
- [ ] Sufficient disk space: `df -h /var/lib/docker`
- [ ] Docker socket is accessible: `ls -la /var/run/docker.sock`
- [ ] Python dependencies installed: `pip list | grep inspect-ai`

## If Still Stuck

1. **Kill existing containers**:
   ```bash
   docker ps -a | grep aisi-basic-agent | awk '{print $1}' | xargs docker rm -f
   ```

2. **Clean up docker networks**:
   ```bash
   docker network prune -f
   ```

3. **Check docker logs**:
   ```bash
   # macOS/Linux
   docker info
   # On macOS, might also check Docker app logs
   ```

4. **Run with verbose output**:
   ```bash
   export PYTHONUNBUFFERED=1
   python -m paperbench.nano.entrypoint \
       paperbench.paper_split="dev" \
       paperbench.solver.agent_id="aisi-basic-agent-my-o3" \
       paperbench.judge.code_only=True \
       paperbench.judge.model="o3-mini" \
       paperbench.solver=paperbench.nano.eval:ExternalPythonCodingSolver \
       paperbench.solver.cluster_config=alcatraz.clusters.local:LocalConfig \
       paperbench.solver.cluster_config.image="aisi-basic-agent-basic:latest" \
       paperbench.solver.cluster_config.local_network=True \
       runner.recorder=nanoeval.json_recorder:json_recorder \
       paperbench.solver.is_nvidia_gpu_env="False"
   ```

5. **Check agent logs in container**:
   ```bash
   # Get container ID from docker ps
   docker exec <CONTAINER_ID> ls -la /home/logs/
   docker exec <CONTAINER_ID> tail -100 /home/logs/inspect.log
   ```

## Files Modified

- ✅ `/Users/luoyujie/Documents/Code/xKG/experiments/paperbench/project/paperbench/scripts/aisi-basic/run.sh` - Added `cd "${PAPERBENCH_DIR}"` before docker build

## Files Created for Debugging

- `scripts/aisi-basic/run_debug.sh` - Run script with debug output
- `comprehensive_debug.sh` - Full diagnostic suite
- `debug_agent_startup.sh` - Quick sanity checks
- `AGENT_DEBUG_GUIDE.md` - This guide

## Next Steps

1. ✅ Apply the fix to `run.sh` (already done)
2. Run the comprehensive diagnostic: `./comprehensive_debug.sh`
3. If tests pass, try: `./scripts/aisi-basic/run.sh`
4. Monitor for `agent.log` file creation as success indicator
5. If still issues, capture output from comprehensive_debug.sh and share for analysis

---

**Key Insight**: The dead loop is almost certainly caused by the Docker build trying to copy files that don't exist at the expected path. By fixing the working directory, the build should complete and allow the agent to start properly.
