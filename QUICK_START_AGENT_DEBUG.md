# Quick Start: Agent Startup Fix & Validation

## TL;DR - The Fix

**Problem**: Agent startup hangs in infinite loop  
**Cause**: Docker build context is wrong  
**Fix**: Change working directory before docker build  
**Status**: ✅ **ALREADY APPLIED** in this commit

---

## Quickest Validation (5 minutes)

### Step 1: Verify the fix exists
```bash
grep "cd.*PAPERBENCH_DIR" experiments/paperbench/project/paperbench/scripts/aisi-basic/run.sh
```

Expected: Should show `cd "${PAPERBENCH_DIR}"`

### Step 2: Run quick diagnostic
```bash
cd experiments/paperbench/project
bash comprehensive_debug.sh 2>&1 | tail -30
```

Expected: `✓ All tests PASSED`

### Step 3: Watch Docker build succeed
```bash
cd experiments/paperbench/project/paperbench
timeout 120 docker build -t test-aisi:latest -f - . <<'EOF'
FROM aisi-basic-agent
ENV HF_ENDPOINT="https://hf-mirror.com"
COPY ./paperbench/agents/aisi-basic-agent/ /home/agent/
EOF
echo "Build exit code: $?"
```

Expected: Exit code 0 (success)

---

## Full Test (10-15 minutes)

### Option A: Comprehensive Diagnostic
```bash
cd /Users/luoyujie/Documents/Code/xKG/experiments/paperbench/project
chmod +x comprehensive_debug.sh
./comprehensive_debug.sh
```

This tests:
- ✓ Docker daemon running
- ✓ Base images available
- ✓ Agent directory structure correct
- ✓ Docker build works (120s timeout)
- ✓ Container startup works (30s timeout)
- ✓ Shell commands execute in container

### Option B: Manual Agent Run (60+ seconds)
```bash
cd /Users/luoyujie/Documents/Code/xKG/experiments/paperbench/project/paperbench

# Terminal 1: Run the agent
scripts/aisi-basic/run.sh

# Terminal 2: Monitor progress
# After ~10 seconds, you should see agent.log appear
ls -la experiments/paperbench/project/paperbench/outputs/*/logs/agent.log 2>/dev/null && \
  echo "✓ agent.log found - agent is running!" || \
  echo "⏳ Still waiting for agent.log..."

# Terminal 2: Once agent.log appears, watch it
tail -f experiments/paperbench/project/paperbench/outputs/*/logs/agent.log
```

---

## What to Expect

### ✅ Success Indicators

- [ ] No "Attempting to start a cluster instance" repeated messages
- [ ] Docker image builds in < 60 seconds
- [ ] Container appears in `docker ps` output
- [ ] `agent.log` file is created in outputs directory
- [ ] `agent.log` contains actual agent execution logs (not errors)
- [ ] Process eventually completes or times out gracefully

### ❌ Failure Indicators

- [ ] Repeated "Attempting to start a cluster instance" (dead loop)
- [ ] Docker build hangs > 60 seconds
- [ ] Container doesn't appear in `docker ps`
- [ ] Process killed after 600 seconds with timeout
- [ ] No `agent.log` file created

---

## Troubleshooting One-Liners

If something doesn't work, try these in order:

### 1. Is Docker running?
```bash
docker ps
# If error: docker daemon is NOT running. Start Docker.
```

### 2. Do base images exist?
```bash
docker images | grep aisi-basic-agent
# If nothing shown: base image missing. Check Docker setup.
```

### 3. Is there leftover state?
```bash
# Kill existing containers
docker ps -a | grep aisi-basic-agent | awk '{print $1}' | xargs docker rm -f

# Clean networks
docker network prune -f

# Try again
```

### 4. Is there disk space?
```bash
df -h /var/lib/docker
# If < 5GB: need to free space
```

### 5. Check actual error
```bash
# Get container ID from docker ps
CONTAINER_ID=$(docker ps -q | head -1)
docker logs $CONTAINER_ID 2>&1 | tail -50
```

---

## Understanding the Output

### Docker Build Phase (Expected Output)
```
Building Docker image from directory: /Users/luoyujie/Documents/Code/xKG/experiments/paperbench/project/paperbench
Image name: aisi-basic-agent-basic:latest
Sending build context to Docker daemon  ...MB
Step 1/7 : FROM aisi-basic-agent
 ---> <hash>
Step 2/7 : ENV HF_ENDPOINT="https://hf-mirror.com"
 ---> <hash>
Step 3/7 : COPY ./paperbench/agents/aisi-basic-agent/ /home/agent/
 ---> <hash>
...
Successfully tagged aisi-basic-agent-basic:latest
```

**Key line**: "Successfully tagged" = build worked

### Cluster Startup Phase (Expected Output)
```
cluster_config: "image='aisi-basic-agent-basic:latest' ... "
Attempting to start a cluster instance. This may take a while...
Starting main container for aisi-basic-agent-basic:latest
Allocated host ports [xxxxx] -> container ports [...]
Started new container! Container name: container0-xxxxx
```

**Key line**: "Started new container!" = cluster working

### Agent Execution Phase (Expected Output)
```
Agent `aisi-basic-agent-my-o3` is attempting to replicate the `xxxxx` paper...
Writing logs for run to ...agent.log
Starting evaluation for task ...
```

Then `agent.log` should appear with actual agent activity.

---

## The Fix Explained (For Reference)

### What Changed
```diff
  #!/bin/bash
  set -e
  PAPERBENCH_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
  ...parameters...

+ # Change to PAPERBENCH_DIR before building Docker image
+ # This ensures COPY commands in the Dockerfile work correctly
+ cd "${PAPERBENCH_DIR}"
+
+ echo "Building Docker image from directory: $(pwd)"
+ echo "Image name: ${NEW_IMAGE}"
  
  docker build -t "${NEW_IMAGE}" -f - . <<'EOF'
```

### Why It Works
- Before: `.` (build context) was the current working directory
- After: `.` (build context) is `PAPERBENCH_DIR` (correct directory with agent files)
- Result: `COPY ./paperbench/agents/aisi-basic-agent/` finds the files ✓

---

## One-Command Test

```bash
cd /Users/luoyujie/Documents/Code/xKG && \
python3 -c "import subprocess; print('Docker works!' if subprocess.run(['docker', 'ps'], capture_output=True).returncode == 0 else 'Docker FAILED!')" && \
echo "✓ System ready for agent tests"
```

---

## Still Having Issues?

See the full guide: `./AGENT_DEBUG_GUIDE.md`

Key sections:
- "Root Causes Identified" - Detailed analysis
- "Troubleshooting Checklist" - Complete verification steps
- "If Still Stuck" - Emergency procedures

---

## Success Confirmation

Once you see this, the fix is working:
```
✓ Docker build completed successfully
✓ Container started in X.X seconds
✓ Shell command executed: Hello from container
✓ All tests PASSED
```

Then agent execution should proceed normally and produce `agent.log`.

---

**Questions?** Refer to the comprehensive debug guide at:
`./AGENT_DEBUG_GUIDE.md`
