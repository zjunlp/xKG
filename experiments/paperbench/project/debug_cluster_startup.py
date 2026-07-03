#!/usr/bin/env python3
"""
Debug script to test cluster startup in isolation
"""
import asyncio
import os
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

sys.path.insert(0, '/Users/luoyujie/Documents/Code/xKG/experiments/paperbench/project/alcatraz')
sys.path.insert(0, '/Users/luoyujie/Documents/Code/xKG/experiments/paperbench/project/paperbench')

from alcatraz.clusters.local import LocalConfig

async def test_cluster_startup():
    """Test basic cluster startup without Jupyter"""

    print("\n" + "="*60)
    print("STEP 1: Creating LocalConfig")
    print("="*60)

    config = LocalConfig(
        image="aisi-basic-agent-basic:latest",
        pull_from_registry=False,
        wait_for_health=False,
        local_network=True,
        environment={},
    )

    print(f"✓ Config created: {config}")

    print("\n" + "="*60)
    print("STEP 2: Building cluster (starting container)")
    print("="*60)
    print("Timeout: 30 seconds")

    try:
        async with asyncio.timeout(30):
            async with config.build() as cluster:
                print(f"✓ Cluster started successfully")
                print(f"  Containers: {await cluster.fetch_container_names()}")

                print("\n" + "="*60)
                print("STEP 3: Testing shell command")
                print("="*60)

                # Test a simple shell command
                result = await cluster.send_shell_command("echo 'Hello from container' && ls /home/")
                print(f"✓ Shell command executed")
                print(f"  Exit code: {result['exit_code']}")
                print(f"  Output:\n{result['result'].decode('utf-8')}")

                print("\n✓ Cluster startup test PASSED")

    except asyncio.TimeoutError:
        print("✗ Cluster startup TIMED OUT after 30 seconds")
        return False
    except Exception as e:
        print(f"✗ Cluster startup FAILED with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = asyncio.run(test_cluster_startup())
    sys.exit(0 if success else 1)
