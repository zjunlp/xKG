"""
Test CodeVerifier: Load unverified Node JSON and verify it in one step.

Usage:
    python -m xKG.source.run_test_verifier
"""

import os
import json
import logging
from dataclasses import asdict
from pathlib import Path

from .utils import initialize_app, get_config, get_code_rag_config
from .components import CodeVerifier
from .schema import Node

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to unverified Node JSON
# Note: Path is relative to xKG package root (two levels up from this file's parent components/)
UNVERIFIED_NODE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "storage", "process", "graph_handler",
    "How_Does_DPO_Reduce_Toxicity_A_Mechanistic_Neuron-Level_Analysis_unverified.json"
)


def main():
    logger.info("=" * 70)
    logger.info("CodeVerifier Test: Load and Verify Node")
    logger.info("=" * 70)

    # Initialize app configuration
    initialize_app()
    config = get_config()

    logger.info(f"Profile: {config.get('profile_name', 'default')}")
    logger.info(f"Model: {config.get('model', 'unknown')}")
    logger.info(f"Code Model: {config.get('code_model', config.get('model', 'unknown'))}")

    # Load unverified Node
    logger.info("\n" + "=" * 70)
    logger.info("Step 1: Loading unverified Node from JSON")
    logger.info("=" * 70)

    if not os.path.exists(UNVERIFIED_NODE_PATH):
        logger.error(f"Unverified Node not found: {UNVERIFIED_NODE_PATH}")
        logger.error("Please run graph_handler to generate unverified Node first.")
        return False

    try:
        with open(UNVERIFIED_NODE_PATH, 'r', encoding='utf-8') as f:
            node_data = json.load(f)
        node = Node.from_dict(node_data)
        logger.info(f"✓ Successfully loaded Node: {node.paper_title}")
        logger.info(f"  Techniques: {len(node.techniques) if node.techniques else 0}")
    except Exception as e:
        logger.error(f"Failed to load Node: {e}", exc_info=True)
        return False

    # Create CodeVerifier
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: Creating CodeVerifier")
    logger.info("=" * 70)

    try:
        docker_image = get_code_rag_config().get('docker_image', 'xkg-coderunner:latest')
        logger.info(f"Docker image: {docker_image}")

        verifier = CodeVerifier(
            model=config.get('code_model', config.get('model')),
            docker_image=docker_image
        )
        logger.info("✓ CodeVerifier created successfully")
    except Exception as e:
        logger.error(f"Failed to create CodeVerifier: {e}", exc_info=True)
        return False

    # Verify Node
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: Running Code Verification")
    logger.info("=" * 70)

    try:
        verified_node = verifier.verify(
            node,
            save_process=True,
            use_parallel=True,
            max_workers=1,  # Serial to ensure complete debug
            max_debug_attempts=10,  # Maximum attempts to fix test code
            skip_dependency_install=False
        )

        if verified_node is None:
            logger.error("verify() returned None")
            return False

        logger.info(f"✓ Verification completed")
        logger.info(f"  Paper title: {verified_node.paper_title}")
        logger.info(f"  Techniques verified: {len(verified_node.techniques) if verified_node.techniques else 0}")
    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        return False
    finally:
        # Cleanup
        verifier.cleanup()

    # Save verified Node
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: Saving Verified Node")
    logger.info("=" * 70)

    try:
        output_path = UNVERIFIED_NODE_PATH.replace("_unverified.json", "_verified.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(verified_node), f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Verified Node saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save verified Node: {e}", exc_info=True)
        return False

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("VERIFICATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
