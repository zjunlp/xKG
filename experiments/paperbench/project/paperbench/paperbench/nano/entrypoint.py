# import chz
# from nanoeval.eval import EvalSpec, RunnerArgs
# from nanoeval.evaluation import run
# from nanoeval.setup import nanoeval_entrypoint
# from paperbench.nano.eval import PaperBench
# from paperbench.nano.utils import run_sanity_checks


# @chz.chz
# class DefaultRunnerArgs(RunnerArgs):
#     concurrency: int = 5


# async def main(paperbench: PaperBench, runner: DefaultRunnerArgs) -> None:
#     run_sanity_checks(paperbench)
#     await run(EvalSpec(eval=paperbench, runner=runner))


# if __name__ == "__main__":
#     nanoeval_entrypoint(chz.entrypoint(main))

# =========================================================================
# Copy and replace the following content to your original paperbench/nano/entrypoint.py file
# =========================================================================

import time
import sys
import asyncio

# --- Probe function for easy reuse ---
def probe(message):
    """Print debug message with timestamp and flush=True"""
    print(f"[{time.time()}] PROBE: {message}", flush=True)


probe("--- paperbench/nano/entrypoint.py: File imported by Python. ---")


try:
    import chz
    from nanoeval.eval import EvalSpec, RunnerArgs
    from nanoeval.evaluation import run
    from nanoeval.setup import nanoeval_entrypoint
    from paperbench.nano.eval import PaperBench
    from paperbench.nano.utils import run_sanity_checks
    probe("--- All modules imported successfully. ---")
except ImportError as e:
    probe(f"--- FAILED to import modules: {e} ---")
    sys.exit(1)


@chz.chz
class DefaultRunnerArgs(RunnerArgs):
    probe("--- DefaultRunnerArgs class is being defined. ---")
    concurrency: int = 5


async def main(paperbench: PaperBench, runner: DefaultRunnerArgs) -> None:
    probe("--- async main() function started. ---")
    
    try:
        probe("--- Calling run_sanity_checks()... ---")
        run_sanity_checks(paperbench)
        probe("--- run_sanity_checks() finished. ---")
        
        probe("--- Preparing to call await run(...). This is the main evaluation step. ---")
        # Core logic is in this run function. If program hangs here, the problem is inside nanoeval.evaluation.run.
        await run(EvalSpec(eval=paperbench, runner=runner))
        probe("--- await run(...) finished. ---")
        
    except Exception as e:
        probe(f"--- An exception occurred inside main(): {e} ---")
        # Capture exception to prevent silent program crash
        import traceback
        traceback.print_exc()

    probe("--- async main() function finished. ---")


if __name__ == "__main__":
    probe("--- __name__ == '__main__': Script is being run directly. ---")
    
    try:
        probe("--- Preparing to call chz.entrypoint(main)... ---")
        # chz.entrypoint is responsible for parsing command line arguments and instantiating PaperBench and DefaultRunnerArgs
        entrypoint_func = chz.entrypoint(main)
        probe("--- chz.entrypoint(main) created. ---")

        probe("--- Preparing to call nanoeval_entrypoint(...). This will start the event loop. ---")
        # nanoeval_entrypoint is responsible for starting the asyncio event loop and running main function
        nanoeval_entrypoint(entrypoint_func)
        probe("--- nanoeval_entrypoint(...) finished. ---")
        
    except Exception as e:
        probe(f"--- An exception occurred in the __main__ block: {e} ---")
        import traceback
        traceback.print_exc()

    probe("--- End of script. ---")


