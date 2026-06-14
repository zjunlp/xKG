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
# 请将以下所有内容复制并替换你原来的 paperbench/nano/entrypoint.py 文件
# =========================================================================

import time
import sys
import asyncio

# --- 探针函数，方便复用 ---
def probe(message):
    """打印带有时间戳和 flush=True 的调试信息"""
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
    # 如果导入失败，这里会打印错误并退出，但你的问题是高CPU占用，所以应该不是这里的问题。
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
        # 核心逻辑就在这个 run 函数里。如果程序卡在这里，说明问题在 nanoeval.evaluation.run 内部。
        await run(EvalSpec(eval=paperbench, runner=runner))
        probe("--- await run(...) finished. ---")
        
    except Exception as e:
        probe(f"--- An exception occurred inside main(): {e} ---")
        # 捕获异常，以防程序悄无声息地崩溃
        import traceback
        traceback.print_exc()

    probe("--- async main() function finished. ---")


if __name__ == "__main__":
    probe("--- __name__ == '__main__': Script is being run directly. ---")
    
    try:
        probe("--- Preparing to call chz.entrypoint(main)... ---")
        # chz.entrypoint 负责解析命令行参数并实例化 PaperBench 和 DefaultRunnerArgs
        entrypoint_func = chz.entrypoint(main)
        probe("--- chz.entrypoint(main) created. ---")
        
        probe("--- Preparing to call nanoeval_entrypoint(...). This will start the event loop. ---")
        # nanoeval_entrypoint 负责启动 asyncio 事件循环并运行 main 函数
        nanoeval_entrypoint(entrypoint_func)
        probe("--- nanoeval_entrypoint(...) finished. ---")
        
    except Exception as e:
        probe(f"--- An exception occurred in the __main__ block: {e} ---")
        import traceback
        traceback.print_exc()

    probe("--- End of script. ---")


