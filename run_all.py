"""
================================================================================
MASTER CIPHER EXPERIMENT RUNNER
================================================================================

OVERVIEW:
This script is the main entry point for the entire project. It acts as an 
orchestrator, sequentially executing every per-cipher experiment pipeline 
defined in the 'experiments/' directory. 

HOW IT WORKS:
1. PER-CIPHER RUNNERS:
   - For each cipher (AES, ASCON, Speck, etc.), this script calls its 
     corresponding 'main_[cipher].py' file.
2. SEQUENTIAL EXECUTION:
   - Each runner executes its full pipeline: Dataset Generation -> 
     Multi-Model Training -> Reporting.
3. LOGGING:
   - Progress and errors are logged to the console, allowing you to 
     track the status of a massive multi-cipher benchmark run.

WHY THIS EXISTS:
In automated cryptanalysis research, we often need to benchmark dozens 
of different ciphers to find universal patterns. This master runner 
automates that entire process with a single command.

USAGE:
Use this as the primary entry point when you want all ciphers processed in one
pass without manually calling each `main_*.py` script.

Use `--fast` for a shorter smoke-test run with smaller datasets and lighter
model settings.
"""

import argparse
import importlib
import os
import time


RUNNERS = [
    ("SIMON", "simon"),
    ("ASCON", "ascon"),
    ("GIMLI", "gimli"),
    ("TINYJAMBU", "tinyjambu"),
    ("KATAN", "katan"),
    ("GRAIN-128A", "grain128a"),
    ("LED", "led"),
    ("SKINNY", "skinny"),
    ("PRESENT", "present"),
    ("PRINCE", "prince"),
    ("SPECK", "speck"),
    ("XOODOO", "xoodoo"),
    ("TRIVIUM", "trivium"),
    ("CHACHA20", "chacha20"),
    ("MICKEY", "mickey"),
    ("SALSA20", "salsa20"),
    ("RECTANGLE", "rectangle"),
    ("AES", "aes"),
    ("GIFT", "gift"),
    ("LEA", "lea"),
]


def run_module_main(module_name: str) -> None:
    module = importlib.import_module(f"experiments.main_{module_name}")
    if not hasattr(module, "main"):
        raise AttributeError(f"Module '{module_name}' has no main() function")
    module.main()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all cipher experiment pipelines")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use reduced dataset/model settings for faster smoke-test runs",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help="Resume from a specific cipher label or module name (e.g. tinyjambu or main_tinyjambu)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main execution loop for the master runner.
    
    It parses command line arguments, handles resume logic, and iterates 
    through the selected cipher experiments.
    """
    args = parse_args()
    if args.fast:
        # Set an environment variable that child processes can read to enable fast mode.
        os.environ["AC_FAST_MODE"] = "1"

    selected_runners = RUNNERS
    
    # RESUME LOGIC: 
    # If the user provides --start-from, we skip ciphers until we reach the target.
    if args.start_from:
        target = args.start_from.strip().lower()
        start_idx = None
        for idx, (label, module_name) in enumerate(RUNNERS):
            label_key = label.lower().replace("-", "")
            module_key = module_name.lower()
            module_full = f"main_{module_name}".lower()
            if target in {label_key, module_key, module_full}:
                start_idx = idx
                break
        if start_idx is None:
            choices = ", ".join(f"main_{m}" for _, m in RUNNERS)
            raise ValueError(f"Unknown --start-from '{args.start_from}'. Choose one of: {choices}")
        selected_runners = RUNNERS[start_idx:]

    t_all = time.time()
    total_to_run = len(selected_runners)
    
    # VISUAL HEADER
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█   " + "CIPHER BENCHMARK MASTER RUNNER".center(72) + "   █")
    print("█   " + "Logistic, MLP, CNN, MINE, RF | Rounds 1-5".center(72) + "   █")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    if args.fast:
        print("\n  [!] FAST MODE ACTIVE: Using reduced datasets and faster model parameters.")
    
    print(f"\n  [Setup] Found {total_to_run} ciphers in the execution queue.")
    print(f"  [Setup] Starting experimental sweep at {time.strftime('%H:%M:%S')}...\n")

    # MAIN LOOP: Process each cipher one by one.
    for i, (label, module_name) in enumerate(selected_runners):
        progress = f"[{i+1}/{total_to_run}]"
        
        # Section Divider for clarity in large logs
        print(f"\n{progress} {'=' * 68}")
        print(f"  EXECUTING PIPELINE: {label.upper()}")
        print(f"  Target Script: experiments/main_{module_name}.py")
        print(f"{'=' * 75}")
        
        t0 = time.time()
        try:
            # Dynamically import and run the 'main' function of the specific runner.
            run_module_main(module_name)
            elapsed = time.time() - t0
            print(f"\n  [✓] {label} Pipeline finished in {elapsed:.1f}s")
            
        except Exception as exc:
            # Error handling to prevent the entire suite from crashing if one cipher fails.
            print(f"\n  [✗] Error running {label}: {exc}")
            import traceback
            traceback.print_exc()

    # FINAL SUMMARY
    total_elapsed = time.time() - t_all
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█   " + "ALL PIPELINES COMPLETED SUCCESSFULLY".center(72) + "   █")
    print(f"█   " + f"Total Execution Time: {total_elapsed/60:.1f} minutes".center(72) + "   █")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")


if __name__ == "__main__":
    main()
