#!/usr/bin/env python3
"""
Autonomous ML research loop for parameter-golf on Apple Silicon.
Inspired by karpathy/autoresearch.

Usage:
    cd autoresearch/
    uv run autoresearch.py [--timeout 1200] [--max-experiments 100] [--model claude-sonnet-4-20250514]
"""

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic

TRAIN_SCRIPT = "train.py"
RESULTS_FILE = "results.tsv"
PROGRAM_FILE = "program.md"


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + list(args),
        check=check, capture_output=True, text=True,
    )


def git_commit(message: str) -> str:
    git("add", TRAIN_SCRIPT)
    git("commit", "-m", message)
    return git("rev-parse", "--short", "HEAD").stdout.strip()


def run_training(timeout: int) -> tuple[str, str, int]:
    env = os.environ.copy()
    env["MAX_WALLCLOCK_SECONDS"] = str(timeout)
    env["VAL_LOSS_EVERY"] = "0"  # only validate at end for speed
    try:
        result = subprocess.run(
            [sys.executable, TRAIN_SCRIPT],
            capture_output=True, text=True,
            timeout=timeout + 300,  # extra time for eval + quantization
            env=env,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", -1


def parse_val_bpb(output: str) -> float | None:
    """Parse val_bpb from training output, preferring final roundtrip metric."""
    for marker in ["final_int8_zlib_roundtrip_exact", "final_int8_zlib_roundtrip", "val_bpb:"]:
        for line in reversed(output.split("\n")):
            if marker in line:
                match = re.search(r"val_bpb:(\d+\.\d+)", line)
                if match:
                    return float(match.group(1))
    return None


def check_size_budget(output: str) -> tuple[bool, int | None]:
    """Check if submission fits under 16MB. Returns (valid, total_bytes)."""
    for line in output.split("\n"):
        if "SIZE_VIOLATION" in line:
            match = re.search(r"total_submission_size:(\d+)", output)
            total = int(match.group(1)) if match else None
            return False, total
    match = re.search(r"total_submission_size:(\d+)", output)
    if match:
        return int(match.group(1)) <= 16 * 1024 * 1024, int(match.group(1))
    return True, None  # no size info = assume ok


def log_result(exp_id: int, commit: str, val_bpb: float | None,
               status: str, description: str) -> None:
    header = "experiment\tcommit\tval_bpb\tstatus\tdescription\ttimestamp\n"
    if not Path(RESULTS_FILE).exists():
        Path(RESULTS_FILE).write_text(header)
    bpb = f"{val_bpb:.6f}" if val_bpb is not None else "N/A"
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{exp_id}\t{commit}\t{bpb}\t{status}\t{description}\t{datetime.now().isoformat()}\n")


def last_n_results(n: int = 20) -> str:
    if not Path(RESULTS_FILE).exists():
        return "No experiments yet."
    lines = Path(RESULTS_FILE).read_text().strip().split("\n")
    if len(lines) <= 1:
        return "No experiments yet."
    return lines[0] + "\n" + "\n".join(lines[1:][-n:])


def propose_modification(client: anthropic.Anthropic, model: str,
                         code: str, history: str, program: str) -> tuple[str | None, str]:
    prompt = f"""<program>
{program}
</program>

<current_train_py>
{code}
</current_train_py>

<experiment_history>
{history}
</experiment_history>

Based on the program instructions and experiment history, propose ONE targeted modification to train.py that you believe will lower val_bpb.

Rules:
- Make exactly ONE change (isolate your hypothesis)
- Keep it simple — prefer small, targeted modifications
- If recent experiments show a pattern, build on what works
- Explain your reasoning in 2-3 sentences before the code
- Output the COMPLETE modified train.py in a ```python code block
- After the code block, write: DESCRIPTION: <one-line summary>
"""

    response = client.messages.create(
        model=model,
        max_tokens=32000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text

    code_match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if not code_match:
        return None, "Failed to extract code from response"

    desc_match = re.search(r"DESCRIPTION:\s*(.+)", text)
    description = desc_match.group(1).strip() if desc_match else "No description"

    return code_match.group(1), description


def main():
    parser = argparse.ArgumentParser(description="Autonomous ML research loop")
    parser.add_argument("--timeout", type=int, default=1200,
                        help="Training timeout in seconds (default: 1200 = 20 min)")
    parser.add_argument("--max-experiments", type=int, default=100)
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514",
                        help="Claude model for proposing modifications")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Resume from existing results")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    program = Path(PROGRAM_FILE).read_text() if Path(PROGRAM_FILE).exists() else ""
    best_bpb: float | None = None
    start_id = 0

    # Resume from existing results
    if Path(RESULTS_FILE).exists() and args.skip_baseline:
        lines = Path(RESULTS_FILE).read_text().strip().split("\n")[1:]
        for line in lines:
            parts = line.split("\t")
            if len(parts) >= 4 and parts[3] in ("kept", "baseline") and parts[2] != "N/A":
                bpb = float(parts[2])
                if best_bpb is None or bpb < best_bpb:
                    best_bpb = bpb
        start_id = len(lines)
        print(f"Resuming from experiment {start_id}, best_bpb={best_bpb}")
    else:
        # Run baseline
        print("=" * 60)
        print("Running baseline...")
        print("=" * 60)
        stdout, stderr, rc = run_training(args.timeout)
        # Print last portion of output
        for line in stdout.split("\n")[-20:]:
            if line.strip():
                print(line)
        if rc != 0:
            print(f"Baseline FAILED!\n{stderr[-2000:]}")
            sys.exit(1)
        best_bpb = parse_val_bpb(stdout)
        if best_bpb is None:
            print("Could not parse val_bpb from baseline!")
            sys.exit(1)
        commit = git_commit("baseline run")
        log_result(0, commit, best_bpb, "baseline", "Initial baseline")
        print(f"\nBaseline val_bpb: {best_bpb:.6f}")
        start_id = 1

    # Experiment loop
    for i in range(start_id, start_id + args.max_experiments):
        print(f"\n{'=' * 60}")
        print(f"Experiment {i} | Best: {best_bpb:.6f}")
        print("=" * 60)

        code = Path(TRAIN_SCRIPT).read_text()
        history = last_n_results(20)

        print("Asking Claude for a modification...")
        try:
            new_code, description = propose_modification(
                client, args.model, code, history, program
            )
        except Exception as e:
            print(f"API error: {e}")
            log_result(i, "N/A", None, "api_error", str(e)[:100])
            time.sleep(10)
            continue

        if new_code is None:
            print(f"No code extracted: {description}")
            log_result(i, "N/A", None, "parse_error", description[:100])
            continue

        print(f"Proposal: {description}")
        original_code = code

        Path(TRAIN_SCRIPT).write_text(new_code)
        try:
            commit = git_commit(f"exp{i}: {description[:60]}")
        except subprocess.CalledProcessError:
            print("No changes or git error, skipping...")
            Path(TRAIN_SCRIPT).write_text(original_code)
            log_result(i, "N/A", None, "skipped", "No diff")
            continue

        print(f"Training ({args.timeout}s budget)...")
        t0 = time.time()
        stdout, stderr, rc = run_training(args.timeout)
        elapsed = time.time() - t0
        print(f"Finished in {elapsed:.0f}s (rc={rc})")

        # Print key output lines
        for line in stdout.split("\n"):
            if any(k in line for k in ["val_bpb:", "stopping_early", "final_int8", "total_submission", "SIZE_VIOLATION", "submission_valid"]):
                print(f"  {line.strip()}")

        if rc != 0:
            print(f"CRASH! Reverting.")
            err_lines = stderr.strip().split("\n")[-5:]
            for l in err_lines:
                print(f"  stderr: {l}")
            Path(TRAIN_SCRIPT).write_text(original_code)
            git_commit(f"revert exp{i}: crash")
            log_result(i, commit, None, "crash", f"CRASH: {description[:80]}")
            continue

        val_bpb = parse_val_bpb(stdout)
        if val_bpb is None:
            print("No val_bpb found! Reverting.")
            Path(TRAIN_SCRIPT).write_text(original_code)
            git_commit(f"revert exp{i}: no metric")
            log_result(i, commit, None, "error", description[:80])
            continue

        # Check 16MB size budget
        size_ok, total_bytes = check_size_budget(stdout)
        size_mb = f"{total_bytes / 1e6:.2f}MB" if total_bytes else "unknown"
        if not size_ok:
            print(f"SIZE VIOLATION ({size_mb} > 16MB)! Reverting.")
            Path(TRAIN_SCRIPT).write_text(original_code)
            git_commit(f"revert exp{i}: over 16MB")
            log_result(i, commit, val_bpb, "size_violation", f">{size_mb}: {description[:60]}")
            continue

        print(f"val_bpb: {val_bpb:.6f} (best: {best_bpb:.6f}) size: {size_mb}")

        if val_bpb < best_bpb:
            delta = best_bpb - val_bpb
            print(f"*** IMPROVEMENT: -{delta:.6f} *** Keeping!")
            best_bpb = val_bpb
            log_result(i, commit, val_bpb, "kept", description)
        else:
            delta = val_bpb - best_bpb
            reason = "same" if delta < 1e-6 else f"+{delta:.6f}"
            print(f"No improvement ({reason}). Reverting.")
            Path(TRAIN_SCRIPT).write_text(original_code)
            git_commit(f"revert exp{i}: {reason}")
            log_result(i, commit, val_bpb, "reverted", description)

    print(f"\nDone! Best val_bpb: {best_bpb:.6f}")
    print(f"Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
