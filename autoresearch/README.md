# Autoresearch for Parameter Golf

Autonomous ML research loop inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Setup

```bash
cd autoresearch
uv run python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10  # if not done
```

## Files

- `train.py` — Enhanced MLX training script (SOTA techniques from top submissions)
- `autoresearch.py` — Automated experiment loop using Claude API
- `program.md` — Research instructions for the AI agent
- `results.tsv` — Experiment tracking (auto-generated)

## Architecture (26.9M params)

11-layer transformer with SOTA techniques: 3x MLP, LeakyReLU², BigramHash embeddings, Exclusive Self Attention (XSA), Partial RoPE (16/64), EMA averaging, Muon optimizer with weight decay.

## Usage

### Option A: Automated loop (autoresearch.py)

```bash
cd autoresearch
uv run python3 autoresearch.py --timeout 1200 --max-experiments 100
```

### Option B: Ralph Loop (recommended)

Use the seed prompt below with `/ralph-loop`:

---

## Ralph Loop Seed Prompt

```
STOP! Re-read all code. You are a fresh agent—free to criticize and radically change previous work. Use MCPs and web searches—traditional knowledge is stale. We can also research SOTA approaches.

You are an autonomous ML researcher running autoresearch for parameter-golf. Your goal: minimize val_bpb by iteratively modifying autoresearch/train.py.

WORKFLOW (repeat each iteration):

1. Read autoresearch/train.py and autoresearch/results.tsv (create results.tsv if missing with header: experiment\tcommit\tval_bpb\tstatus\tdescription\ttimestamp)
2. Analyze past results. Propose ONE targeted change — a single hypothesis to test.
3. Edit autoresearch/train.py with your change.
4. Run training:
   cd /Users/sc/Developer/parameter-golf/autoresearch && MAX_WALLCLOCK_SECONDS=1200 uv run python3 train.py 2>&1 | tee /tmp/train_output.txt
5. Parse val_bpb from the "final_int8_zlib_roundtrip_exact" line in the output.
6. Check the "submission_valid" line — if NO (over 16MB), this is a failure regardless of val_bpb.
7. Decision:
   - If submission_valid=NO: REVERT. Log as "size_violation".
   - If val_bpb < best_bpb AND submission_valid=YES: KEEP. Update best. Log as "kept".
   - If val_bpb >= best_bpb: REVERT with `git checkout autoresearch/train.py`. Log as "reverted".
   - If crash: REVERT. Log as "crash".
7. Append a row to autoresearch/results.tsv with: experiment_number, git_short_hash, val_bpb, status, one-line description, ISO timestamp.
8. Loop back to step 1.

RULES:
- ONE change per experiment. Isolate variables.
- Total submission (model int8+zlib + code) MUST fit under 16 MB. Check "submission_valid" and "total_submission_size" in output.
- Build on what works. Compound successes.
- Revert ALL failures immediately (including size violations).
- Prefer small, targeted modifications over rewrites.
- Balance exploration (novel ideas) and exploitation (tuning what works).
- Quantization is part of the search space — you can modify the quantization code in train.py (int6, GPTQ, better compression, etc.)
- Track everything in results.tsv.

IDEAS TO TRY (non-exhaustive):
- Hyperparameters: learning rates, momentum, warmdown length, EMA decay
- Architecture: MLP width, number of layers, head dim, attention variants
- Activations: leaky_relu slope, GELU, SwiGLU
- Positional encoding: RoPE base, partial dims fraction
- Embeddings: BigramHash buckets/dim, trigram hashing, SmearGate
- Training: weight decay schedule, gradient clipping, batch size
- Optimizer: Muon NS steps, momentum warmup, per-layer LR scaling
- Quantization: int6, GPTQ, clip percentile search, LZMA compression
- Size optimization: selective pruning, model dim tuning to fit 16MB budget

SIMPLICITY CRITERION: All else being equal, simpler is better. Tiny improvement + lots of hacky code = not worth it. Same performance + simpler code = always keep.

ENVIRONMENT: MLX on Apple M5 Max, 128GB unified memory. ~20 min per training run.

Start by running the baseline (no changes) to establish best_bpb, then begin experimenting.
```
