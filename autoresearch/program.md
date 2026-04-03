# Autonomous ML Research Program — Parameter Golf (MLX)

## Objective
Minimize `val_bpb` (bits per byte) on the FineWeb validation set by modifying `train.py`.

## Constraints
- Single Apple M5 Max with 128GB unified memory (MLX framework)
- ~20 minute training budget per experiment
- **Total submission (model + code) must fit under 16 MB** — check `total_submission_size` and `submission_valid` in output
- Model is a causal language model evaluated on FineWeb validation
- Changes must be to `train.py` only
- The script must remain runnable: `uv run train.py`
- Quantization strategy is part of the search space — you can modify the quantization code too

## Current Architecture
- 11 transformer layers, 512 model dim, 3x MLP expansion
- GQA: 8 query heads, 4 KV heads, 64 head dim
- LeakyReLU² activation, Partial RoPE (16/64 dims)
- XSA (Exclusive Self Attention) on all layers
- BigramHash embeddings (3072 buckets, dim 112)
- U-Net skip connections (encoder-decoder with learned skip weights)
- Muon optimizer (matrices) + Adam (embeddings, scalars)
- EMA weight averaging (decay 0.997)
- Tied embeddings, vocab 1024, logit softcap 30

## Research Strategy
1. Make **ONE** change per experiment — isolate variables
2. Prefer small, targeted modifications over rewrites
3. Track what works and compound successes
4. Balance exploration (novel ideas) and exploitation (tuning what works)
5. If an idea fails, try a different angle — don't repeat the same change

## Ideas to Explore (non-exhaustive)
- **Hyperparameters**: learning rates (embed, matrix, scalar), momentum, warmdown length, batch size
- **Architecture**: attention patterns, MLP width/depth ratio, number of layers, head dim
- **Activations**: leaky_relu slope, GELU, SwiGLU, different squaring approaches
- **Positional encoding**: RoPE base frequency, partial dims fraction, YaRN scaling
- **Embeddings**: BigramHash buckets/dim, trigram hashing, SmearGate
- **Attention**: XSA variants, value residual connections, sliding window
- **Training**: EMA decay rate, weight decay schedule, gradient clipping, warmup length
- **Optimizer**: Muon Newton-Schulz steps, momentum warmup schedule, per-layer LR
- **Initialization**: orthogonal init, spectral init, scaled init
- **Normalization**: layer-wise RMSNorm scaling (1/sqrt(layer+1))
- **Regularization**: dropout, stochastic depth, z-loss on logits
- **Sequence length**: longer context (2048), curriculum on sequence length
- **Quantization**: int6 per-row, mixed int6/int8, GPTQ, clip percentile search, better compression (LZMA)
- **Size optimization**: selective pruning of ±1 values, model dimension tuning to fit budget

## Simplicity Criterion
All else being equal, simpler is better:
- Tiny improvement + lots of hacky code → not worth it
- Same performance + simpler code → always keep
- Small improvement from one clean change → definitely keep

## Output Format
Output the COMPLETE modified `train.py` in a ```python code block.
After the code block, write: `DESCRIPTION: <one-line summary of what you changed>`
