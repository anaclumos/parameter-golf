#!/usr/bin/env python3
"""
Enhanced MLX training script for parameter-golf autoresearch.
Starting point: baseline train_gpt_mlx.py + SOTA techniques from top submissions.

Key additions over baseline:
- 11 layers, 3x MLP expansion
- LeakyReLU² activation
- BigramHash embeddings (3072 buckets, dim 112)
- Exclusive Self Attention (XSA) on all layers
- Partial RoPE (16/64 dims)
- EMA weight averaging
- Muon weight decay
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time
import uuid
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

COMPUTE_DTYPE = mx.bfloat16

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================

class Hyperparameters:
    data_path: str = os.environ.get("DATA_PATH", "../data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "../data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 65_536))
    mlx_eager_eval: bool = bool(int(os.environ.get("MLX_EAGER_EVAL", "0")))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 3500))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 1200.0))

    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 11))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_chunk_tokens: int = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # SOTA additions
    leaky_relu_slope: float = float(os.environ.get("LEAKY_RELU_SLOPE", 0.5))
    rope_partial_dims: int = int(os.environ.get("ROPE_PARTIAL_DIMS", 16))
    xsa_enabled: bool = bool(int(os.environ.get("XSA_ENABLED", "1")))
    bigram_vocab_size: int = int(os.environ.get("BIGRAM_VOCAB_SIZE", 3072))
    bigram_dim: int = int(os.environ.get("BIGRAM_DIM", 112))
    bigram_enabled: bool = bool(int(os.environ.get("BIGRAM_ENABLED", "1")))
    ema_decay: float = float(os.environ.get("EMA_DECAY", 0.997))

    # Optimizer
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.035))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_wd: float = float(os.environ.get("MUON_WD", 0.04))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    out_dir: str = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        if self.warmdown_iters <= 0:
            return 1.0
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            if warmdown_start <= step < self.iterations:
                return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",") if p
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",") if p
)

# ==============================================================================
# HELPERS
# ==============================================================================

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)

def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)

def token_chunks(total_tokens: int, seq_len: int, max_chunk_tokens: int) -> list[int]:
    usable_total = (total_tokens // seq_len) * seq_len
    if usable_total <= 0:
        raise ValueError(f"token budget too small for seq_len={seq_len}")
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks: list[int] = []
    remaining = usable_total
    while remaining > 0:
        chunks.append(min(remaining, usable_chunk))
        remaining -= chunks[-1]
    return chunks

def accumulate_flat_grads(accum: dict[str, mx.array] | None, grads_tree: dict, scale: float) -> dict[str, mx.array]:
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * np.dtype("<u2").itemsize:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)

class TokenStream:
    def __init__(self, pattern: str, log_fn: Callable[[str], None] | None = None, dataset_name: str = ""):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch, self.file_idx, self.pos = 1, 0, 0
        self.log_fn, self.dataset_name = log_fn, dataset_name
        self.tokens = load_data_shard(self.files[0])

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn:
                self.log_fn(f"WARNING: starting epoch:{self.epoch} dataset:{self.dataset_name}")
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)

class TokenLoader:
    def __init__(self, pattern: str, log_fn: Callable[[str], None] | None = None, dataset_name: str = ""):
        self.stream = TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable + 1)
        return mx.array(chunk[:-1].reshape(-1, seq_len), dtype=mx.int32), mx.array(chunk[1:].reshape(-1, seq_len), dtype=mx.int32)

# ==============================================================================
# MODEL
# ==============================================================================

class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)
    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T

class RMSNormNoWeight(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)

class BigramHashEmbedding(nn.Module):
    def __init__(self, num_buckets: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.embed = nn.Embedding(num_buckets, bigram_dim)
        self.proj = CastedLinear(bigram_dim, model_dim)

    def __call__(self, token_ids: mx.array) -> mx.array:
        B, T = token_ids.shape
        prev = mx.concatenate([mx.zeros((B, 1), dtype=mx.int32), token_ids[:, :-1]], axis=1)
        h = (prev * 92821 + token_ids) % self.num_buckets
        return self.proj(self.embed(h))

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float, rope_partial_dims: int, xsa_enabled: bool):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.xsa_enabled = xsa_enabled
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(rope_partial_dims, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")

        if self.xsa_enabled:
            B, H, T, D = y.shape
            gs = H // self.num_kv_heads
            y_g = y.reshape(B, self.num_kv_heads, gs, T, D)
            v_e = v[:, :, None, :, :]
            vn = v_e * mx.rsqrt(mx.sum(v_e * v_e, axis=-1, keepdims=True) + 1e-6)
            dot = mx.sum(y_g * vn, axis=-1, keepdims=True)
            y = (y_g - dot * vn).reshape(B, H, T, D)

        return self.proj(y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim))

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, leaky_relu_slope: float):
        super().__init__()
        self.fc = CastedLinear(dim, dim * mlp_mult)
        self.proj = CastedLinear(dim * mlp_mult, dim)
        self.slope = leaky_relu_slope

    def __call__(self, x: mx.array) -> mx.array:
        h = self.fc(x)
        h = mx.where(h > 0, h, self.slope * h)
        return self.proj(h * h)

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, rope_partial_dims: int,
                 xsa_enabled: bool, leaky_relu_slope: float):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base,
                                         qk_gain_init, rope_partial_dims, xsa_enabled)
        self.mlp = MLP(dim, mlp_mult, leaky_relu_slope)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((np.ones(dim, dtype=np.float32), np.zeros(dim, dtype=np.float32))))

    def __call__(self, x: mx.array, x0: mx.array) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, dim: int, num_heads: int,
                 num_kv_heads: int, mlp_mult: int, logit_chunk_tokens: int,
                 logit_softcap: float, rope_base: float, tied_embed_init_std: float,
                 qk_gain_init: float, rope_partial_dims: int, xsa_enabled: bool,
                 leaky_relu_slope: float, bigram_enabled: bool, bigram_vocab_size: int,
                 bigram_dim: int):
        super().__init__()
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.bigram_enabled = bigram_enabled
        if bigram_enabled:
            self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base,
                  qk_gain_init, rope_partial_dims, xsa_enabled, leaky_relu_slope)
            for _ in range(num_layers)
        ]
        self.final_norm = RMSNormNoWeight()
        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std).astype(COMPUTE_DTYPE)

    def softcap(self, logits: mx.array) -> mx.array:
        return self.logit_softcap * mx.tanh(logits / self.logit_softcap)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = self.tok_emb(input_ids).astype(COMPUTE_DTYPE)
        if self.bigram_enabled:
            x = x + self.bigram(input_ids).astype(COMPUTE_DTYPE)
        x = rms_norm(x)
        x0 = x
        skips: list[mx.array] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits = self.softcap(x @ self.tok_emb.weight.astype(x.dtype).T)
            return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")
        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits = self.softcap(x[s:e] @ self.tok_emb.weight.astype(x.dtype).T)
            loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="sum")
        return loss_sum / float(n)

# ==============================================================================
# OPTIMIZERS
# ==============================================================================

class Muon:
    def __init__(self, keys: list[str], params: dict[str, mx.array], args: Hyperparameters):
        self.keys, self.args = keys, args
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(self, params: dict[str, mx.array], grads: dict[str, mx.array], step: int, lr_mul: float) -> dict[str, mx.array]:
        t = min(step / self.args.muon_momentum_warmup_steps, 1.0) if self.args.muon_momentum_warmup_steps else 1.0
        momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        wd = self.args.muon_wd
        out: dict[str, mx.array] = {}
        for k in self.keys:
            p, g = params[k], grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_ortho = zeropower_newtonschulz5(g + momentum * buf, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            out[k] = p * (1.0 - lr * wd) - lr * (g_ortho * scale).astype(p.dtype)
        return out

class SplitOptimizers:
    def __init__(self, model: GPT, args: Hyperparameters):
        self.args = args
        params = dict(tree_flatten(model.parameters()))
        self.embed_key = "tok_emb.weight"
        self.matrix_keys = [k for k, p in params.items()
                            if p.ndim == 2 and k != self.embed_key
                            and not any(pat in k for pat in CONTROL_TENSOR_NAME_PATTERNS)]
        self.scalar_keys = [k for k, p in params.items()
                            if k == "skip_weights"
                            or (k.startswith("blocks.") and (p.ndim < 2 or any(pat in k for pat in CONTROL_TENSOR_NAME_PATTERNS)))]
        self.muon = Muon(self.matrix_keys, params, args)
        self.adam_embed = optim.Adam(learning_rate=args.tied_embed_lr, betas=[args.beta1, args.beta2], eps=args.adam_eps, bias_correction=True)
        self.adam_scalar = optim.Adam(learning_rate=args.scalar_lr, betas=[args.beta1, args.beta2], eps=args.adam_eps, bias_correction=True)

    def step(self, model: GPT, grads_tree: dict, step: int, lr_mul: float) -> None:
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)
        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))
        self.adam_embed.learning_rate = self.args.tied_embed_lr * lr_mul
        updated.update(self.adam_embed.apply_gradients({self.embed_key: grads[self.embed_key]}, {self.embed_key: params[self.embed_key]}))
        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        sg = {k: grads[k] for k in self.scalar_keys if k in grads}
        sp = {k: params[k] for k in self.scalar_keys if k in params}
        if sg:
            updated.update(self.adam_scalar.apply_gradients(sg, sp))
        model.update(tree_unflatten(list(updated.items())))

# ==============================================================================
# QUANTIZATION (INT8 + ZLIB) — uses numpy serialization instead of pickle
# ==============================================================================

MX_DTYPE_FROM_NAME = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
INT8_CLIP_Q = 99.99984 / 100.0

def _np_float32(arr: mx.array) -> np.ndarray:
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)

def keep_float_array(name: str, arr: mx.array, passthrough_orig_dtypes: dict[str, str]) -> np.ndarray:
    if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(np.array(arr, copy=True))

def quantize_float_array(arr: mx.array) -> tuple[np.ndarray, np.ndarray]:
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE))
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8)
    return np.ascontiguousarray(q), scale

def quantize_state_dict_int8(flat_state: dict[str, mx.array]) -> tuple[dict, dict]:
    quantized, scales, dtypes = {}, {}, {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict] = {}
    stats = dict.fromkeys(("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"), 0)
    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if not mx.issubdtype(arr.dtype, mx.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue
        if int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_array(arr)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name], scales[name] = q, s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)
    obj = {"__quant_format__": "int8_clean_per_row_v1", "quantized": quantized, "scales": scales,
           "dtypes": dtypes, "passthrough": passthrough}
    if qmeta: obj["qmeta"] = qmeta
    if passthrough_orig_dtypes: obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(quant_obj: dict) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            out_arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np.astype(np.float32) * float(scale)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[quant_obj["dtypes"][name]])
    for name, arr in quant_obj["passthrough"].items():
        out_arr = np.array(arr, copy=True)
        orig_dtype = passthrough_orig_dtypes.get(name)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[orig_dtype]) if isinstance(orig_dtype, str) else mx.array(out_arr)
    return out

def serialize_quant_obj(obj: dict, path: Path) -> int:
    """Serialize quantized model using numpy .npz + zlib instead of pickle."""
    import io, struct
    buf = io.BytesIO()
    # Simple format: JSON metadata + concatenated numpy arrays
    arrays = {}
    meta = {"__quant_format__": obj["__quant_format__"], "dtypes": obj["dtypes"]}
    if "qmeta" in obj: meta["qmeta"] = obj["qmeta"]
    if "passthrough_orig_dtypes" in obj: meta["passthrough_orig_dtypes"] = obj["passthrough_orig_dtypes"]
    for name, arr in obj["quantized"].items():
        arrays[f"q/{name}"] = np.asarray(arr)
    for name, arr in obj["scales"].items():
        arrays[f"s/{name}"] = np.asarray(arr)
    for name, arr in obj["passthrough"].items():
        arrays[f"p/{name}"] = np.asarray(arr)
    meta_bytes = json.dumps(meta).encode("utf-8")
    np_buf = io.BytesIO()
    np.savez(np_buf, **arrays)
    np_bytes = np_buf.getvalue()
    # Pack: [4 bytes meta_len][meta_json][npz_data]
    raw = struct.pack("<I", len(meta_bytes)) + meta_bytes + np_bytes
    compressed = zlib.compress(raw, level=9)
    with open(path, "wb") as f:
        f.write(compressed)
    return len(compressed)

def deserialize_quant_obj(path: Path) -> dict:
    """Deserialize quantized model from our custom format."""
    import io, struct
    with open(path, "rb") as f:
        compressed = f.read()
    raw = zlib.decompress(compressed)
    meta_len = struct.unpack("<I", raw[:4])[0]
    meta = json.loads(raw[4:4+meta_len].decode("utf-8"))
    np_data = np.load(io.BytesIO(raw[4+meta_len:]))
    quantized, scales, passthrough = {}, {}, {}
    for key in np_data.files:
        prefix, name = key.split("/", 1)
        if prefix == "q": quantized[name] = np_data[key]
        elif prefix == "s": scales[name] = np_data[key]
        elif prefix == "p": passthrough[name] = np_data[key]
    obj = {"__quant_format__": meta["__quant_format__"], "quantized": quantized,
           "scales": scales, "dtypes": meta["dtypes"], "passthrough": passthrough}
    if "qmeta" in meta: obj["qmeta"] = meta["qmeta"]
    if "passthrough_orig_dtypes" in meta: obj["passthrough_orig_dtypes"] = meta["passthrough_orig_dtypes"]
    return obj

# ==============================================================================
# VALIDATION
# ==============================================================================

def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes = np.zeros((table_size,), dtype=np.int16)
    has_space = np.zeros((table_size,), dtype=np.bool_)
    is_boundary = np.ones((table_size,), dtype=np.bool_)
    for tid in range(sp_vocab_size):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            has_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return base_bytes, has_space, is_boundary

def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str):
    dataset_dir = Path(data_path).resolve()
    actual = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if entry is None:
        return dataset_dir.name, actual, None
    tok_name = entry.get("tokenizer_name")
    tok_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tok_name), None) if tok_name else None
    expected_name = Path((tok_entry or {}).get("model_path") or (tok_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")
    expected = (entry.get("stats") or {}).get("files_train")
    if expected is not None:
        expected = int(expected)
        if actual > expected:
            raise ValueError(f"Too many train shards: {actual} > {expected}")
    return dataset_dir.name, actual, expected

def load_validation_tokens(pattern: str, seq_len: int) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(f) for f in files], axis=0))
    usable = ((tokens.size - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for seq_len={seq_len}")
    return tokens[: usable + 1]

def eval_val(args, compiled_loss, val_tokens, base_bytes, has_space, is_boundary, log_fn=None):
    vbt = args.val_batch_size // args.grad_accum_steps
    vbs = vbt // args.train_seq_len
    total_seqs = (val_tokens.size - 1) // args.train_seq_len
    total_batches = max((total_seqs + vbs - 1) // vbs, 1)
    loss_sum, tok_count, byte_count = 0.0, 0.0, 0.0
    for bi, bss in enumerate(range(0, total_seqs, vbs), 1):
        bse = min(bss + vbs, total_seqs)
        rs, re = bss * args.train_seq_len, bse * args.train_seq_len + 1
        chunk = val_tokens[rs:re]
        xn, yn = chunk[:-1].reshape(-1, args.train_seq_len), chunk[1:].reshape(-1, args.train_seq_len)
        x, y = mx.array(xn, dtype=mx.int32), mx.array(yn, dtype=mx.int32)
        bl = compiled_loss(x, y).astype(mx.float32)
        mx.synchronize()
        tc = float(y.size)
        loss_sum += float(bl.item()) * tc
        prev, tgt = xn.reshape(-1), yn.reshape(-1)
        bts = base_bytes[tgt].astype(np.int16, copy=True)
        bts += (has_space[tgt] & ~is_boundary[prev]).astype(np.int16)
        tok_count += tc
        byte_count += float(bts.astype(np.float64).sum())
        if log_fn and total_batches > 1 and (bi == 1 or bi == total_batches or bi % 25 == 0):
            log_fn(f"val_progress:{bi}/{total_batches}")
    vl = loss_sum / tok_count
    return vl, vl / math.log(2.0) * (tok_count / byte_count)

# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad):
    chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    for ct in chunk_sizes:
        x, y = train_loader.next_batch(ct, args.train_seq_len)
        loss, grads = compiled_loss_and_grad(x, y)
        s = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * s
        grad_accum = accumulate_flat_grads(grad_accum, grads, s)
        if args.mlx_eager_eval:
            mx.synchronize()
    return loss_value, tree_unflatten(list(grad_accum.items()))

def clip_grad_tree(grads_tree: dict, max_norm: float) -> dict:
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    tsq = sum(float(np.sum(np.square(_np_float32(g)), dtype=np.float64)) for g in flat.values())
    if tsq <= 0.0:
        return grads_tree
    tn = math.sqrt(tsq)
    if tn <= max_norm:
        return grads_tree
    s = max_norm / (tn + 1e-12)
    return tree_unflatten([(k, g * s) for k, g in flat.items()])

def main() -> None:
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    print(logfile)

    def log(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    log(code, console=False)
    log("=" * 100, console=False)
    log(f"Python {sys.version}", console=False)
    log(f"MLX {mx.__version__}", console=False)
    log("=" * 100, console=False)

    if not args.tie_embeddings:
        raise NotImplementedError("Only tied embeddings supported")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} != tokenizer={int(sp.vocab_size())}")
    dataset_name, actual_train_files, expected = validate_dataset_tokenizer_pair(args.data_path, args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes, has_space, is_boundary = build_sentencepiece_luts(sp, args.vocab_size)

    mx.random.seed(args.seed)
    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init, rope_partial_dims=args.rope_partial_dims,
        xsa_enabled=args.xsa_enabled, leaky_relu_slope=args.leaky_relu_slope,
        bigram_enabled=args.bigram_enabled, bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
    )
    opt = SplitOptimizers(model, args)

    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state, outputs=model.state,
    )

    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    log(f"model_params:{n_params}")
    log(f"layers:{args.num_layers} dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads}")
    log(f"mlp_mult:{args.mlp_mult} xsa:{args.xsa_enabled} partial_rope:{args.rope_partial_dims}")
    log(f"bigram:{args.bigram_enabled} ema_decay:{args.ema_decay} muon_wd:{args.muon_wd}")
    log(f"leaky_relu_slope:{args.leaky_relu_slope}")
    log(f"train_batch:{args.train_batch_tokens} seq_len:{args.train_seq_len} iters:{args.iterations}")
    log(f"warmdown:{args.warmdown_iters} max_wall:{args.max_wallclock_seconds}")
    log(f"embed_lr:{args.tied_embed_lr} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}")
    log(f"dataset:{dataset_name} shards:{actual_train_files} val_tokens:{val_tokens.size - 1}")

    # EMA state
    ema_params: dict[str, mx.array] = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}

    # Warmup (compile priming, no weight updates)
    if args.warmup_steps > 0:
        for ws in range(args.warmup_steps):
            acc: dict[str, mx.array] | None = None
            gs = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                _, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
                acc = accumulate_flat_grads(acc, grads, gs)
            mx.synchronize()
            if args.warmup_steps <= 20 or (ws + 1) % 10 == 0:
                log(f"warmup_step:{ws + 1}/{args.warmup_steps}")
        # Prime eval
        vbt = args.val_batch_size // args.grad_accum_steps
        wvs = min(vbt // args.train_seq_len, (val_tokens.size - 1) // args.train_seq_len)
        wc = val_tokens[: wvs * args.train_seq_len + 1]
        compiled_loss(mx.array(wc[:-1].reshape(-1, args.train_seq_len), dtype=mx.int32),
                      mx.array(wc[1:].reshape(-1, args.train_seq_len), dtype=mx.int32))
        mx.synchronize()
        train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    train_time_ms = 0.0
    max_wc_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after: int | None = None
    t0 = time.perf_counter()
    step = 0

    while True:
        last = step == args.iterations or (stop_after is not None and step >= stop_after)
        if last or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            cur = dict(tree_flatten(model.parameters()))
            model.update(tree_unflatten(list(ema_params.items())))
            vl, vb = eval_val(args, compiled_loss, val_tokens, base_bytes, has_space, is_boundary, log)
            model.update(tree_unflatten(list(cur.items())))
            log(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} "
                f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms")
            t0 = time.perf_counter()
        if last:
            if stop_after is not None and step < args.iterations:
                log(f"stopping_early: wallclock_cap train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        lm = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        st0 = time.perf_counter()
        acc: dict[str, mx.array] | None = None
        tl = mx.array(0.0, dtype=mx.float32)
        gs = 1.0 / args.grad_accum_steps
        for _ in range(args.grad_accum_steps):
            loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
            acc = accumulate_flat_grads(acc, grads, gs)
            tl = tl + loss.astype(mx.float32) * gs
            if args.mlx_eager_eval:
                mx.synchronize()
        grads = tree_unflatten(list(acc.items()))
        grads = clip_grad_tree(grads, args.grad_clip_norm)
        tlv = float(tl.item())
        opt.step(model, grads, step=step, lr_mul=lm)
        mx.synchronize()

        # EMA update
        d = args.ema_decay
        for k, v in tree_flatten(model.parameters()):
            ema_params[k] = d * ema_params[k] + (1.0 - d) * v
        mx.synchronize()

        sms = 1000.0 * (time.perf_counter() - st0)
        atm = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after is not None):
            log(f"step:{step}/{args.iterations} train_loss:{tlv:.4f} train_time:{atm:.0f}ms "
                f"step_avg:{atm / step:.2f}ms tok_s:{args.train_batch_tokens / (sms / 1000.0):.0f}")
        if max_wc_ms is not None and stop_after is None and atm >= max_wc_ms:
            stop_after = step

    # Save with EMA weights
    model.update(tree_unflatten(list(ema_params.items())))
    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    flat_state = {k: v for k, v in tree_flatten(model.state)}
    mx.savez(str(out_path), **flat_state)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")

    quant_obj, quant_stats = quantize_state_dict_int8(flat_state)
    quant_path = out_dir / f"{args.run_id}_mlx_model.int8.ptz"
    quant_file_bytes = serialize_quant_obj(quant_obj, quant_path)
    ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
    log(f"serialized_model_int8_zlib:{quant_file_bytes} bytes (payload:{quant_stats['int8_payload_bytes']} ratio:{ratio:.2f}x)")

    # Roundtrip validation
    quant_flat = dequantize_state_dict_int8(deserialize_quant_obj(quant_path))
    model.update(tree_unflatten(list(quant_flat.items())))
    qt0 = time.perf_counter()
    qvl, qvb = eval_val(args, compiled_loss, val_tokens, base_bytes, has_space, is_boundary, log)
    qms = 1000.0 * (time.perf_counter() - qt0)
    log(f"final_int8_zlib_roundtrip val_loss:{qvl:.4f} val_bpb:{qvb:.4f} eval_time:{qms:.0f}ms")
    log(f"final_int8_zlib_roundtrip_exact val_loss:{qvl:.8f} val_bpb:{qvb:.8f}")

if __name__ == "__main__":
    main()
