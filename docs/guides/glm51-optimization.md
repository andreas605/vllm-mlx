# GLM-5.1 Performance Optimization on Mac Studio M3 Ultra

This guide covers four optimizations that together reduce time-to-first-token (TTFT) by 60-80% for long-context GLM-5.1 workloads on a Mac Studio M3 Ultra (512GB unified memory).

## SpecPrefill for Long Context

**What it does.** SpecPrefill uses a small draft model to score token importance across the input, then sparse-prefills the target model with only the important tokens. Unimportant tokens are skipped, cutting the prefill compute roughly in proportion to the keep fraction.

**When to use it.** Enable SpecPrefill when prompts regularly exceed ~8k tokens. Agentic coding workflows — where the system prompt carries a codebase snapshot or long tool schema — are the primary target. Shorter prompts pay the draft model scoring overhead without enough savings to break even.

**CLI flags.**

| Flag | Default | Description |
|------|---------|-------------|
| `--specprefill` | off | Enable SpecPrefill |
| `--specprefill-draft-model` | — | Path to a small draft model for importance scoring |
| `--specprefill-threshold` | 8192 | Minimum suffix tokens required to trigger sparse prefill |
| `--specprefill-keep-pct` | 0.3 | Fraction of tokens retained (lower = faster, more quality loss) |

**Recommended settings for GLM-5.1.**

```
--specprefill
--specprefill-draft-model mlx-community/GLM-4-Flash-4bit
--specprefill-threshold 8192
--specprefill-keep-pct 0.3
```

The draft model must share the same tokenizer as the target. `GLM-4-Flash-4bit` (or any similarly compact GLM-family 4-bit model) is a good choice because its vocabulary is identical to GLM-5.1.

**Expected impact.** 60-80% TTFT reduction on prompts longer than 8k tokens. The scoring pass on the draft model adds ~1-2s of latency, so prompts shorter than the threshold fall back to normal prefill automatically.

---

## System KV Cache Reuse

**What it does.** After the first request, the engine snapshots the KV state produced by prefilling the system prompt. Subsequent turns with the same system prompt restore that snapshot and prefill only the new user suffix tokens.

**When it helps.** Any agentic workflow that opens each turn with the same system prompt — tool definitions, repository context, coding instructions — benefits. The cache invalidates whenever the system prompt text changes.

**How to enable it.** Nothing to configure. System KV caching is automatic when using the chat API (`/v1/chat/completions`). On the first request the engine logs a `MISS` and stores the snapshot; subsequent requests with the same system prefix log a `HIT`.

The cache is disabled on sliding-window models (those returning `RotatingKVCache` entries). GLM-5.1 uses standard attention and is fully compatible.

**Expected impact.** Near-instant TTFT on turns 2+ when the system prompt is unchanged. The suffix tokens (typically a short user message) are the only tokens prefilled.

---

## 8-bit KV Cache Quantization

**What it does.** Stored KV pairs are quantized from FP16 to INT8 before being written to the prefix cache. Values are dequantized on read. This trades a small rounding error for an ~8x reduction in cache memory.

**CLI flags.**

| Flag | Default | Description |
|------|---------|-------------|
| `--kv-cache-quantization` | off | Enable quantization |
| `--kv-cache-quantization-bits` | 8 | Bit width: 4 or 8 |
| `--kv-cache-quantization-group-size` | 64 | Quantization group size |
| `--kv-cache-min-quantize-tokens` | 256 | Skip quantization for sequences shorter than this |

**When to use it.** Quantization is most valuable when running many concurrent conversations or when individual contexts are very long (100k+ tokens). On a 512GB system with moderate concurrency you may not need it, but it provides headroom when the cache fills under load. Requires `--continuous-batching`.

**Expected impact.** ~8x cache capacity at 8-bit. Quality degradation is minimal on 8-bit; 4-bit is available for maximum compression if that tradeoff is acceptable.

---

## Prefill Chunk Size Tuning

**What it does.** During prefill, the engine slices the prompt into fixed-size chunks and calls the model layer-by-layer on each chunk. Smaller chunks fit better in L2 cache; larger chunks reduce the number of graph dispatches. The default of 2048 is a reasonable middle ground, but models with non-standard hidden dimensions may respond differently.

**CLI flag.**

```
--prefill-step-size 2048
```

**Guidance for GLM-5.1.** Start with the default. If you want to profile: try 1024, 2048, and 4096 with `vllm-mlx bench-serve` on a long-context workload. TTFT is the signal to optimize; throughput on short prompts is not sensitive to this setting.

---

## Full Example Command

Serving GLM-5.1 on Mac Studio M3 Ultra with all optimizations enabled:

```bash
vllm-mlx serve mlx-community/GLM-5.1-32B-A16B-Instruct-4bit \
  --host 127.0.0.1 \
  --port 8000 \
  --max-tokens 32768 \
  --max-request-tokens 32768 \
  --prefill-step-size 2048 \
  --specprefill \
  --specprefill-draft-model mlx-community/GLM-4-Flash-4bit \
  --specprefill-threshold 8192 \
  --specprefill-keep-pct 0.3 \
  --continuous-batching \
  --kv-cache-quantization \
  --kv-cache-quantization-bits 8 \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm4 \
  --gpu-memory-utilization 0.95
```

Notes on this command:

- `--gpu-memory-utilization 0.95` gives the model maximum headroom on a 512GB system. Lower this to 0.90 if you observe Metal allocation failures.
- `--continuous-batching` is required for `--kv-cache-quantization`. Drop it if you need single-user maximum throughput without quantization.
- `--tool-call-parser glm47` handles GLM-4.7-style `<tool_call>` XML with `<arg_key>`/`<arg_value>` tags. Use this for GLM-5.1 if its tool format matches GLM-4.7; verify by inspecting a raw response.
- `--reasoning-parser glm4` extracts `<think>...</think>` blocks into a separate `reasoning` field in the API response.
- System KV cache reuse is active automatically once the server is running; no additional flag is needed.
