# Fast-Path Flags & Dynamic Batching

This repo now has a gated padded-attention path for inference plus a lightweight request batcher. All flags default to **off** so the original behavior is unchanged.

## New flags
- `use_padded_attention` (GPTConfig): enables key padding masks + per-row rotary positions. Default `False`.
- `enable_batch_decode` (Engine / `scripts/chat_web.py`): turns on padded attention and the batched decode path. Default `False`.
- `--enable-dynamic-batching` (chat_web): queues requests and forms batches up to `--batch-size` or `--batch-timeout-ms`. Defaults off; the server keeps the single-request path when disabled.

## Usage
- Programmatic: instantiate `Engine(model, tokenizer, enable_batch_decode=True)` to use `generate_batched(prompts, max_tokens=..., temperature=..., top_k=...)` on padded multi-prompt batches. Per-request temperatures/top-k/max-tokens are accepted as scalars or lists.
- Web server: 
  ```bash
  PYTHONPATH=labs/nanochat \
  python -m scripts.chat_web \
    --enable-batch-decode \
    --enable-dynamic-batching \
    --batch-size 4 \
    --batch-timeout-ms 10 \
    --temperature 0.8 --top-k 50 --max-tokens 512
  ```
  With all flags off, chat_web runs the original single-request decode path.

## Benchmarks
- Target: prefill + decode tokens/sec with `enable_batch_decode` on vs off using a realistic checkpoint and mixed-length prompts.
- How to run: `python -m scripts.chat_web --enable-batch-decode --enable-dynamic-batching --batch-size 4 --batch-timeout-ms 10 ...` and measure end-to-end throughput on your checkpoint; repeat with the flags off for a baseline.
- What to record: prefill tok/s, decode tok/s, and end-to-end request latency for a mixed-length prompt batch. Single-request latency should stay within noise because the flags default to off.
- Quick local smoke (NVIDIA B200, `karpathy/nanochat-d32` step 650, bf16 autocast, 8 mixed-length prompts, `max_new_tokens=64`, `temperature=0.0`, `top_k=50`):
  - Single-path (`enable_batch_decode` off): 512 new tokens in 8.25s → ~62 tok/s.
  - Batched decode (`enable_batch_decode` on): 512 new tokens in 2.57s → ~199 tok/s (**~3.2x faster decode throughput** vs single-path).
  - Reuse-ids buffer: no meaningful uplift in this micro-run (~199 tok/s without reuse vs ~175 tok/s with reuse).
- Benchmark (real checkpoint, GPU): NVIDIA B200, `karpathy/nanochat-d32` step 650, 8 mixed-length prompts (Matrix summary, EV pros/cons, code/email, etc.), `max_new_tokens=64`, `temperature=0.0`, `top_k=50`, 5 runs:
  - Baseline (flags off, single-path): p50 6.55s, p90 6.55s, mean 6.61s; ~77.4 tok/s.
  - + `enable_batch_decode` (padded attention + batched decode): p50 2.61s, p90 2.61s, mean 2.63s; ~194.5 tok/s; **+151% throughput** and **~2.5x lower p50/p90 latency** vs baseline.
  - Dynamic batching in `chat_web` will use the batched path once batches form (`--enable-dynamic-batching --batch-size ... --batch-timeout-ms ...`); expect similar gains when your request mix allows batching.
  - Replace with your own numbers if you run on different hardware/checkpoints; the key is the delta between flags-off and flags-on.

## Sanity checks
- Engine KV-cache resize test: `PYTHONPATH=labs/nanochat python -m pytest tests/test_engine.py -q`.
- Manual smoke: default (unpadded) forward with loss and padded-attention forward with an attention mask both execute on small random inputs.
- Single-request inference remains the default path unless `--enable-batch-decode` is provided.
- Training path: unchanged. A quick micro-bench on the d32 checkpoint (B200, bf16 autocast, batch=1, seq=512, 5 train steps with backward) runs at ~3.4K tok/s. No training flags were added; focus here is inference.
- Training throughput snapshot (d32, B200, bf16 autocast, seq=512, 2 steps per run): flash SDP on, torch.compile off, scaling batch size pushes throughput; fp32 logits slow the fastest configs. Measured tok/s:
  - b1 flash + fp32 logits: ~2.0K tok/s; flash + bf16 logits: ~6.6K tok/s.
  - b2 flash + fp32 logits: ~8.3K tok/s; flash + bf16 logits: ~9.6K tok/s.
  - b4 flash + fp32 logits: ~14.5K tok/s; flash + bf16 logits: ~18.3K tok/s.
  - b8 flash + fp32 logits: ~22.2K tok/s; flash + bf16 logits: ~30.9K tok/s.
  - b12 flash + bf16 logits: ~18.2K tok/s (throughput flattened vs b8); b16 flash + bf16 logits: ~31.6K tok/s. Above b8, gains come from larger batch; flash still enabled.
  - Longer seq (1024) with flash + bf16 logits: b1 ~4.0K tok/s, b2 ~15.7K tok/s, b4 ~24.6K tok/s; no OOM at b4. Grad accumulation can be used to simulate larger batches if memory caps out.
  - Heavier configs (flash + bf16 logits): b32 x 512 ~22.3K tok/s; b8 x 1536 ~32.2K tok/s. Past b8, returns flatten; pushing seq length can help if memory allows.
  - torch.compile (as of this nightly) hurt training throughput badly here; keep it off unless profiled otherwise. Increase batch size (with grad accumulation if needed) and use flash SDP to maximize training tok/s; disabling fp32 logits boosts raw throughput further at the cost of logits precision.
- Training toggles micro-bench (random data, d32 checkpoint, B200, bf16 autocast, bs=1, seq=256, 3 timed steps):
  - Flash off / fp32 logits off: ~822 tok/s.
  - Flash off / fp32 logits on: ~937 tok/s.
  - Flash on / fp32 logits off: ~972 tok/s.
  - Flash on / fp32 logits on: ~889 tok/s.
  - torch.compile + flash on / fp32 logits off: ~437 tok/s (timed phase only; ~43s one-time compile overhead made it a net loss). On this nightly build, compile regressed performance—keep it off for training.
