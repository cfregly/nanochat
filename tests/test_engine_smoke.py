import os

import torch

from nanochat.engine import Engine, KVCache
from nanochat.gpt import GPT, GPTConfig


class _StubTokenizer:
    def get_bos_token_id(self):
        return 0

    def encode_special(self, token):
        return 1


def _small_config(**overrides):
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=8,
        **overrides,
    )
    return cfg


def test_cuda_graphs_disabled_when_persistent_decode_enabled(monkeypatch):
    # Avoid torch.compile in the test environment
    monkeypatch.setenv("NANOCHAT_DISABLE_COMPILE", "1")
    config = _small_config(enable_persistent_decode=True, use_cuda_graphs=True)
    model = GPT(config)
    tokenizer = _StubTokenizer()

    engine = Engine(model, tokenizer, enable_batch_decode=False)

    assert engine.enable_persistent_decode is True
    # Graphs remain enabled but use the default stream (no dedicated persistent stream)
    assert engine.use_cuda_graphs is True
    assert engine._persistent_stream is None


def test_cuda_graphs_disabled_after_kv_growth(monkeypatch):
    monkeypatch.setenv("NANOCHAT_DISABLE_COMPILE", "1")
    config = _small_config(enable_persistent_decode=False, use_cuda_graphs=True)
    model = GPT(config)
    tokenizer = _StubTokenizer()
    engine = Engine(model, tokenizer, enable_batch_decode=False)

    # Patch out the heavy forward path
    engine._decode_forward_step = lambda *args, **kwargs: torch.zeros((1, 1, 1))

    kv_cache = KVCache(**engine._kv_cache_params(batch_size=1, seq_len=2))
    kv_cache.cache_gen += 1  # simulate reallocation

    ids = torch.tensor([[1]], dtype=torch.long)
    _ = engine._execute_decode(ids, kv_cache)

    # Graphs should recapture instead of being permanently disabled
    assert engine._decode_graph_disabled is False
