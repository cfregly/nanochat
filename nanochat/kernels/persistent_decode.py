import torch


class PersistentDecodeRunner:
    """
    CUDA Graph-based persistent decode runner with shape/cache-gen awareness.
    Captures a decode step (ids -> logits) and replays it for steady-state decode.
    """

    def __init__(self):
        self.graph = None
        self.static_ids = None
        self.static_attention = None
        self.static_token_mask = None
        self.output = None
        self.device = None
        self.cache_gen = None
        self.stream = None

    def reset(self):
        self.graph = None
        self.static_ids = None
        self.static_attention = None
        self.static_token_mask = None
        self.output = None
        self.device = None
        self.cache_gen = None
        self.stream = None

    def _shapes_match(self, ids, attention_mask, token_mask, kv_cache):
        if self.static_ids is None:
            return False
        if self.static_ids.shape != ids.shape:
            return False
        if self.device != ids.device:
            return False
        cache_gen = getattr(kv_cache, "cache_gen", None)
        if cache_gen != self.cache_gen:
            return False
        attn_present = attention_mask is not None
        token_present = token_mask is not None
        if attn_present != (self.static_attention is not None):
            return False
        if token_present != (self.static_token_mask is not None):
            return False
        return True

    def capture(self, model, ids, kv_cache, attention_mask=None, token_mask=None):
        if not torch.cuda.is_available() or not ids.is_cuda:
            raise RuntimeError("Persistent decode capture requires CUDA tensors")
        torch.cuda.synchronize(ids.device)
        self.graph = torch.cuda.CUDAGraph()
        self.stream = torch.cuda.Stream(device=ids.device)
        self.static_ids = ids.clone()
        self.static_attention = attention_mask.clone() if attention_mask is not None else None
        self.static_token_mask = token_mask.clone() if token_mask is not None else None
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.output = model.forward(
                self.static_ids,
                kv_cache=kv_cache,
                attention_mask=self.static_attention,
                token_mask=self.static_token_mask,
            )
        self.device = ids.device
        self.cache_gen = getattr(kv_cache, "cache_gen", None)

    def forward(self, model, ids, kv_cache, attention_mask=None, token_mask=None):
        if (
            self.graph is None
            or not self._shapes_match(ids, attention_mask, token_mask, kv_cache)
        ):
            self.capture(model, ids, kv_cache, attention_mask, token_mask)

        self.static_ids.copy_(ids)
        if self.static_attention is not None and attention_mask is not None:
            self.static_attention.copy_(attention_mask)
        if self.static_token_mask is not None and token_mask is not None:
            self.static_token_mask.copy_(token_mask)
        self.graph.replay()
        return self.output
