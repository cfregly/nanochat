import torch
import torch.nn.functional as F


def _flash3_clustered(q, k, v, causal: bool, num_sm_clusters: int | None):
    # q,k,v: (B, H, T, D)
    B, Hq, Tq, D = q.shape
    _, Hk, Tk, _ = k.shape
    q_flat = q.transpose(1, 2).reshape(B * Tq, Hq, D)
    k_flat = k.transpose(1, 2).reshape(B * Tk, Hk, D)
    v_flat = v.transpose(1, 2).reshape(B * Tk, Hk, D)
    cu_q = torch.arange(0, (B + 1) * Tq, step=Tq, device=q.device, dtype=torch.int32)
    cu_k = torch.arange(0, (B + 1) * Tk, step=Tk, device=q.device, dtype=torch.int32)

    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func  # type: ignore
        import inspect

        kwargs = dict(
            dropout_p=0.0,
            causal=causal,
        )
        if num_sm_clusters is not None and "num_sm_clusters" in inspect.signature(flash_attn_varlen_func).parameters:
            kwargs["num_sm_clusters"] = num_sm_clusters
        out = flash_attn_varlen_func(  # type: ignore[misc]
            q_flat,
            k_flat,
            v_flat,
            cu_q,
            cu_k,
            Tq,
            Tk,
            **kwargs,
        )
        return out.view(B, Tq, Hq, D).transpose(1, 2).contiguous()
    except Exception:
        return None


def clustered_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None,
    causal: bool,
    num_sm_clusters: int | None = None,
    enable_gqa: bool = False,
):
    """
    Clustered attention entry point.
    - Tries FlashAttention-3 varlen with num_sm_clusters if available.
    - Falls back to SDPA.
    """
    # Masks are not supported in FA3 varlen path; fall back to SDPA when provided.
    use_mask = attn_mask is not None
    if enable_gqa and q.size(1) != k.size(1):
        repeat_k = q.size(1) // k.size(1)
        k = k.repeat_interleave(repeat_k, dim=1)
        v = v.repeat_interleave(repeat_k, dim=1)
    if (
        not use_mask
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
    ):
        fa3_out = _flash3_clustered(q, k, v, causal=causal, num_sm_clusters=num_sm_clusters)
        if fa3_out is not None:
            return fa3_out

    # SDPA fallback
    # attn_mask semantics: True=keep, False=mask
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True):
        if use_mask:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
