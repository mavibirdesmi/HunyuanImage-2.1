import torch
from einops import rearrange

use_flash_attn_v3 = False
try:
    from flash_attn_interface import flash_attn_varlen_func, _flash_attn_forward

    def flash_attn_varlen_qkvpacked_func_v3(
        qkv,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        qv=None,
        q_descale=None, k_descale=None, v_descale=None,
        window_size=(-1, -1),
        num_splits=1,
        pack_gqa=None,
        attention_chunk=0,
        softcap=0.0,
        deterministic=False,
        num_heads_q=None,
        sm_margin=0,
        return_softmax=False,
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        if qkv.dim() == 5:
            assert qkv.shape[-3] == 3
            q, k, v = qkv.unbind(dim=-3)
        else:
            assert qkv.dim() == 4
            assert num_heads_q is not None
            num_heads_k = (qkv.shape[2] - num_heads_q) // 2
            assert num_heads_k * 2 + num_heads_q == qkv.shape[2]
            q, k, v = qkv.split([num_heads_q, num_heads_k, num_heads_k], dim=-2)

        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None, None,  # k_new, v_new
            qv,  # qv
            None,  # out
            cu_seqlens_q,
            cu_seqlens_k,
            None,   # cu_seqlens_k_new
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            None, None, None,   # page_table, kv_batch_idx, leftpad_k,
            None, None, None,  # rotary_cos/sin, seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
        )
        return (out, softmax_lse) if return_softmax else out

    print("Using FlashAttention v3.")
    use_flash_attn_v3 = True
except ImportError:
    print("FlashAttention v3 not found, falling back to v2.")
    from flash_attn import flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func

from flash_attn.bert_padding import pad_input, unpad_input


def get_cu_seqlens(text_mask: torch.Tensor, img_len: int):
    """
    Compute cumulative sequence lengths (cu_seqlens) for FlashAttention.

    Args:
        text_mask (torch.Tensor): Boolean mask of shape (batch_size, text_seq_len).
        img_len (int): Length of image sequence.

    Returns:
        cu_seqlens (torch.Tensor): 1D tensor of cumulative sequence lengths for each segment.
        max_len (int): Maximum sequence length (text + image).
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device=text_mask.device)
    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens, max_len


def flash_attn_v3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_s: int,
    causal: bool = False,
    deterministic: bool = False,
):
    """
    FlashAttention v3 wrapper.

    Args:
        q, k, v (torch.Tensor): Query, key, value tensors of shape (batch, seq, nheads, head_dim).
        cu_seqlens (torch.Tensor): Cumulative sequence lengths.
        max_s (int): Maximum sequence length.
        causal (bool): Whether to apply causal masking.
        deterministic (bool): Deterministic computation.

    Returns:
        torch.Tensor: Output tensor of shape (batch, seq, nheads, head_dim).
    """
    batch_size, seqlen = q.shape[:2]
    q = q.reshape(-1, *q.shape[2:])
    k = k.reshape(-1, *k.shape[2:])
    v = v.reshape(-1, *v.shape[2:])
    output = flash_attn_varlen_func(
        q, k, v, cu_seqlens, cu_seqlens, max_s, max_s, causal=causal, deterministic=deterministic
    )
    output = output.view(batch_size, seqlen, *output.shape[-2:])
    return output


def flash_attn_no_pad(
    qkv: torch.Tensor,
    key_padding_mask: torch.Tensor,
    causal: bool = False,
    dropout_p: float = 0.0,
    softmax_scale=None,
    deterministic: bool = False,
):
    """
    FlashAttention for packed QKV input without padding.

    Args:
        qkv (torch.Tensor): Input tensor of shape (batch, seq, 3, nheads, head_dim).
        key_padding_mask (torch.Tensor): Boolean mask of shape (batch, seq).
        causal (bool): Whether to apply causal masking.
        dropout_p (float): Dropout probability.
        softmax_scale (float, optional): Softmax scaling factor.
        deterministic (bool): Deterministic computation.

    Returns:
        torch.Tensor: Output tensor of shape (batch, seq, nheads, head_dim).
    """
    batch_size, seqlen, _, nheads, head_dim = qkv.shape
    x = rearrange(qkv, "b s three h d -> b s (three h d)")

    # Unpad input for FlashAttention, drop `used_seqlens_in_batch` for version compatibility
    x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)[:4]
    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)

    if use_flash_attn_v3:
        output_unpad = flash_attn_varlen_qkvpacked_func_v3(
            qkv=x_unpad,
            cu_seqlens_k=cu_seqlens,
            cu_seqlens_q=cu_seqlens,
            seqused_q=seqlen,
            seqused_k=seqlen,
            max_seqlen_q=max_s,
            max_seqlen_k=max_s,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )
    else:
        output_unpad = flash_attn_varlen_qkvpacked_func(
            x_unpad,
            cu_seqlens,
            max_s,
            dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )

    if isinstance(output_unpad, tuple):
        output_unpad = output_unpad[0]

    # Pad output back to original shape
    output = pad_input(
        rearrange(output_unpad, "nnz h d -> nnz (h d)"),
        indices,
        batch_size,
        seqlen,
    )
    output = rearrange(output, "b s (h d) -> b s h d", h=nheads)
    return output
