import torch
import torch.nn.functional as F
from einops import rearrange

use_flash_attn_v3 = False
try:
    from flash_attn_interface import flash_attn_varlen_func, _flash_attn_forward

    def flash_attn_varlen_qkvpacked_func_v3(
        qkv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        seqused_q = None,
        seqused_k = None,
        softmax_scale = None,
        causal = False,
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
        q, k, v = qkv[:, 0].detach(), qkv[:, 1].detach(), qkv[:, 2].detach()
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])

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
            None, None,  # rotary_cos/sin
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal=causal,
            window_size=window_size,
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

def unpad_input(hidden_states, attention_mask, unused_mask=None):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
        indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
        seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
    """
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices.
    return (
        rearrange(hidden_states, "b s ... -> (b s) ...")[indices],
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[1:]
    output = torch.zeros((batch * seqlen), *dim, device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


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
