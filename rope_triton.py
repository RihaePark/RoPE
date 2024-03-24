import torch
import triton
import triton.language as tl
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

## temp
import transformer_engine_extensions as tex


# 일단 seq_len dim으로 쪼갬
@triton.jit
def rope_kernel_forward(t_ptr, 
                freqs_ptr, 
                output_ptr, 
                seq_len,
                b,
                h,
                d,
                f_dim: tl.constexpr,
                n_elements,
                ):
    pid = tl.program_id(axis=0)
    s_id = pid // (b*h)
    seq_start = pid * d
    
    d_offsets = seq_start + tl.arange(0, f_dim//2)
    d_mask = d_offsets < n_elements

    f_offsets = s_id * f_dim + tl.arange(0, f_dim//2)
    f_mask = f_offsets < f_dim * seq_len
    thetas = tl.load(freqs_ptr + f_offsets, mask=f_mask)
    cos = tl.cos(thetas)
    sin = tl.sin(thetas)
    
    d0 = tl.load(t_ptr + d_offsets, mask = d_mask)
    d1 = tl.load(t_ptr + f_dim//2 + d_offsets, mask = d_mask)
    output0 = d0 * cos - d1 * sin
    output1 = d1 * cos + d0 * sin
    tl.store(output_ptr + d_offsets, output0, mask = d_mask)
    tl.store(output_ptr + f_dim//2 + d_offsets, output1, mask = d_mask)

def rope_wrapper_forward(t: torch.Tensor, freqs: torch.Tensor):
    """
    Args:
        t (torch.Tensor): torch.Size([4096, 2, 64, 256])
        freqs (torch.Tensor): torch.Size([4096, 1, 1, 256])

    Returns:
        output (torch.Tensor): torch.Size([4096, 2, 64, 256])
    """
    output = torch.empty_like(t)
    assert t.is_cuda and freqs.is_cuda and output.is_cuda
        
    seq_len = output.shape[0]
    b = output.shape[1]
    h = output.shape[2]
    d = output.shape[-1]
    n_rows = seq_len * b * h
    
    f_dim =  freqs.shape[-1]
    output[...,f_dim:] = t[...,f_dim:]
    
    grid = lambda meta: (n_rows,)
    rope_kernel_forward[grid](t, freqs, output, seq_len, b, h, d, f_dim, n_rows*d)
    return output

@triton.jit
def rope_kernel_backward(grad_out_ptr, 
                freqs_ptr, 
                grad_input_ptr, 
                seq_len,
                b,
                h,
                d,
                f_dim: tl.constexpr,
                n_elements,
                ):
    pid = tl.program_id(axis=0)
    s_id = pid // (b*h)
    seq_start = pid * d
    
    d_offsets = seq_start + tl.arange(0, f_dim//2)
    d_mask = d_offsets < n_elements

    f_offsets = s_id * f_dim + tl.arange(0, f_dim//2)
    f_mask = f_offsets < f_dim * seq_len
    thetas = tl.load(freqs_ptr + f_offsets, mask=f_mask)
    cos = tl.cos(thetas)
    sin = tl.sin(thetas)
    
    d0 = tl.load(grad_out_ptr + d_offsets, mask = d_mask)
    d1 = tl.load(grad_out_ptr + f_dim//2 + d_offsets, mask = d_mask)
    output0 = d0 * cos + d1 * sin
    output1 = d1 * cos - d0 * sin
    tl.store(grad_input_ptr + d_offsets, output0, mask = d_mask)
    tl.store(grad_input_ptr + f_dim//2 + d_offsets, output1, mask = d_mask)


def rope_wrapper_backward(grad_output: torch.Tensor, freqs: torch.Tensor):
    """
    Args:
        grad_output (torch.Tensor): torch.Size([4096, 2, 64, 256])
        freqs (torch.Tensor): torch.Size([4096, 1, 1, 256])

    Returns:
        grad_input (torch.Tensor): torch.Size([4096, 2, 64, 256])
    """
    grad_input = torch.empty_like(grad_output)
    assert grad_output.is_cuda and freqs.is_cuda and grad_input.is_cuda

    seq_len = grad_input.shape[0] 
    b = grad_input.shape[1]
    h = grad_input.shape[2]
    d = grad_input.shape[-1]
    n_rows = seq_len * b * h
    f_dim =  freqs.shape[-1]
    
    grid = lambda meta: (n_rows,)
    grad_input[...,f_dim:] = grad_output[...,f_dim:]
    grad_output = grad_output.contiguous()
    rope_kernel_backward[grid](grad_output, freqs, grad_input, seq_len, b, h, d, f_dim, n_rows*d)
    return grad_input


class FusedRoPETriton(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor, 
        tensor_format: str = "sbhd",
        cu_seqlens: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        
        if tensor_format == "sbhd":
            output = rope_wrapper_forward(t, freqs)
        elif tensor_format == "bshd":
            output = rope_wrapper_forward(
                t.transpose(0, 1), freqs).transpose(0, 1)
        # elif tensor_format == "thd":
        #     output = tex.fused_rope_thd_forward(t, cu_seqlens, freqs)
        else:
            raise ValueError(f"Unsupported tensor_format: {tensor_format}.")
        
        ctx.save_for_backward(freqs, cu_seqlens)
        ctx.tensor_format = tensor_format
        
        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        freqs, cu_seqlens = ctx.saved_tensors
        
        if ctx.tensor_format == "sbhd":
            grad_input = rope_wrapper_backward(grad_output, freqs) # ours
        elif ctx.tensor_format == "bshd":
            grad_input = rope_wrapper_backward(grad_output.transpose(0, 1), freqs).transpose(0, 1)
        # elif ctx.tensor_format == "thd":
        #     grad_input = rope_wrapper_backward(grad_output, cu_seqlens, freqs)
        else:
            raise ValueError(f"Unsupported tensor_format: {ctx.tensor_format}.")
        
        return grad_input, None, None, None, None
    

def apply_rotary_pos_emb_triton(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    fused: bool = False,
    cu_seqlens: Union[torch.Tensor, None] = None,
) -> torch.Tensor:
    """
    Apply rotary positional embedding tensor to the input tensor.

    Parameters
    ----------
    t: torch.Tensor
        Input tensor of shape `[s, b, h, d]`, `[s, b, h, d]` or `[t, h, d]`, on which
        rotary positional embedding will be applied.
    freqs: torch.Tensor
        Rotary positional embedding tensor of shape `[s2, 1, 1, d2]` and dtype 'float',
        with `s2 >= s` and `d2 <= d`.
    fused: bool, default = False
        Whether to use a fused applying RoPE implementation.
    tensor_format: {'sbhd', 'bshd', 'thd'}, default = 'sbhd'
        is `bshd` if `t` is of shape `[bs, seq, ...]`, or `sbhd` if `t` is
        of shape `[seq, bs, ...]`. 'thd' is only supported when `fused` is True.
    cu_seqlens: torch.Tensor, default = None.
        Cumulative sum of sequence lengths in a batch for `t`, with shape [b + 1] and
        dtype torch.int32. Only valid when `tensor_format` is 'thd'.
    """
    if fused:
        assert (
            tensor_format != "thd" or cu_seqlens is not None
        ), "cu_seqlens must not be None when tensor_format is 'thd'."
        
        return FusedRoPETriton.apply(t, freqs, tensor_format, cu_seqlens)

    assert tensor_format in ("sbhd", "bshd"), (
        "Only formats `sbhd` or `bshd` are supported for input tensor `t` "
        f"when fused is False, got {tensor_format}."
    )

    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[1] if tensor_format == "bshd" else t.shape[0]

    # Only apply the rotary embeddings up to the sequence length of the running
    # input.
    assert cur_seq_len <= max_seq_len, (
        f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"
    )
    freqs = freqs[:cur_seq_len]
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)  # [seq, 1, 1, dim] -> [1, seq, 1, dim]
    # cos/sin first then dtype conversion for better precision

    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * cos_) + (_rotate_half(t) * sin_)
    return torch.cat((t, t_pass), dim=-1)

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x = x.view(x.shape[:-1] + torch.Size((2, x.shape[-1] // 2)))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)