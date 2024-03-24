import pytest
import torch
import os
import triton

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from typing import Callable, Dict, Tuple, Union
from transformer_engine.pytorch.attention import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
)

from test_fused_rope import apply_rotary_pos_emb, _overlapping_grad, _non_overlapping_grad, get_tol
from rope_triton import apply_rotary_pos_emb_triton

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq_length", [1024, 2048, 4096])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("rotary_percent", [0.5, 1.0])
@pytest.mark.parametrize("margin", [0])
@pytest.mark.parametrize("transpose", [None])
@pytest.mark.parametrize("tensor_format", ["sbhd"])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
def test_fused_rope_ours(
    dtype: torch.dtype,
    seq_length: int,
    hidden_size: int,
    rotary_percent: float,
    margin: int,
    transpose: Union[Tuple, None],
    tensor_format: str,
    loss_func: Callable,
) -> None:
    
    device = torch.device("cuda:0")
    batch_size, head_num = 2, 64
    t = torch.rand(
        (seq_length - margin, batch_size, head_num, hidden_size),
        dtype=dtype,
        device=device,
    ) 
    t.requires_grad = True

    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent) # emb를 return.. inference 시 fwd는 아님 이후엔 계속 emb 이용 (여기서는)
    emb = rotary_pos_emb(seq_length) 

    # base unfused output
    output_unfused = apply_rotary_pos_emb(
        t, emb, tensor_format=tensor_format, fused=False
    )
    loss_unfused = loss_func(output_unfused)
    loss_unfused.backward()
    grad_unfused = t.grad.detach().clone()
    t.grad = None   
    
    # base fused output (CUDA)
    output_fused = apply_rotary_pos_emb(
        t,
        emb,
        tensor_format=tensor_format,
        fused=True,
    )
    loss_fused = loss_func(output_fused)
    loss_fused.backward() # 이후 grad is not None
    grad_fused = t.grad.detach().clone()
    t.grad = None
    
    # our output (triton)
    output_ours = apply_rotary_pos_emb_triton(
        t,
        emb,
        tensor_format=tensor_format,
        fused=True,
    )
    loss_ours = loss_func(output_ours)
    loss_ours.backward()
    grad_ours = t.grad.detach().clone()
    t.grad = None
        
    # test for fused AND unfused
    torch.testing.assert_close(output_fused, output_unfused, **get_tol(dtype))
    torch.testing.assert_close(grad_fused, grad_unfused, **get_tol(dtype))

    # test for fused AND ours
    torch.testing.assert_close(output_fused, output_ours, **get_tol(dtype))
    torch.testing.assert_close(grad_fused, grad_ours, **get_tol(dtype))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[1024, 2048, 4096, 8192],  # Different possible values for `x_name`. seq_len
        x_log=False,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'cuda'],  # Possible values for `line_arg`.
        line_names=['Triton', 'CUDA'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='rope-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark_seqlen(size, provider):
    quantiles = [0.5, 0.2, 0.8]
    seq_length = 4096
    hidden_size = 256
    rotary_percent = 1.0
    dtype = torch.float16
    tensor_format = "sbhd"
    device = torch.device("cuda:0")
    t = torch.rand((4096, 2, 64, 256), dtype=dtype, device=device)
    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent)
    emb = rotary_pos_emb(seq_length) 
    
    if provider == 'cuda':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: apply_rotary_pos_emb(t, emb, tensor_format=tensor_format, fused=True), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: apply_rotary_pos_emb_triton(t, emb, tensor_format=tensor_format, fused=True), quantiles=quantiles)
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__=="__main__":
    torch.manual_seed(0)
    benchmark_seqlen.run(print_data=True, show_plots=False)
    
