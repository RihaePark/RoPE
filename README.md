# Rotary Position Embedding

## Dependencies
- cuda 11.8
- cudnn 8.9.7

## Initial directory tree
```
.
├── env.yaml
├── README.md
├── rope_triton.py
└── test_ours.py
```

## Installation
```bash
conda env create -n rope -f env.yaml
conda activate rope
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
export NVTE_FRAMEWORK=pytorch
pip install .

mv ../rope_triton.py tests/pytorch/
mv ../test_ours.py tests/pytorch/
cd tests/pytorch/
```

## Test
```
cd tests/pytorch/
pytest test_ours.py # Functionality Check 
python test_ours.py # Performance Measure
```

## Code 설명 및 분석 결과

### file 및 function 설명

- (test_ours.py) `test_fused_rope_ours`:
    - `apply_rotary_pos_emb`를 통해 torch(fused=False), CUDA(fused=True)에 대한 각각의 RoPE output인 output_unfused, output_fused를 구한다.
    - 구현한 `apply_rotary_pos_emb_triton`를 통해 triton으로 구현된 RoPE output인 output_ours를 구한다.
    - `test_fused_rope_ours` 함수 호출 시 `torch.testing.assert_close`를 통해 각각의 output과 grad 값이 같은 것을 확인하고 종료한다.

- (test_ours.py) `benchmark_seqlen`:
    - `pytest test_ours.py` 실행 &rarr; 여러 dtype, seq_length, hidden_size, rotary_percent, loss_func에 대한 72개의 configuration에서 구현한 `apply_rotary_pos_emb_triton`의 funtionality를 체크한다.

- (rope_triton.py) `test_fused_rope_ours`:
    - `class FusedRoPETriton(torch.autograd.Function)`
        - TransformerEngine/transformer_engine/pytorch/attention.py의 `class FusedRoPEFunc`를 참고하여 forward, backward를 구성했다.
        - `apply_rotary_pos_emb_triton`에서 `FusedRoPETriton.apply`를 실행한다.
        - forward 시, `.forward` &rarr; `rope_wrapper_forward` &rarr; `rope_kernel_forward`와 같은 순으로 함수가 실행되어 triton kernel인 `rope_kernel_forward`에서 실질적으로 rope 연산을 한다.
        - backward 시, `.backward` &rarr; `rope_wrapper_backward` &rarr; `rope_kernel_backward`와 같은 순으로 함수가 실행되어 triton kernel인 `rope_kernel_forward`에서 실질적으로 backward 연산을 한다.

    - `rope_kernel_forward`: 
        ```
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
            pid = tl.program_id(axis=0)     # t_ptr, output_ptr의 idx에 사용됨
            s_id = pid // (b*h)             # seq_len에 대한 idx, freqs_ptr의 idx에 사용됨
            seq_start = pid * d
            
            d_offsets = seq_start + tl.arange(0, f_dim//2)
            d_mask = d_offsets < n_elements

            f_offsets = s_id * f_dim + tl.arange(0, f_dim//2)   # f_dim의 절반만 load
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
        ```
        - seq_len * batch * head_num 개의 위 kernel이 실행되며, 각 kernel에서는 hidden_dim 크기의 1-d tensor를 담당한다.
        - theta들의 모음인 freqs에는 같은 theta들이 두번씩 들어가 있는 점을 감안하여, freqs의 앞쪽 절반만 load하여 cos, sin을 구하고 이를 재사용 했다. (e.g., [s=4096, 1, 1, d=256] 크기의 freqs는 [s=4096, 1, 1, d=128] 크기의 동일한 두 텐서가 concat 되어있는 형태. 이는 hidden_dim의 앞쪽 절반과 뒷쪽 절반이 각각 pair를 이뤄 같은 theta에 대한 rope가 적용되기 때문이다.)
        - 실제 연산은 각 pair마다 한 theta에 대한 rotary 연산(R) 해주는, output = R*t 를 진행한다.
        - rotary_percent < hidden_dim일 경우, `rope_wrapper_forward`에서 rope가 적용되지 않고 pass되는 부분을 처리한다.
    - `rope_kernel_backward`: 
        ```
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
        ```
        - `rope_kernel_forward`와 기본적인 구조는 같으며, grad_out을 읽어와서 grad_input = R.T * grad_output 연산을 해준다.
        -  R.T이기 때문에 sin의 부호만 바뀌었다.

### 1. Functionality Check
- result of running `pytest test_ours.py`:
```
=================================================== test session starts ====================================================
platform linux -- Python 3.10.13, pytest-8.1.1, pluggy-1.4.0
rootdir: /home/arcuser/RoPE/TransformerEngine
collected 72 items                                                                                                         

test_ours.py ........................................................................                                [100%]

===================================================== warnings summary =====================================================
../../../../anaconda3/envs/trtr/lib/python3.10/site-packages/transformer_engine/pytorch/attention.py:16
  /home/arcuser/anaconda3/envs/trtr/lib/python3.10/site-packages/transformer_engine/pytorch/attention.py:16: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
    from pkg_resources import packaging

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================================== 72 passed, 1 warning in 4.11s ===============================================
```

### 2. Performance Measure
- seq_len = {1024, 2048, 4096, 8192}에 대해서 실험했다. 
- seq_len에 따라 linear하게 소요 시간이 증가하는 것을 볼 수 있다. 이는 같은 triton kernel을 seq_len * batch_size * head_num 개 실행되기 때문이다.
- result of running `python test_ours.py`:

rope-performance:
| | seq_len|    Triton|      CUDA|
|---|---|---|---|
|0|  1024.0|  0.016140|  0.010870|
|1|  2048.0|  0.032832|  0.021759|
|2|  4096.0|  0.065406|  0.043478|
|3|  8192.0|  0.130612|  0.087035|