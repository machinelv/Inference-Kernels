# Kernel Optimization

This project aims to optimize LLM inference kernels in CPU and GPU.

The basic kernels include:

- Reduce CUDA(Need Moving)
- GEMM CPU:
    - SVE
    - AVX-512
    - SME
    - AMX
- GEMM GPU
    - SGEMM: Finished
    - HGEMM&FP8 GEMM: TODO
- MLA
- MoE

The kernels will be realized by:
- C++ 17 standard
- triton
- pytorch
