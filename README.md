# Kernel Optimization

This project aims to optimize LLM inference kernels in GPU.

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
- C++ 17/20 standard
- triton
- pytorch


## Realized Kernels

- [x] GEMM CUDA FP16
- [ ] Mix Precision GEMM CUDA FP8
- [ ] TopK CUDA
- [ ] Softmax Triton