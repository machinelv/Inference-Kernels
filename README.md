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
    - SGEMM: [SGEMM](https://github.com/machinelv/Kernels-Optimization/tree/master/GEMM/gpu/CUDA)
    - Low Precision GEMM: [QGEMM](https://github.com/machinelv/QGEMM)
    - Mix Precision GEMM: [HIP Source FP8](https://github.com/machinelv/AMD-Infer-25/tree/master/fp8-mm)
- MLA
- MoE

The kernels will be realized by:
- C++ 17/20 standard
- triton
- pytorch


## Realized Kernels

- [x] GEMM CUDA BF16
- [ ] Mix Precision GEMM CUDA FP8
- [ ] TopK CUDA
- [x] Softmax Triton


## Results
- CUDA GEMM BF16 has achieved 97% performance of cublas