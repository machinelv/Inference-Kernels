from torch.utils.cpp_extension import load_inline
import torch
from torch import Tensor, _int, _bool, SymInt
from typing import Union, Optional
CPP_WRAPPER = """
void topk(torch::Tensor input, int k, int dim, bool largest, bool sorted, torch::Tensor out);
"""

CUDA_SRC = """
"""


import os
os.environ["CXX"] = "clang++"

module = load_inline(
    name='topk',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=['topk'],
    verbose=True,
    extra_cflags=["--offload-arch=sm80", "-std=c++20", "-O3"],
    extra_cuda_cflags=["--offload-arch=sm80", "-std=c++20", "-v", "-O3"],
)


def topk(input: Tensor, k: Union[_int, SymInt], dim: _int = -1, largest: _bool = True, sorted: _bool = True):
    out = torch.empty((input.shape[0], k), dtype=input.dtype, device=input.device)
    module.topk(input, k, dim, largest, sorted, out)
    return out