import triton
import triton.language as tl

@triton.jit
def topk_kernel(input, k, output, dim, largest, sorted, out):
    '''
    Implement the top-k kernel using Triton
    input: 2D tensor of shape (N, M)
    k: number of top elements to select
    output: 2D tensor of shape (N, k) or (M, k)
    dim: dimension along which to select top-k
    largest: whether to select the largest or smallest elements
    sorted: whether to return the elements in sorted order
    block partition: each block processes one row/column
    grid: (N,) if dim=0 else (M,)
    '''
    block_id = tl.program_id(0)
    


    
    pass

def topk(input, k, dim=None, largest=True, sorted=True):
    if dim is None:
        dim = -1
    out = triton.empty((input.shape[0], k), dtype=input.dtype)
    topk_kernel[(input.shape[0],)](input, k, dim, largest, sorted, out)
    return out