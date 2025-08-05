# TopK

The topK problem is how to select the top k elements from a given array. It's a common operation in LLM, recommender systems, and other applications. This project implements the topK operation using triton and cuda.

The classical topK operation has three methods:
1. Scan the array k times, and each time find the maximum element and remove it.
2. Sort the array and select the first k elements.
3. Find the k-th largest element and partition the array around it.

The time complexity of the first method is O(n*k), the second method is O(n*log(n)). The third method usally uses the radix-select algorithm, which has an advantage when k and n are large.


# radix-select

The radix-select algorithm is a linear time selection algorithm that can find the k-th largest element in an array. It works by partitioning the array into buckets based on the bits of the elements, and then recursively selecting the k-th largest element from the appropriate bucket.

Let's take an example of an unsigned integer array to illustrate the radix-select algorithm:



## radix-select For Float


The core idea of float radix-selection is to map the float array to a monotonic ascending unsigned integer array, and then apply the radix-select algorithm on the mapped array. The mapping process for float32 is as follows:
1. Covert the float number $x$ to an unsigned integer $u$: `u = reinterpret_cast<uint32_t&>(x)`;
2. If $x$ is positive, we need to flip the sign bit: `u = u ^ 0x80000000`;
3. If $x$ is negative, we need to flip all bits: `u = u ^ 0xFFFFFFFF`.

Therefore, we can map the float array to an unsigned integer array using the following formula:
```cpp
u = reinterpret_cast<uint32_t&>(x);
u = u ^ ((u >> 31) ? 0xFFFFFFFF : 0x80000000);
```

What's more, there are some corner cases that need handling:
- NaN
- $\pm 0$
- Infinity

#### NaN

NaN is a special floating-point number that represents "not a number". It's used to indicate that the result of a floating-point operation is undefined. For example, 0.0 / 0.0 is NaN.

There are two strategies to handle NaN:
1. Filter out NaN values before the selection process.
2. Put NaN at the end of the array after the selection process.

Take float32 as an example, we can map NaN to the largest unsigned integer:
```cpp
uint32_t u = bit_cast<uint32_t>(x);
bool is_nan = ((u & 0x7F800000u) == 0x7F800000u) && (u & 0x007FFFFFu);
if (is_nan) return 0u; // If program requires k smallest elements, we can return UINT32_MAX to indicate NaN.
```

### $\pm 0$

- Make $\pm 0$ equal to 0, so we can use the same mapping as above
- Make $-0 < +0$t

### Infinity & Subnormal Numbers

- There is no need to handle infinity, as it will not affect the selection process.

## CUDA Kernel Design

We assume that the input array is a 2D tensor, where the first dimension is the batch size and the second dimension is the number of elements in each batch.

### v1 

- Grid parallelism
  - We partition the input array's batch dimension into blocks, and each block processes one batch.
- Block parallelism
  - Each thread within a block processes one element of the batch, allowing for efficient parallelism.

#### Radix-Select within a Block

radix_bits: 
radix_size: 

STEP 1:
- thread: Each thread processes count the number of elements in each local buckets
- warp:
  - Each warp will hold shared memory buckets
  - Each warp add each threads' buckets into shared memory buckets
  - Each warp add all the shared memory buckets into shared memory buckets
- theard: Each thread read the shared memory bucket and write the result to the local buckets

STEP 2:




# Reference

1. [TOPK算子CUDA实现代码分析](https://zhuanlan.zhihu.com/p/1924510640476762596)

