#include <iostream>
#include <cstdlib>

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define LDMATRIX_BF16_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))

#define LDMATRIX_BF16_TRANSPOSE_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))

#define WARP_SIZE 32
#define VECTYPE float4
#define PADDING 0

template<size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t WARP_TILE_SIZE_M, size_t WARP_TILE_SIZE_N, size_t MAT_TILE_SIZE_M, size_t MAT_TILE_SIZE_N, size_t THREAD_NUM>
__global__ void matrix_transpose_kernel_v1(const __nv_bfloat16* idata, __nv_bfloat16* odata, int M, int N) {
    // Get index
    size_t block_id_m = blockIdx.y;
    size_t block_id_n = blockIdx.x;
    size_t thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    size_t warp_id = thread_id / WARP_SIZE;
    size_t lane_id = thread_id % WARP_SIZE;

    size_t warp_id_m = warp_id / (BLOCK_TILE_SIZE_M / WARP_TILE_SIZE_M);
    size_t warp_id_n = warp_id % (BLOCK_TILE_SIZE_N / WARP_TILE_SIZE_N);

    // load matrix from global memory to shared memory using vector fetching
    __shared__ __nv_bfloat16 block_tile[BLOCK_TILE_SIZE_M][BLOCK_TILE_SIZE_N + PADDING];

    static_assert(sizeof(VECTYPE) % sizeof(__nv_bfloat16) == 0, "VECTYPE must be a multiple of T size");
    constexpr size_t NUM_VECTOR_UNITS{sizeof(VECTYPE) / sizeof(__nv_bfloat16)};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_N{BLOCK_TILE_SIZE_N / NUM_VECTOR_UNITS};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_M{BLOCK_TILE_SIZE_M / NUM_VECTOR_UNITS};

    static_assert(BLOCK_TILE_SIZE_M % NUM_VECTOR_UNITS == 0);
    static_assert(BLOCK_TILE_SIZE_N % NUM_VECTOR_UNITS == 0);

    union VectorAccess {
        VECTYPE vec;
        __nv_bfloat16 elements[NUM_VECTOR_UNITS];
    };

    size_t block_tile_start_m = block_id_m * BLOCK_TILE_SIZE_M;
    size_t block_tile_start_n = block_id_n * BLOCK_TILE_SIZE_N;

    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_M * VECTORIZED_BLOCK_TILE_SIZE_N + THREAD_NUM - 1) / THREAD_NUM; load_idx ++) {
        size_t block_tile_M_id{(thread_id + load_idx * THREAD_NUM) / VECTORIZED_BLOCK_TILE_SIZE_N};
        size_t block_tile_N_id{(thread_id + load_idx * THREAD_NUM) % VECTORIZED_BLOCK_TILE_SIZE_N * NUM_VECTOR_UNITS};

        size_t M_id{block_tile_M_id + block_tile_start_m};
        size_t N_id{block_tile_N_id + block_tile_start_n};

        VectorAccess row_vector_vals;

        row_vector_vals.vec = *reinterpret_cast<VECTYPE const*>(&idata[M_id * N + N_id]);
        if (block_tile_N_id < BLOCK_TILE_SIZE_N && block_tile_M_id < BLOCK_TILE_SIZE_M) {
            *reinterpret_cast<VECTYPE*>(&block_tile[block_tile_M_id][block_tile_N_id]) = row_vector_vals.vec;
        }
    }
    __syncthreads();
    // use movmatrix instruction to transpose the matrix per warp
    constexpr size_t MAT_TILE_PER_WARP_M = WARP_TILE_SIZE_M / MAT_TILE_SIZE_M;
    constexpr size_t MAT_TILE_PER_WARP_N = WARP_TILE_SIZE_N / MAT_TILE_SIZE_N;

    uint32_t warp_tile[MAT_TILE_PER_WARP_N][MAT_TILE_PER_WARP_M][4];

    size_t smem_n_offset = lane_id / 16 * 8;
    size_t smem_m_offset = lane_id % 16;

    const size_t warp_tile_start_m = warp_id_m * WARP_TILE_SIZE_M;
    const size_t warp_tile_start_n = warp_id_n * WARP_TILE_SIZE_N;

    #pragma unroll(MAT_TILE_PER_WARP_M)
    for (size_t warp_tile_m{0}; warp_tile_m < MAT_TILE_PER_WARP_M; ++warp_tile_m) {
        #pragma unroll(MAT_TILE_PER_WARP_N)
        for (size_t warp_tile_n{0}; warp_tile_n < MAT_TILE_PER_WARP_N; ++warp_tile_n) {
            size_t src_m = warp_tile_start_m + warp_tile_m * MAT_TILE_SIZE_M;
            size_t src_n = warp_tile_start_n + warp_tile_n * MAT_TILE_SIZE_N;
            uint32_t smem_addr = __cvta_generic_to_shared(&block_tile[src_m + smem_m_offset][src_n + smem_n_offset]);
            LDMATRIX_BF16_TRANSPOSE_X4(warp_tile[warp_tile_n][warp_tile_m][0], warp_tile[warp_tile_n][warp_tile_m][1], warp_tile[warp_tile_n][warp_tile_m][2], warp_tile[warp_tile_n][warp_tile_m][3], smem_addr);
        }
    }

    smem_m_offset = lane_id % 4 * 2;
    smem_n_offset = lane_id / 4;
    // store the transposed matrix from register to shared memory
    #pragma unroll(MAT_TILE_PER_WARP_N)
        for (size_t warp_tile_n{0}; warp_tile_n < MAT_TILE_PER_WARP_N; ++warp_tile_n) {
        #pragma unroll(MAT_TILE_PER_WARP_M)
        for (size_t warp_tile_m{0}; warp_tile_m < MAT_TILE_PER_WARP_M; ++warp_tile_m) {
            size_t src_m = warp_tile_start_m + warp_tile_m * MAT_TILE_SIZE_M;
            size_t src_n = warp_tile_start_n + warp_tile_n * MAT_TILE_SIZE_N;
#if CUDA_SM >= 90
            uint32_t smem_addr = __cvta_generic_to_shared(&block_tile[src_n + smem_n_offset][src_m + smem_m_offset]);
            STMATRIX_BF16_X4(smem_addr, warp_tile[warp_tile_n][warp_tile_m][0], warp_tile[warp_tile_n][warp_tile_m][1], warp_tile[warp_tile_n][warp_tile_m][2], warp_tile[warp_tile_n][warp_tile_m][3]);
#else
            // not support  
            (reinterpret_cast<uint32_t*>(&block_tile[src_n + smem_n_offset][src_m + smem_m_offset]))[0] = warp_tile[warp_tile_n][warp_tile_m][0];
            (reinterpret_cast<uint32_t*>(&block_tile[src_n + smem_n_offset][src_m + smem_m_offset + 8]))[0] = warp_tile[warp_tile_n][warp_tile_m][1];
            (reinterpret_cast<uint32_t*>(&block_tile[src_n + smem_n_offset + 8][src_m + smem_m_offset]))[0] = warp_tile[warp_tile_n][warp_tile_m][2];
            (reinterpret_cast<uint32_t*>(&block_tile[src_n + smem_n_offset + 8][src_m + smem_m_offset + 8]))[0] = warp_tile[warp_tile_n][warp_tile_m][3];            
#endif
        }
    }
    __syncthreads();

    // store the transposed matrix from shared memory to global memory
    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_N * VECTORIZED_BLOCK_TILE_SIZE_M + THREAD_NUM - 1) / THREAD_NUM; load_idx ++) {
        size_t block_tile_M_id{(thread_id + load_idx * THREAD_NUM) % VECTORIZED_BLOCK_TILE_SIZE_M * NUM_VECTOR_UNITS};
        size_t block_tile_N_id{(thread_id + load_idx * THREAD_NUM) / VECTORIZED_BLOCK_TILE_SIZE_M};

        size_t M_id{block_tile_M_id + block_tile_start_m};
        size_t N_id{block_tile_N_id + block_tile_start_n};

        VectorAccess row_vector_vals;
        if (block_tile_N_id < BLOCK_TILE_SIZE_N && block_tile_M_id < BLOCK_TILE_SIZE_M) {
           *reinterpret_cast<VECTYPE*>(&odata[N_id * M + M_id]) = *reinterpret_cast<VECTYPE*>(&block_tile[block_tile_N_id][block_tile_M_id]);
        }
    }
}


template<size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t THREAD_NUM>
__global__ void matrix_transpose_kernel_v2(const __nv_bfloat16* idata, __nv_bfloat16* odata, int M, int N)
{
    // Get index
    size_t block_id_m = blockIdx.y;
    size_t block_id_n = blockIdx.x;
    size_t thread_id = threadIdx.x + threadIdx.y * blockDim.x;

    // load matrix from global memory to shared memory using vector fetching
    __shared__ __nv_bfloat16 block_tile[BLOCK_TILE_SIZE_M][BLOCK_TILE_SIZE_N];

    static_assert(sizeof(VECTYPE) % sizeof(__nv_bfloat16) == 0, "VECTYPE must be a multiple of T size");
    constexpr size_t NUM_VECTOR_UNITS{sizeof(VECTYPE) / sizeof(__nv_bfloat16)};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_N{BLOCK_TILE_SIZE_N / NUM_VECTOR_UNITS};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_M{BLOCK_TILE_SIZE_M / NUM_VECTOR_UNITS};

    static_assert(BLOCK_TILE_SIZE_M % NUM_VECTOR_UNITS == 0);
    static_assert(BLOCK_TILE_SIZE_N % NUM_VECTOR_UNITS == 0);

    union VectorAccess {
        VECTYPE vec;
        __nv_bfloat16 elements[NUM_VECTOR_UNITS];
    };

    size_t block_tile_start_m = block_id_m * BLOCK_TILE_SIZE_M;
    size_t block_tile_start_n = block_id_n * BLOCK_TILE_SIZE_N;

    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_M * VECTORIZED_BLOCK_TILE_SIZE_N + THREAD_NUM - 1) / THREAD_NUM; load_idx ++) {
        size_t block_tile_M_id{(thread_id + load_idx * THREAD_NUM) / VECTORIZED_BLOCK_TILE_SIZE_N};
        size_t block_tile_N_id{(thread_id + load_idx * THREAD_NUM) % VECTORIZED_BLOCK_TILE_SIZE_N * NUM_VECTOR_UNITS};

        size_t M_id{block_tile_M_id + block_tile_start_m};
        size_t N_id{block_tile_N_id + block_tile_start_n};

        VectorAccess row_vector_vals;

        row_vector_vals.vec = *reinterpret_cast<VECTYPE const*>(&idata[M_id * N + N_id]);
        *reinterpret_cast<VECTYPE*>(&block_tile[block_tile_M_id][block_tile_N_id]) = row_vector_vals.vec;
    }
    __syncthreads();

    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_M * VECTORIZED_BLOCK_TILE_SIZE_N + THREAD_NUM - 1) / THREAD_NUM; load_idx ++) {
        size_t block_tile_M_id{(thread_id + load_idx * THREAD_NUM) / VECTORIZED_BLOCK_TILE_SIZE_N};
        size_t block_tile_N_id{(thread_id + load_idx * THREAD_NUM) % VECTORIZED_BLOCK_TILE_SIZE_N * NUM_VECTOR_UNITS};

        size_t M_id{block_tile_M_id + block_tile_start_m};
        size_t N_id{block_tile_N_id + block_tile_start_n};

        for (size_t i = 0; i < 8; i++) {
            odata[M_id + (N_id + i) * M] = block_tile[block_tile_M_id][block_tile_N_id + i];
        }
    }

}


template<size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t THREAD_NUM>
__global__ void matrix_transpose_kernel_v3(const __nv_bfloat16* idata, __nv_bfloat16* odata, int M, int N)
{
    // Get index
    size_t block_id_m = blockIdx.y;
    size_t block_id_n = blockIdx.x;
    size_t thread_id = threadIdx.x + threadIdx.y * blockDim.x;

    // load matrix from global memory to shared memory using vector fetching
    __shared__ __nv_bfloat16 block_tile[BLOCK_TILE_SIZE_N][BLOCK_TILE_SIZE_M + 16];

    static_assert(sizeof(VECTYPE) % sizeof(__nv_bfloat16) == 0, "VECTYPE must be a multiple of T size");
    constexpr size_t NUM_VECTOR_UNITS{sizeof(VECTYPE) / sizeof(__nv_bfloat16)};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_N{BLOCK_TILE_SIZE_N / NUM_VECTOR_UNITS};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_M{BLOCK_TILE_SIZE_M / NUM_VECTOR_UNITS};

    static_assert(BLOCK_TILE_SIZE_M % NUM_VECTOR_UNITS == 0);
    static_assert(BLOCK_TILE_SIZE_N % NUM_VECTOR_UNITS == 0);

    size_t block_tile_start_m = block_id_m * BLOCK_TILE_SIZE_M;
    size_t block_tile_start_n = block_id_n * BLOCK_TILE_SIZE_N;


    size_t block_tile_M_id{(thread_id ) / VECTORIZED_BLOCK_TILE_SIZE_N};
    size_t block_tile_N_id{(thread_id ) % VECTORIZED_BLOCK_TILE_SIZE_N * NUM_VECTOR_UNITS};

    size_t M_id{block_tile_M_id + block_tile_start_m};
    size_t N_id{block_tile_N_id + block_tile_start_n};

    VECTYPE row_vector_vals;

    row_vector_vals = *reinterpret_cast<VECTYPE const*>(&idata[M_id * N + N_id]);
    // *reinterpret_cast<VECTYPE*>(&block_tile[block_tile_M_id][block_tile_N_id]) = row_vector_vals.vec;
    #pragma unroll
    for (size_t i = 0; i < NUM_VECTOR_UNITS; i++) {
        block_tile[block_tile_N_id + i][block_tile_M_id] = reinterpret_cast<__nv_bfloat16 const*>(&row_vector_vals)[i];
    }
    __syncthreads();

    block_tile_N_id = (thread_id) / VECTORIZED_BLOCK_TILE_SIZE_M;
    block_tile_M_id = (thread_id) % VECTORIZED_BLOCK_TILE_SIZE_M * NUM_VECTOR_UNITS;

    M_id = block_tile_M_id + block_tile_start_m;
    N_id = block_tile_N_id + block_tile_start_n;

    row_vector_vals = *reinterpret_cast<VECTYPE const*>(&block_tile[block_tile_N_id][block_tile_M_id]);
    *reinterpret_cast<VECTYPE*>(&odata[N_id * M + M_id]) = row_vector_vals;
}




template <typename T>
void matrix_transpose_v1(const T* idata, T* odata, int M, int N) {
    
    if constexpr (sizeof(T) == 2) {
        constexpr size_t BLOCK_TILE_SIZE_M = 128;
        constexpr size_t BLOCK_TILE_SIZE_N = 128;
        constexpr size_t WARP_TILE_SIZE_M = 32;
        constexpr size_t WARP_TILE_SIZE_N = 32;
        constexpr size_t MAT_TILE_SIZE_M = 16;
        constexpr size_t MAT_TILE_SIZE_N = 16;
        constexpr size_t THREAD_NUM = (BLOCK_TILE_SIZE_M / WARP_TILE_SIZE_M) * (BLOCK_TILE_SIZE_N / WARP_TILE_SIZE_N) * WARP_SIZE;
        
        dim3 blockDim(THREAD_NUM, 1, 1);
        dim3 gridDim((N + BLOCK_TILE_SIZE_N - 1) / BLOCK_TILE_SIZE_N, (M + BLOCK_TILE_SIZE_M - 1) / BLOCK_TILE_SIZE_M, 1);

        matrix_transpose_kernel_v1<BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, WARP_TILE_SIZE_M, WARP_TILE_SIZE_N, MAT_TILE_SIZE_M, MAT_TILE_SIZE_N, THREAD_NUM><<<gridDim, blockDim>>>((__nv_bfloat16*)idata, (__nv_bfloat16*)odata, M, N);
    } else {
        std::cerr << "Only support bf16 data type now." << std::endl;
        return;
    }
}


template <typename T, size_t BLOCK_TILE_SIZE_X = 128,
          size_t BLOCK_TILE_SIZE_Y = 128>
__global__ void transpose_swizzling(T* output_matrix, T const* input_matrix,
                                    size_t M, size_t N)
{
    __shared__ T shm[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_X];

    // In some algorithms, such as matrix multiplication,
    // a warp of threads have to access a column of the 2D matrix in the shared
    // memory. Using the conventional index mapping, if the column size is not a
    // multiple of the warp size, there will be bank conflicts.
    size_t const input_matrix_from_idx_x{threadIdx.x + blockIdx.x * blockDim.x};
    size_t const input_matrix_from_idx_y{threadIdx.y + blockIdx.y * blockDim.y};
    size_t const input_matrix_from_idx{input_matrix_from_idx_x +
                                       input_matrix_from_idx_y * N};
    size_t const shm_to_idx_x{threadIdx.x};
    size_t const shm_to_idx_y{threadIdx.y};
    size_t const shm_to_idx_x_swizzled{(shm_to_idx_x ^ shm_to_idx_y) %
                                       BLOCK_TILE_SIZE_X};

    if ((input_matrix_from_idx_y < M) && (input_matrix_from_idx_x < N))
    {
        // Coalesced global memory access.
        // No shared memory bank conflict.
        shm[shm_to_idx_y][shm_to_idx_x_swizzled] =
            input_matrix[input_matrix_from_idx];
    }

    // Make sure the buffer in a block is filled.
    __syncthreads();

    size_t const block_thread_idx{threadIdx.x + threadIdx.y * blockDim.x};
    size_t const shm_from_idx_x{block_thread_idx / BLOCK_TILE_SIZE_Y};
    size_t const shm_from_idx_y{block_thread_idx % BLOCK_TILE_SIZE_Y};
    size_t const shm_from_idx_x_swizzled{(shm_from_idx_x ^ shm_from_idx_y) %
                                         BLOCK_TILE_SIZE_X};
    size_t const output_matrix_to_idx_x{shm_from_idx_y +
                                        blockIdx.y * blockDim.y};
    size_t const output_matrix_to_idx_y{shm_from_idx_x +
                                        blockIdx.x * blockDim.x};
    size_t const output_matrix_to_idx{output_matrix_to_idx_x +
                                      output_matrix_to_idx_y * M};

    if ((output_matrix_to_idx_y < N) && (output_matrix_to_idx_x < M))
    {
        // Coalesced global memory access.
        // No shared memory bank conflict.
        output_matrix[output_matrix_to_idx] =
            shm[shm_from_idx_y][shm_from_idx_x_swizzled];
    }
}


template <typename T>
void launch_transpose_without_shm_bank_conflict_via_swizzling(
    T* d_output_matrix, T const* d_input_matrix, size_t M, size_t N)
{
    constexpr size_t BLOCK_TILE_SIZE_X{32};
    constexpr size_t BLOCK_TILE_SIZE_Y{32};
    dim3 const block_size{BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y};
    // dim3 const grid_size{static_cast<unsigned int>(div_up(N, block_size.x)),
    //                      static_cast<unsigned int>(div_up(M, block_size.y))};
    dim3 const grid_size((N + BLOCK_TILE_SIZE_X - 1) / BLOCK_TILE_SIZE_X, (M + BLOCK_TILE_SIZE_Y - 1) / BLOCK_TILE_SIZE_Y, 1);

    transpose_swizzling<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y><<<grid_size, block_size>>>(
        d_output_matrix, d_input_matrix, M, N);
}

template <typename T>
void matrix_transpose_v2(const T* idata, T* odata, int M, int N) {
    if constexpr (sizeof(T) == 2) {
        constexpr size_t THREAD_NUM = 256;
        constexpr size_t BLOCK_TILE_SIZE_M = 64;
        constexpr size_t BLOCK_TILE_SIZE_N = 32;
        
        dim3 blockDim(THREAD_NUM, 1, 1);
        dim3 gridDim((N + BLOCK_TILE_SIZE_N - 1) / BLOCK_TILE_SIZE_N, (M + BLOCK_TILE_SIZE_M - 1) / BLOCK_TILE_SIZE_M, 1);

        matrix_transpose_kernel_v3<BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, THREAD_NUM><<<gridDim, blockDim>>>((__nv_bfloat16*)idata, (__nv_bfloat16*)odata, M, N);
    } else {
        std::cerr << "Only support bf16 data type now." << std::endl;
        return;
    }
    // launch_transpose_without_shm_bank_conflict_via_swizzling(odata, idata, M, N);

}

#ifdef LOCAL_TEST
#include <iostream>
#include <vector>
#include <random>
#include <cassert>

void test_matrix_transpose() {
    const int M = 8192;
    const int N = 8192;
    const size_t size = M * N;
    
    // 分配主机内存
    std::vector<__nv_bfloat16> h_input(size);
    std::vector<__nv_bfloat16> h_output(size);
    std::vector<__nv_bfloat16> h_reference(size);
    
    // 初始化输入数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < size; ++i) {
        h_input[i] = __float2bfloat16(dis(gen) + i);
    }
    
    // 计算参考结果（CPU转置）
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            h_reference[j * M + i] = h_input[i * N + j];
        }
    }
    
    // 分配GPU内存
    __nv_bfloat16* d_input;
    __nv_bfloat16* d_output;
    
    cudaMalloc(&d_input, size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_output, size * sizeof(__nv_bfloat16));
    
    // 复制数据到GPU
    cudaMemcpy(d_input, h_input.data(), size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    
    // 执行GPU转置
    size_t TESTS = 10;

    // 创建CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (size_t i = 0; i < TESTS; i++) {
        matrix_transpose_v2(d_input, d_output, M, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // 计算执行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float average_time_ms = milliseconds / TESTS;
    float average_time_us = average_time_ms * 1000.0f;

    auto data_size = size * sizeof(__nv_bfloat16);
    double GB_per_second = (double)data_size / (double)(average_time_ms * 1e-3) / (double)1e9;
    std::cout << "GPU transpose time: " << average_time_us << " us" << std::endl;
    std::cout << "GPU transpose bandwidth: " << GB_per_second << " GB/s" << std::endl;

    // 清理CUDA事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 复制结果回主机
    cudaMemcpy(h_output.data(), d_output, size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    
    // 验证结果
    bool passed = true;
    float max_error = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float gpu_val = __bfloat162float(h_output[i]);
        float ref_val = __bfloat162float(h_reference[i]);
        float error = std::abs(gpu_val - ref_val);
        max_error = std::max(max_error, error);
        
        if (error > 1e-3f) {
            passed = false;
            std::cout << "Mismatch at index " << i << ": GPU=" << gpu_val 
                      << ", Reference=" << ref_val << ", Error=" << error << std::endl;
            break;
        }
    }
    
    if (passed) {
        std::cout << "Test PASSED! Max error: " << max_error << std::endl;
    } else {
        std::cout << "Test FAILED!" << std::endl;
    }
    
    // 清理内存
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    std::cout << "Starting matrix transpose test..." << std::endl;

    // 检查CUDA设备
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return -1;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    // 运行测试
    test_matrix_transpose();
    
    std::cout << "Test completed." << std::endl;
    return 0;
}
#endif