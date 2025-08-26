#include <iostream>
#include <cstdlib>

#include "gpu_lib.h"
#include "gpu_type.h"
#include "matrix_transpose.h"


#define VECTYPE float4


template<size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t WARP_TILE_SIZE_M, size_t WARP_TILE_SIZE_N, size_t MAT_TILE_SIZE_M, size_t MAT_TILE_SIZE_N, size_t THREAD_NUM>
__global__ void matrix_tranpose_v1(const __nv_bfloat16* idata, __nv_bfloat16* odata, int M, int N) {
    // Get index
    size_t block_id_m = blockIdx.y;
    size_t block_id_n = blockIdx.x;
    size_t thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    size_t warp_id = thread_id / WARP_SIZE;
    size_t lane_id = thread_id % WARP_SIZE;

    size_t warp_id_m = warp_id / (BLOCK_TILE_SIZE_M / WARP_TILE_SIZE_M);
    size_t warp_id_n = warp_id % (BLOCK_TILE_SIZE_N / WARP_TILE_SIZE_N);

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
        if (block_tile_N_id < BLOCK_TILE_SIZE_N && block_tile_M_id < BLOCK_TILE_SIZE_M) {
            *reinterpret_cast<VECTYPE*>(&block_tile[block_tile_M_id][block_tile_N_id]) = row_vector_vals.vec;
        }
    }
    __syncthreads();
    // use movmatrix instruction to transpose the matrix per warp
    constexpr size_t MAT_TILE_PER_WARP_M = WARP_TILE_SIZE_M / MAT_TILE_SIZE_M;
    constexpr size_t MAT_TILE_PER_WARP_N = WARP_TILE_SIZE_N / MAT_TILE_SIZE_N;

    uint32_t warp_tile[MAT_TILE_PER_WARP_N][MAT_TILE_PER_WARP_M][4];

    const size_t smem_n_offset = lane_id / 16 * 8;
    const size_t smem_m_offset = lane_id % 16;

    const size_t warp_tile_start_m = warp_id_m * WARP_TILE_SIZE_M;
    const size_t warp_tile_start_n = warp_id_n * WARP_TILE_SIZE_N;

    #pragma unroll (MAT_TILE_PER_WARP_M)
    for (size_t warp_tile_m{0}; warp_tile_m < MAT_TILE_PER_WARP_M; ++warp_tile_m) {
        #pragma unroll (MAT_TILE_PER_WARP_N)
        for (size_t warp_tile_n{0}; warp_tile_n < MAT_TILE_PER_WARP_N; ++warp_tile_n) {
            size_t src_m = warp_tile_start_m + warp_tile_m * MAT_TILE_SIZE_M;
            size_t src_n = warp_tile_start_n + warp_tile_n * MAT_TILE_SIZE_N;
            uint32_t smem_addr = __cvta_generic_to_shared(&block_tile[src_m + smem_m_offset][src_n + smem_n_offset]);
            LDMATRIX_BF16_TRANSPOSE_X4(warp_tile[warp_tile_n][warp_tile_m][0], warp_tile[warp_tile_n][warp_tile_m][1], warp_tile[warp_tile_n][warp_tile_m][2], warp_tile[warp_tile_n][warp_tile_m][3], smem_addr);
        }
    }

    // store the transposed matrix from register to shared memory
    #pragma unroll (MAT_TILE_PER_WARP_N)
        for (size_t warp_tile_n{0}; warp_tile_n < MAT_TILE_PER_WARP_N; ++warp_tile_n) {
        #pragma unroll (MAT_TILE_PER_WARP_M)
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

template <typename T>
void matrix_transpose_v1(const T* idata, T* odata, int M, int N) {
    
    if constexpr (sizeof(T) == 2) {
        constexpr size_t BLOCK_TILE_SIZE_M = 128;
        constexpr size_t BLOCK_TILE_SIZE_N = 128;
        constexpr size_t WARP_TILE_SIZE_M = 64;
        constexpr size_t WARP_TILE_SIZE_N = 64;
        constexpr size_t MAT_TILE_SIZE_M = 16;
        constexpr size_t MAT_TILE_SIZE_N = 16;
        constexpr size_t THREAD_NUM = (BLOCK_TILE_SIZE_M / WARP_TILE_SIZE_M) * (BLOCK_TILE_SIZE_N / WARP_TILE_SIZE_N) * WARP_SIZE;
        
        dim3 blockDim(THREAD_NUM, 1, 1);
        dim3 gridDim((N + BLOCK_TILE_SIZE_N - 1) / BLOCK_TILE_SIZE_N, (M + BLOCK_TILE_SIZE_M - 1) / BLOCK_TILE_SIZE_M, 1);

        matrix_tranpose_v1<BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, WARP_TILE_SIZE_M, WARP_TILE_SIZE_N, MAT_TILE_SIZE_M, MAT_TILE_SIZE_N, THREAD_NUM><<<gridDim, blockDim>>>((__nv_bfloat16*)idata, (__nv_bfloat16*)odata, M, N);
    } else {
        std::cerr << "Only support bf16 data type now." << std::endl;
        return;
    }
}


#ifdef LOCAL_TEST
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <chrono>

void test_matrix_transpose() {
    const int M = 512;
    const int N = 512;
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
        h_input[i] = __float2bfloat16(dis(gen));
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
    auto TESTS = 10;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TESTS; i++) {
        matrix_transpose_v1(d_input, d_output, M, N);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto data_size = size * sizeof(__nv_bfloat16);

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / TESTS;
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start) / TESTS;
    double GB_per_second = (double)data_size / duration_ns.count() * 1000;
    std::cout << "GPU transpose time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "GPU transpose bandwidth: " << GB_per_second << " GB/s" << std::endl;

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