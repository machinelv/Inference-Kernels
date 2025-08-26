all:
	nvcc -O3 -DUSE_CUDA -DLOCAL_TEST  -o ref ref.cu -arch=sm_89 --std=c++17
	nvcc -O3 -DUSE_CUDA -DLOCAL_TEST  -o test_transpose transpose.cu -arch=sm_89 --std=c++17