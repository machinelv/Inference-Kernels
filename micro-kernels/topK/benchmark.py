import torch
import topk_triton 
import topk_cuda
import time
import pandas as pd

# def timeit(func):
#     def wrapper(*args, **kwargs):
#         import time
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         time_taken = (end_time - start_time) * 1000  # Convert to milliseconds
#         return result
#     return wrapper

# @timeit
def topk_pytorch(input, k, dim=None, largest=True, sorted=True):
    """
    Invokes the PyTorch built-in topk function.
    """
    if dim is None:
        dim = -1
    if largest:
        return torch.topk(input, k, dim=dim, largest=largest, sorted=sorted)
    else:
        return torch.topk(input, k, dim=dim, largest=largest, sorted=sorted)
    
def check_results(ref, test, sorted=False):
    """
    
    """

    pass

def benchmark():
    header = ["Kernel", "Input Size", "K", "Time (ms)"]
    ref_kernel = topk_pytorch
    bench_kernel_list = [
        topk_triton.topk,
        topk_cuda.topk,
    ]
    # Generate some random input data
    input_size_list = [1000, 2000, 3000]  # Different input sizes to benchmark
    k_list = [10, 50, 100]  # Different k values to benchmark
    result_df = pd.DataFrame(columns=header)

    for input_size in input_size_list:
        input_tensor = torch.randn(input_size, input_size, device='cuda')
        for k in k_list:
            print(f"Benchmarking with input size {input_size} and k={k}...")

            start_time = time.time()
            ref_output = ref_kernel(input_tensor, k, dim=-1, largest=True, sorted=True)
            end_time = time.time()
            ref_time = (end_time - start_time) * 1000
            result_df = result_df.append({
                "Kernel": "PyTorch",
                "Input Size": input_size,
                "K": k,
                "Time (ms)": ref_time
            }, ignore_index=True)

            for kernel in bench_kernel_list:
                print(f"Benchmarking {kernel.__name__}...")
                # Here you would implement the actual benchmarking logic
                start_time = time.time()
                output = kernel(input_tensor, k, dim=-1, largest=True, sorted=True)
                end_time = time.time()
                time_taken = (end_time - start_time) * 1000
                result_df = result_df.append({
                    "Kernel": kernel.__name__,
                    "Input Size": input_size,
                    "K": k,
                    "Time (ms)": time_taken
                }, ignore_index=True)
                print(f"{kernel.__name__} benchmark completed.")
    print("Benchmarking completed.")
    return result_df

def plot_results(result_df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    unique_k_values = result_df["K"].unique()
    kernels = result_df["Kernel"].unique()

    for k in unique_k_values:
        plt.figure(figsize=(10, 6))
        subset = result_df[result_df["K"] == k]
        for kernel in kernels:
            kernel_data = subset[subset["Kernel"] == kernel]
            plt.plot(kernel_data["Input Size"], kernel_data["Time (ms)"], marker='o', label=kernel)

        plt.title(f"Benchmark Results for K={k}")
        plt.xlabel("Input Size")
        plt.ylabel("Time (ms)")
        plt.legend(title="Kernel")
        plt.tight_layout()
        plt.show()
