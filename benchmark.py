#!/usr/bin/env python3
"""
Performance Benchmark for codec5jchain
Optimized for Windows 10 with NVIDIA RTX 3050 and AMD Ryzen 5 5600X

This script benchmarks CPU and GPU performance.
"""

import sys
import time
import platform
import multiprocessing as mp
from datetime import datetime

try:
    import numpy as np
except ImportError:
    print("Error: numpy not installed. Run: pip install numpy")
    sys.exit(1)

try:
    import psutil
except ImportError:
    print("Warning: psutil not installed. Some features disabled.")
    psutil = None

# Check for PyTorch (for GPU benchmarks)
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    print("Warning: PyTorch not installed. GPU benchmarks disabled.")
    print("To enable GPU benchmarks, run:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")


class BenchmarkResults:
    """Store and display benchmark results"""
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, name, duration, details=None):
        """Add a benchmark result"""
        self.results[name] = {
            'duration': duration,
            'details': details or {}
        }
    
    def display(self):
        """Display all results"""
        print("\n" + "=" * 80)
        print(f"{'BENCHMARK RESULTS':^80}")
        print("=" * 80)
        
        for name, data in self.results.items():
            print(f"\n{name}:")
            print(f"  Duration: {data['duration']:.4f} seconds")
            
            for key, value in data['details'].items():
                print(f"  {key}: {value}")
        
        print("\n" + "=" * 80)


def cpu_single_thread_benchmark():
    """Benchmark single-threaded CPU performance"""
    print("\nRunning Single-Thread CPU Benchmark...")
    
    size = 2000
    start_time = time.time()
    
    # Matrix multiplication
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    c = np.dot(a, b)
    
    duration = time.time() - start_time
    
    flops = 2 * size**3  # Approximate FLOPs for matrix multiplication
    gflops = (flops / duration) / 1e9
    
    return duration, {
        'Matrix Size': f"{size}x{size}",
        'Performance': f"{gflops:.2f} GFLOPS"
    }


def cpu_multi_thread_benchmark():
    """Benchmark multi-threaded CPU performance on Ryzen 5 5600X"""
    print("Running Multi-Thread CPU Benchmark...")
    
    num_threads = mp.cpu_count()
    
    def worker(n):
        """Worker function for parallel processing"""
        result = 0
        for i in range(n):
            result += i * i
        return result
    
    start_time = time.time()
    
    # Use all available threads
    with mp.Pool(num_threads) as pool:
        results = pool.map(worker, [1000000] * num_threads)
    
    duration = time.time() - start_time
    
    return duration, {
        'Threads Used': num_threads,
        'Total Operations': f"{sum(results):,}"
    }


def cpu_numpy_benchmark():
    """Benchmark NumPy operations"""
    print("Running NumPy Operations Benchmark...")
    
    size = 5000
    iterations = 10
    
    start_time = time.time()
    
    for _ in range(iterations):
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        
        # Various NumPy operations
        c = np.dot(a, b)
        d = np.linalg.inv(a + np.eye(size) * 0.1)
        e = np.fft.fft2(a)
    
    duration = time.time() - start_time
    avg_duration = duration / iterations
    
    return duration, {
        'Matrix Size': f"{size}x{size}",
        'Iterations': iterations,
        'Avg per Iteration': f"{avg_duration:.4f} seconds"
    }


def gpu_benchmark():
    """Benchmark GPU performance on NVIDIA RTX 3050"""
    
    if not TORCH_AVAILABLE or not CUDA_AVAILABLE:
        print("\nGPU Benchmark: SKIPPED (CUDA not available)")
        return None, {}
    
    print("\nRunning GPU Benchmark on NVIDIA RTX 3050...")
    
    device = torch.device("cuda")
    
    # Display GPU information
    print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  PyTorch Version: {torch.__version__}")
    
    results = {}
    
    # Benchmark 1: Matrix Multiplication
    print("  - Matrix Multiplication...")
    size = 4096
    
    a = torch.rand(size, size, device=device)
    b = torch.rand(size, size, device=device)
    
    # Warm-up
    _ = torch.mm(a, b)
    torch.cuda.synchronize()
    
    # Actual benchmark
    start_time = time.time()
    c = torch.mm(a, b)
    torch.cuda.synchronize()
    mm_duration = time.time() - start_time
    
    flops = 2 * size**3
    tflops = (flops / mm_duration) / 1e12
    
    results['Matrix Multiplication'] = {
        'Matrix Size': f"{size}x{size}",
        'Duration': f"{mm_duration:.4f} seconds",
        'Performance': f"{tflops:.2f} TFLOPS"
    }
    
    # Benchmark 2: Tensor Operations
    print("  - Tensor Operations...")
    size = 1024
    batch_size = 128
    
    start_time = time.time()
    
    for _ in range(100):
        x = torch.rand(batch_size, size, size, device=device)
        y = torch.relu(x)
        z = torch.sigmoid(y)
        w = torch.softmax(z, dim=-1)
    
    torch.cuda.synchronize()
    tensor_duration = time.time() - start_time
    
    results['Tensor Operations'] = {
        'Batch Size': batch_size,
        'Iterations': 100,
        'Duration': f"{tensor_duration:.4f} seconds"
    }
    
    # Memory usage
    mem_allocated = torch.cuda.memory_allocated() / (1024**2)
    mem_reserved = torch.cuda.memory_reserved() / (1024**2)
    
    results['Memory Usage'] = {
        'Allocated': f"{mem_allocated:.2f} MB",
        'Reserved': f"{mem_reserved:.2f} MB"
    }
    
    total_duration = mm_duration + tensor_duration
    
    return total_duration, results


def memory_benchmark():
    """Benchmark memory operations"""
    print("\nRunning Memory Benchmark...")
    
    if not psutil:
        return None, {}
    
    mem_before = psutil.virtual_memory()
    
    # Allocate large arrays
    size = 1000
    arrays = []
    
    start_time = time.time()
    
    for i in range(100):
        arr = np.random.rand(size, size)
        arrays.append(arr)
    
    duration = time.time() - start_time
    
    mem_after = psutil.virtual_memory()
    mem_used = (mem_after.used - mem_before.used) / (1024**2)
    
    # Clean up
    del arrays
    
    return duration, {
        'Arrays Created': 100,
        'Array Size': f"{size}x{size}",
        'Memory Used': f"{mem_used:.2f} MB"
    }


def main():
    """Main benchmark function"""
    print("=" * 80)
    print(f"{'CODEC5JCHAIN PERFORMANCE BENCHMARK':^80}")
    print(f"{'Windows 10 + NVIDIA RTX 3050 + AMD Ryzen 5 5600X':^80}")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python Version: {platform.python_version()}")
    
    if psutil:
        print(f"CPU Cores: {psutil.cpu_count(logical=False)}")
        print(f"CPU Threads: {psutil.cpu_count(logical=True)}")
        print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    print("\n" + "=" * 80)
    
    results = BenchmarkResults()
    
    # CPU Benchmarks
    print("\n" + "─" * 80)
    print("CPU BENCHMARKS (AMD Ryzen 5 5600X)")
    print("─" * 80)
    
    duration, details = cpu_single_thread_benchmark()
    results.add_result("CPU Single-Thread Matrix Multiplication", duration, details)
    
    duration, details = cpu_multi_thread_benchmark()
    results.add_result("CPU Multi-Thread Processing", duration, details)
    
    duration, details = cpu_numpy_benchmark()
    results.add_result("CPU NumPy Operations", duration, details)
    
    # Memory Benchmark
    print("\n" + "─" * 80)
    print("MEMORY BENCHMARKS")
    print("─" * 80)
    
    duration, details = memory_benchmark()
    if duration:
        results.add_result("Memory Allocation", duration, details)
    
    # GPU Benchmark
    print("\n" + "─" * 80)
    print("GPU BENCHMARKS (NVIDIA RTX 3050)")
    print("─" * 80)
    
    duration, details = gpu_benchmark()
    if duration:
        results.add_result("GPU Operations", duration, details)
        
        for name, data in details.items():
            print(f"\n{name}:")
            for key, value in data.items():
                print(f"  {key}: {value}")
    
    # Display all results
    results.display()
    
    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
