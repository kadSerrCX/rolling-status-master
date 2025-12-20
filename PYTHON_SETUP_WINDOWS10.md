# Python Setup Guide for Windows 10 (NVIDIA RTX 3050 + AMD Ryzen 5 5600X)

## System Requirements

### Hardware Configuration
- **CPU**: AMD Ryzen 5 5600X (6-core, 12-thread processor)
- **GPU**: NVIDIA RTX 3050 (8GB GDDR6)
- **OS**: Windows 10 (64-bit)
- **RAM**: Minimum 16GB recommended for optimal performance

## Project Objectives

This project (`codec5jchain`) focuses on:
1. **Build Automation**: Facilitating automated builds and workflow management
2. **UX/UI Development**: Creating efficient user experience interfaces
3. **Cross-Platform Compatibility**: Supporting multiple architectures (x86, x64, ARM)
4. **Integration**: Working with ASP.NET, C/C++, SQLite, and other frameworks

## Python Installation

### Step 1: Install Python 3.11 or Higher

1. **Download Python**:
   - Visit: https://www.python.org/downloads/windows/
   - Download Python 3.11+ (64-bit) - recommended for AMD64 architecture
   - **Important**: Select "Add Python to PATH" during installation

2. **Verify Installation**:
   ```cmd
   python --version
   pip --version
   ```

### Step 2: Configure Python for NVIDIA RTX 3050

#### Install CUDA Toolkit for GPU Acceleration

1. **Download NVIDIA CUDA Toolkit**:
   - Visit: https://developer.nvidia.com/cuda-downloads
   - Select: Windows > x86_64 > 10 > exe (network)
   - Version: CUDA 12.x or compatible with RTX 3050

2. **Install cuDNN** (for deep learning):
   ```cmd
   # Download cuDNN from NVIDIA Developer site
   # Extract to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x
   ```

3. **Verify CUDA Installation**:
   ```cmd
   nvcc --version
   nvidia-smi
   ```

### Step 3: Optimize for AMD Ryzen 5 5600X

#### Configure Python for Multi-threading

1. **Set CPU Affinity** (create a Python script):
   ```python
   import os
   import psutil
   
   # Use all 12 threads of Ryzen 5 5600X
   process = psutil.Process()
   process.cpu_affinity(list(range(os.cpu_count())))
   ```

2. **Enable Performance Mode**:
   ```cmd
   # Run PowerShell as Administrator
   powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
   ```

## Python Environment Setup

### Create Virtual Environment

```cmd
# Navigate to project directory
cd C:\path\to\codec5jchain

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

### Install Required Packages

```cmd
# Install essential packages
pip install -r requirements.txt

# For GPU acceleration (NVIDIA RTX 3050)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For data processing
pip install numpy pandas scipy

# For UI/UX development
pip install PyQt6 PySide6 customtkinter

# For build automation
pip install invoke pybuilder

# For XML processing (project uses XML configs)
pip install lxml xmltodict

# For SQLite database operations
pip install sqlalchemy

# Performance monitoring
pip install psutil gputil py-cpuinfo
```

## Hardware-Specific Optimizations

### NVIDIA RTX 3050 Configuration

#### Enable GPU Acceleration in Python

```python
# Check GPU availability
import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Device: {torch.cuda.get_device_name(0)}")

# Set default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Monitor GPU usage
import GPUtil
gpus = GPUtil.getGPUs()
for gpu in gpus:
    print(f"GPU {gpu.id}: {gpu.name}")
    print(f"  Memory Total: {gpu.memoryTotal}MB")
    print(f"  Memory Used: {gpu.memoryUsed}MB")
    print(f"  Memory Free: {gpu.memoryFree}MB")
    print(f"  GPU Load: {gpu.load*100}%")
```

#### NVIDIA Control Panel Settings

1. Open **NVIDIA Control Panel**
2. Navigate to **Manage 3D Settings**
3. Set **Power Management Mode** to "Prefer Maximum Performance"
4. Set **CUDA - GPUs** to use RTX 3050
5. Apply changes

### AMD Ryzen 5 5600X Configuration

#### Optimize CPU Performance

```python
import os
import multiprocessing as mp

# Utilize all 6 cores / 12 threads
num_cores = mp.cpu_count()  # Should return 12
print(f"Available CPU threads: {num_cores}")

# Set environment variables for multi-threading
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_cores)

# For parallel processing
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=num_cores) as executor:
    # Your parallel tasks here
    pass
```

#### Windows Power Settings

```cmd
# Set High Performance power plan
powercfg /list
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# Disable CPU parking for better performance
# Run in PowerShell as Administrator:
# Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\Power\PowerSettings\54533251-82be-4824-96c1-47b60b740d00\0cc5b647-c1df-4637-891a-dec35c318583" -Name "ValueMax" -Value 0
```

## UX Customization Scripts

### Example: System Monitor Dashboard

Create `system_monitor.py`:

```python
import psutil
import GPUtil
import time
import os

def monitor_system():
    """Monitor CPU, GPU, Memory, and Disk usage"""
    
    # CPU Information
    print(f"\n{'='*50}")
    print(f"AMD Ryzen 5 5600X - CPU Monitoring")
    print(f"{'='*50}")
    print(f"CPU Cores: {psutil.cpu_count(logical=False)}")
    print(f"CPU Threads: {psutil.cpu_count(logical=True)}")
    print(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")
    
    # Per-core usage
    per_cpu = psutil.cpu_percent(interval=1, percpu=True)
    for i, usage in enumerate(per_cpu):
        print(f"  Thread {i}: {usage}%")
    
    # Memory Information
    memory = psutil.virtual_memory()
    print(f"\nMemory Usage: {memory.percent}%")
    print(f"  Total: {memory.total / (1024**3):.2f} GB")
    print(f"  Used: {memory.used / (1024**3):.2f} GB")
    print(f"  Free: {memory.available / (1024**3):.2f} GB")
    
    # GPU Information (NVIDIA RTX 3050)
    print(f"\n{'='*50}")
    print(f"NVIDIA RTX 3050 - GPU Monitoring")
    print(f"{'='*50}")
    
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.name}")
            print(f"  Load: {gpu.load*100:.1f}%")
            print(f"  Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB ({gpu.memoryUtil*100:.1f}%)")
            print(f"  Temperature: {gpu.temperature}°C")
    except Exception as e:
        print(f"GPU monitoring unavailable: {e}")
    
    # Disk Information
    disk = psutil.disk_usage('C:')
    print(f"\nDisk Usage (C:): {disk.percent}%")
    print(f"  Total: {disk.total / (1024**3):.2f} GB")
    print(f"  Used: {disk.used / (1024**3):.2f} GB")
    print(f"  Free: {disk.free / (1024**3):.2f} GB")

if __name__ == "__main__":
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        monitor_system()
        time.sleep(5)  # Update every 5 seconds
```

### Example: Performance Benchmark

Create `benchmark.py`:

```python
import time
import numpy as np
import multiprocessing as mp

def cpu_benchmark():
    """Benchmark CPU performance on Ryzen 5 5600X"""
    print("Running CPU Benchmark...")
    
    # Matrix multiplication benchmark
    size = 2000
    start = time.time()
    
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    c = np.dot(a, b)
    
    end = time.time()
    print(f"Matrix multiplication ({size}x{size}): {end - start:.2f} seconds")
    
    # Multi-threaded benchmark
    def worker(n):
        return sum(i*i for i in range(n))
    
    start = time.time()
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(worker, [1000000] * 12)
    end = time.time()
    
    print(f"Multi-threaded computation (12 threads): {end - start:.2f} seconds")

def gpu_benchmark():
    """Benchmark GPU performance on RTX 3050"""
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("CUDA not available. Skipping GPU benchmark.")
            return
        
        print("\nRunning GPU Benchmark...")
        
        # Matrix multiplication on GPU
        size = 4096
        device = torch.device("cuda")
        
        a = torch.rand(size, size, device=device)
        b = torch.rand(size, size, device=device)
        
        # Warm-up
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        end = time.time()
        
        print(f"GPU Matrix multiplication ({size}x{size}): {end - start:.4f} seconds")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
        
    except ImportError:
        print("PyTorch not installed. Install with: pip install torch --index-url https://download.pytorch.org/whl/cu121")

if __name__ == "__main__":
    cpu_benchmark()
    gpu_benchmark()
```

## Troubleshooting

### Common Issues

#### 1. CUDA Not Found
```cmd
# Check CUDA installation
nvcc --version
nvidia-smi

# Reinstall CUDA Toolkit if needed
# Verify environment variables:
echo %CUDA_PATH%
echo %CUDA_PATH_V12_X%
```

#### 2. Python Not in PATH
```cmd
# Add Python to PATH manually
# System Properties > Environment Variables > PATH
# Add: C:\Users\YourUser\AppData\Local\Programs\Python\Python311
# Add: C:\Users\YourUser\AppData\Local\Programs\Python\Python311\Scripts
```

#### 3. DLL Load Failed
```cmd
# Install Visual C++ Redistributables
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
```

#### 4. GPU Not Detected
```cmd
# Update NVIDIA drivers
# Visit: https://www.nvidia.com/Download/index.aspx
# Select: RTX 3050 > Windows 10 64-bit
```

## Performance Tips

### 1. Memory Management
- Monitor memory usage with `psutil` or Task Manager
- Use generators for large datasets
- Clear GPU cache: `torch.cuda.empty_cache()`

### 2. Multi-threading
- Use `multiprocessing` for CPU-bound tasks
- Use `threading` for I/O-bound tasks
- Utilize all 12 threads of Ryzen 5 5600X

### 3. GPU Optimization
- Batch operations when possible
- Keep data on GPU to avoid transfer overhead
- Use mixed precision training: `torch.cuda.amp`

### 4. Disk I/O
- Use SSD for better performance
- Enable write caching in Device Manager
- Defragment HDD regularly (not needed for SSD)

## Next Steps

1. Run `system_monitor.py` to verify hardware detection
2. Run `benchmark.py` to test performance
3. Install project-specific dependencies from `requirements.txt`
4. Configure project settings in XML files
5. Start developing UX/UI components

## Additional Resources

- **NVIDIA Developer**: https://developer.nvidia.com/
- **AMD Ryzen Documentation**: https://www.amd.com/en/products/processors/desktops/ryzen
- **Python Documentation**: https://docs.python.org/3/
- **PyTorch GPU Guide**: https://pytorch.org/get-started/locally/
