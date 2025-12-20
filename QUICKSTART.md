# Quick Start Guide - codec5jchain UX Customization

## For Windows 10 with NVIDIA RTX 3050 + AMD Ryzen 5 5600X

### Step 1: Install Python

1. Download Python 3.11 or higher from https://www.python.org/downloads/
2. During installation, check "Add Python to PATH"
3. Verify installation:
   ```cmd
   python --version
   pip --version
   ```

### Step 2: Install NVIDIA CUDA Toolkit (for GPU acceleration)

1. Download CUDA Toolkit 12.x from https://developer.nvidia.com/cuda-downloads
2. Install with default settings
3. Verify installation:
   ```cmd
   nvcc --version
   nvidia-smi
   ```

### Step 3: Set Up Python Environment

```cmd
# Navigate to project directory
cd C:\path\to\codec5jchain

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (for RTX 3050)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Verify Installation

```cmd
# Run system monitor to check hardware detection
python system_monitor.py
```

Expected output:
- AMD Ryzen 5 5600X detected with 6 cores / 12 threads
- NVIDIA RTX 3050 GPU detected
- Memory and disk information displayed
- Real-time performance metrics

### Step 5: Run Performance Benchmark

```cmd
# Run benchmark to test system performance
python benchmark.py
```

This will:
- Test CPU performance (single and multi-threaded)
- Test GPU performance (if CUDA is available)
- Test memory operations
- Display performance metrics

### Step 6: Optimize Windows 10 for Performance

#### Enable High Performance Power Plan
```cmd
# Run in Command Prompt as Administrator
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
```

#### Configure NVIDIA Settings
1. Right-click on desktop → NVIDIA Control Panel
2. Navigate to "Manage 3D Settings"
3. Set "Power Management Mode" to "Prefer Maximum Performance"
4. Set "CUDA - GPUs" to use RTX 3050
5. Click "Apply"

#### Disable CPU Parking (for Ryzen 5 5600X)
1. Open Registry Editor (Win + R, type "regedit")
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Power\PowerSettings\54533251-82be-4824-96c1-47b60b740d00\0cc5b647-c1df-4637-891a-dec35c318583`
3. Set "ValueMax" to 0 (requires administrator privileges)

### Common Tasks

#### Monitor System Performance
```cmd
python system_monitor.py
```
Press Ctrl+C to exit

#### Run Benchmarks
```cmd
python benchmark.py
```

#### Install Additional Packages
```cmd
# Activate virtual environment first
venv\Scripts\activate

# Install package
pip install package-name
```

#### Update Dependencies
```cmd
pip install --upgrade -r requirements.txt
```

### Troubleshooting

#### Issue: "Python not recognized"
**Solution**: Add Python to PATH manually
- System Properties → Environment Variables → PATH
- Add: `C:\Users\YourUser\AppData\Local\Programs\Python\Python311`
- Add: `C:\Users\YourUser\AppData\Local\Programs\Python\Python311\Scripts`

#### Issue: "CUDA not available"
**Solution**: 
1. Install/Update NVIDIA drivers from https://www.nvidia.com/Download/index.aspx
2. Install CUDA Toolkit
3. Verify with `nvidia-smi`

#### Issue: "DLL load failed"
**Solution**: Install Visual C++ Redistributable
- Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

#### Issue: GPU not detected in Python
**Solution**:
```python
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Should show RTX 3050
```

If False:
1. Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
2. Check CUDA_PATH environment variable
3. Restart computer

### Performance Tips

1. **Use all CPU threads**: The Ryzen 5 5600X has 12 threads - use them!
   ```python
   import multiprocessing as mp
   num_workers = mp.cpu_count()  # Returns 12
   ```

2. **Keep data on GPU**: Minimize CPU-GPU transfers
   ```python
   import torch
   device = torch.device("cuda")
   data = data.to(device)  # Keep on GPU
   ```

3. **Monitor resource usage**: Use system_monitor.py to track usage
   - CPU usage per thread
   - GPU utilization
   - Memory consumption

4. **Batch operations**: Process data in batches for better GPU utilization
   ```python
   batch_size = 128  # Adjust based on GPU memory
   ```

### Next Steps

1. ✅ Install Python and dependencies
2. ✅ Verify hardware detection
3. ✅ Run benchmarks
4. ✅ Optimize Windows settings
5. 📝 Read full documentation in `PYTHON_SETUP_WINDOWS10.md`
6. 🚀 Start developing!

### Additional Resources

- Full Setup Guide: `PYTHON_SETUP_WINDOWS10.md`
- System Monitor: `system_monitor.py`
- Benchmark Tool: `benchmark.py`
- Dependencies: `requirements.txt`

### Support

For issues or questions:
1. Check `PYTHON_SETUP_WINDOWS10.md` for detailed instructions
2. Review troubleshooting section above
3. Verify all dependencies are installed: `pip list`

---

**Project**: codec5jchain  
**Target Platform**: Windows 10 64-bit  
**Hardware**: NVIDIA RTX 3050 + AMD Ryzen 5 5600X  
**Python Version**: 3.11+
