# Project Overview - codec5jchain

## Customized UX Instructions for Windows 10 with NVIDIA RTX 3050 and AMD Ryzen 5 5600X

### What is codec5jchain?

codec5jchain is a comprehensive project focused on:
- **Build Automation**: Streamlining development workflows
- **UX/UI Development**: Creating efficient user experiences
- **Cross-Platform Support**: Compatible with multiple architectures
- **Performance Optimization**: Leveraging modern hardware capabilities

### Target Hardware Configuration

This setup is specifically optimized for:
- **Operating System**: Windows 10 (64-bit)
- **CPU**: AMD Ryzen 5 5600X (6 cores, 12 threads)
- **GPU**: NVIDIA RTX 3050 (8GB GDDR6, CUDA-capable)
- **RAM**: 16GB+ recommended

### Documentation Structure

#### 📚 Getting Started (Read in Order)

1. **[README.md](README.md)** - Project overview and quick links
2. **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup guide
3. **[PYTHON_SETUP_WINDOWS10.md](PYTHON_SETUP_WINDOWS10.md)** - Complete installation guide
4. **[COMMANDS.md](COMMANDS.md)** - Quick reference for common commands

#### 🛠️ Tools & Scripts

- **`setup.bat`** - Automated Windows installation script
- **`configure_hardware.py`** - Hardware detection and configuration
- **`system_monitor.py`** - Real-time performance monitoring
- **`benchmark.py`** - CPU and GPU performance testing
- **`requirements.txt`** - Python package dependencies

#### 📋 Configuration Files

- **`.gitignore`** - Excludes virtual environments and build artifacts
- **`KinoNichVolling.xml`** - Project-specific XML configuration
- **`publish.txt`** - Build and deployment information

### Quick Start Instructions

#### Option 1: Automated Setup (Recommended)

```cmd
# Run the automated setup script
setup.bat
```

This will:
1. Check Python installation
2. Create virtual environment
3. Install all dependencies
4. Install PyTorch with CUDA support
5. Run hardware configuration check

#### Option 2: Manual Setup

```cmd
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python configure_hardware.py
```

### What Each Tool Does

#### configure_hardware.py
**Purpose**: Verify your system configuration
**When to use**: After installation or when troubleshooting
**What it checks**:
- Python version compatibility
- CPU cores and threads (should detect 12 threads)
- GPU availability and CUDA support
- Memory configuration
- Required Python packages

**Example output**:
```
✅ Python version is compatible
✅ Detected 12 threads (matches Ryzen 5 5600X)
✅ CUDA is available in PyTorch
✅ GPU Device: NVIDIA GeForce RTX 3050
```

#### system_monitor.py
**Purpose**: Monitor system performance in real-time
**When to use**: During development or when analyzing performance
**What it shows**:
- CPU usage per thread (all 12 threads of Ryzen 5 5600X)
- GPU load, memory, and temperature
- RAM usage
- Disk usage
- Network statistics

**How to use**:
```cmd
python system_monitor.py
# Press Ctrl+C to exit
```

#### benchmark.py
**Purpose**: Test CPU and GPU performance
**When to use**: After setup or when optimizing code
**What it tests**:
- Single-threaded CPU performance
- Multi-threaded CPU performance (utilizing all 12 threads)
- NumPy operations
- GPU matrix multiplication
- GPU tensor operations
- Memory allocation speed

**How to use**:
```cmd
python benchmark.py
```

**Expected performance** (approximate):
- CPU matrix multiplication: 50-100 GFLOPS
- GPU matrix multiplication: 2-4 TFLOPS (RTX 3050)
- Multi-threaded processing: Utilizes all 12 threads

### Key Features

#### Hardware Optimization

**For AMD Ryzen 5 5600X (CPU)**:
- Multi-threading support for all 12 threads
- Environment variables automatically set (OMP_NUM_THREADS, etc.)
- CPU affinity configuration
- High-performance power plan recommendations

**For NVIDIA RTX 3050 (GPU)**:
- CUDA Toolkit integration
- PyTorch GPU acceleration
- Memory management utilities
- Performance monitoring
- Temperature tracking

#### Python Environment

**Core Dependencies**:
- NumPy, Pandas, SciPy - Data processing
- PyTorch - GPU-accelerated computing
- lxml, xmltodict - XML processing
- SQLAlchemy - Database operations
- PSUtil, GPUtil - System monitoring

**UI/UX Libraries**:
- PyQt6, PySide6 - Desktop GUI frameworks
- customtkinter - Modern Tkinter widgets

**Build Tools**:
- invoke, pybuilder - Build automation
- pytest - Testing framework
- black, flake8 - Code formatting and linting

### Performance Tips

#### CPU Optimization (Ryzen 5 5600X)

1. **Use All Threads**:
```python
import multiprocessing as mp
num_workers = mp.cpu_count()  # Returns 12
```

2. **Set Environment Variables**:
```python
import os
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
```

3. **Enable High Performance Power Plan**:
```cmd
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
```

#### GPU Optimization (RTX 3050)

1. **Use GPU for Computation**:
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)
```

2. **Batch Operations**:
```python
batch_size = 128  # Adjust based on GPU memory
```

3. **Clear GPU Cache**:
```python
torch.cuda.empty_cache()
```

4. **Monitor GPU Usage**:
```python
import GPUtil
gpus = GPUtil.getGPUs()
print(f"GPU Load: {gpus[0].load*100}%")
```

### Common Use Cases

#### Development Workflow

1. Activate environment: `venv\Scripts\activate`
2. Monitor system: `python system_monitor.py` (in separate terminal)
3. Develop your application
4. Test performance: `python benchmark.py`
5. Deactivate: `deactivate`

#### Performance Analysis

1. Run baseline benchmark: `python benchmark.py > baseline.txt`
2. Make optimization changes
3. Run new benchmark: `python benchmark.py > optimized.txt`
4. Compare results

#### System Health Check

```cmd
python configure_hardware.py
```

Review output for any warnings or issues.

### Troubleshooting Guide

#### GPU Not Detected

**Symptoms**: CUDA not available, GPU not shown in system_monitor.py

**Solutions**:
1. Update NVIDIA drivers from nvidia.com
2. Install CUDA Toolkit
3. Reinstall PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
4. Restart computer

#### Python Not in PATH

**Symptoms**: "python is not recognized" error

**Solutions**:
1. Reinstall Python with "Add to PATH" checked
2. Manually add to PATH:
   - `C:\Users\YourUser\AppData\Local\Programs\Python\Python311`
   - `C:\Users\YourUser\AppData\Local\Programs\Python\Python311\Scripts`

#### Poor Performance

**Symptoms**: Slow execution, low CPU/GPU usage

**Solutions**:
1. Enable High Performance power plan
2. Check CPU parking (see PYTHON_SETUP_WINDOWS10.md)
3. Update GPU drivers
4. Close background applications
5. Verify environment variables are set

### Project Structure

```
codec5jchain/
├── README.md                      # Project overview
├── QUICKSTART.md                  # Fast setup guide
├── PYTHON_SETUP_WINDOWS10.md      # Complete installation guide
├── COMMANDS.md                    # Command reference
├── PROJECT_OVERVIEW.md            # This file
├── setup.bat                      # Automated installation script
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── configure_hardware.py          # Hardware configuration tool
├── system_monitor.py              # Performance monitoring tool
├── benchmark.py                   # Benchmark testing tool
├── KinoNichVolling.xml           # Project XML configuration
└── publish.txt                    # Build information
```

### Next Steps

1. ✅ Complete installation using `setup.bat` or manual steps
2. ✅ Run `python configure_hardware.py` to verify setup
3. ✅ Test with `python benchmark.py`
4. ✅ Monitor with `python system_monitor.py`
5. 📖 Read full documentation in `PYTHON_SETUP_WINDOWS10.md`
6. 🚀 Start developing!

### Support & Resources

- **Project Repository**: https://github.com/kadSerrCX/codec5jchain
- **Python Documentation**: https://docs.python.org/3/
- **PyTorch Documentation**: https://pytorch.org/docs/
- **NVIDIA CUDA**: https://developer.nvidia.com/cuda-toolkit
- **AMD Ryzen**: https://www.amd.com/en/products/processors/desktops/ryzen

### Version Information

- **Project Version**: v1.8.17
- **Python Required**: 3.11+
- **CUDA Toolkit**: 12.x
- **PyTorch**: 2.0+
- **Target OS**: Windows 10 (64-bit)

---

**Last Updated**: December 2025  
**Optimized For**: Windows 10 + NVIDIA RTX 3050 + AMD Ryzen 5 5600X
