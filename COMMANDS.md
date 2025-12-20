# Command Reference Card - codec5jchain

## Quick Reference for Windows 10 + NVIDIA RTX 3050 + AMD Ryzen 5 5600X

### Initial Setup

```cmd
# Install Python 3.11+ from python.org
# Then run these commands:

cd C:\path\to\codec5jchain
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Daily Use

```cmd
# Activate virtual environment
venv\Scripts\activate

# Deactivate virtual environment
deactivate
```

### Monitoring & Testing

```cmd
# Check hardware configuration
python configure_hardware.py

# Monitor system performance (real-time)
python system_monitor.py

# Run performance benchmarks
python benchmark.py
```

### Package Management

```cmd
# Install new package
pip install package-name

# Update all packages
pip install --upgrade -r requirements.txt

# List installed packages
pip list

# Check for outdated packages
pip list --outdated
```

### CUDA/GPU Commands

```cmd
# Check CUDA installation
nvcc --version

# Check GPU status
nvidia-smi

# Check in Python
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### System Optimization

```cmd
# Enable High Performance power plan
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# Check current power plan
powercfg /list

# Check CPU info
wmic cpu get name,NumberOfCores,NumberOfLogicalProcessors
```

### Python Environment Variables

```cmd
# Set multi-threading for Ryzen 5 5600X (12 threads)
set OMP_NUM_THREADS=12
set MKL_NUM_THREADS=12
set NUMEXPR_NUM_THREADS=12
```

Or add to your Python script:
```python
import os
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
```

### Common Python Imports

```python
# System monitoring
import psutil
import GPUtil

# GPU acceleration
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Multi-processing
import multiprocessing as mp
num_workers = mp.cpu_count()  # Returns 12 for Ryzen 5 5600X

# Data processing
import numpy as np
import pandas as pd

# XML processing
import lxml
import xmltodict
```

### Troubleshooting Commands

```cmd
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clear pip cache
pip cache purge

# Reinstall all packages
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Check Python installation
where python
python --version
pip --version
```

### Performance Monitoring (Python)

```python
# Monitor CPU usage
import psutil
cpu_percent = psutil.cpu_percent(interval=1)
per_cpu = psutil.cpu_percent(interval=1, percpu=True)

# Monitor GPU
import GPUtil
gpus = GPUtil.getGPUs()
for gpu in gpus:
    print(f"GPU Load: {gpu.load*100}%")
    print(f"GPU Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB")

# Monitor Memory
mem = psutil.virtual_memory()
print(f"RAM Usage: {mem.percent}%")
```

### Git Commands (for contributing)

```cmd
# Check status
git status

# Stage changes
git add .

# Commit changes
git commit -m "Your message"

# Push changes
git push origin main
```

### File Locations

- **Documentation**: `README.md`, `PYTHON_SETUP_WINDOWS10.md`, `QUICKSTART.md`
- **Scripts**: `system_monitor.py`, `benchmark.py`, `configure_hardware.py`
- **Dependencies**: `requirements.txt`
- **Config**: `KinoNichVolling.xml`, `publish.txt`

### Keyboard Shortcuts (Windows)

- `Ctrl + C` - Stop running Python script
- `Ctrl + Z + Enter` - Exit Python REPL
- `Alt + Tab` - Switch between windows
- `Win + R` - Run dialog (type `cmd` for command prompt)
- `Win + X` - Quick access menu (select "Command Prompt (Admin)")

### Quick Checks

```cmd
# Is Python installed?
python --version

# Is pip working?
pip --version

# Are NVIDIA drivers installed?
nvidia-smi

# Is CUDA available?
nvcc --version

# Is virtual environment active?
# Look for (venv) at start of command prompt
```

### Getting Help

```cmd
# Python help
python --help

# Pip help
pip --help

# Module help in Python
python -c "import module_name; help(module_name)"

# Script help
python script_name.py --help
```

### Useful Links

- Python Documentation: https://docs.python.org/3/
- PyTorch Documentation: https://pytorch.org/docs/
- NVIDIA CUDA: https://developer.nvidia.com/cuda-toolkit
- NumPy Documentation: https://numpy.org/doc/

---

**Save this file for quick reference!**

For detailed instructions, see:
- [QUICKSTART.md](QUICKSTART.md) - Fast setup guide
- [PYTHON_SETUP_WINDOWS10.md](PYTHON_SETUP_WINDOWS10.md) - Complete guide
