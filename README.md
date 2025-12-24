# codec5jchain - Build Automation & UX Development

A comprehensive project for build automation, workflow management, and UX/UI development with cross-platform compatibility.

## Quick Start

🚀 **[Get Started in 5 Minutes](QUICKSTART.md)** - Fast setup guide for Windows 10

## Documentation

### For Windows 10 Users (NVIDIA RTX 3050 + AMD Ryzen 5 5600X)

📖 **[Complete Python Setup Guide](PYTHON_SETUP_WINDOWS10.md)** - Detailed installation and configuration instructions

This guide includes:
- Python installation and environment setup
- NVIDIA CUDA Toolkit configuration for RTX 3050
- AMD Ryzen 5 5600X optimization
- Hardware-specific performance tuning
- Troubleshooting and best practices

### Tools & Scripts

- **`system_monitor.py`** - Real-time system performance monitoring
- **`benchmark.py`** - CPU and GPU performance benchmarking
- **`requirements.txt`** - Python dependencies

## Project Objectives

1. **Build Automation**: Automated build processes and workflow management
2. **UX/UI Development**: Efficient user interface and experience development
3. **Cross-Platform Support**: Compatible with x86, x64, and ARM architectures
4. **Integration**: Works with ASP.NET, C/C++, SQLite, and modern frameworks

## System Requirements

### Minimum Requirements
- **OS**: Windows 10 64-bit, macOS, or Linux
- **Python**: 3.11 or higher
- **RAM**: 8GB minimum, 16GB recommended

### Recommended Hardware (Windows 10)
- **CPU**: AMD Ryzen 5 5600X or equivalent (6-core, 12-thread)
- **GPU**: NVIDIA RTX 3050 or better (for GPU-accelerated tasks)
- **RAM**: 16GB or more
- **Storage**: SSD recommended

## Installation

### Windows 10 Quick Install

```cmd
# 1. Clone repository
git clone https://github.com/kadSerrCX/codec5jchain.git
cd codec5jchain

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install PyTorch with CUDA (for NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Verify installation
python system_monitor.py
```

### Other Platforms

See [PYTHON_SETUP_WINDOWS10.md](PYTHON_SETUP_WINDOWS10.md) for detailed instructions.

## Features

- ✅ Real-time system monitoring
- ✅ CPU and GPU performance benchmarking
- ✅ Multi-threaded processing support
- ✅ CUDA/GPU acceleration
- ✅ XML configuration processing
- ✅ SQLite database integration
- ✅ Build automation tools
- ✅ UI/UX development frameworks

## Version History

| Component           | Version  |
| ------------------- | -------- |
| Propiedad.Interna   | v1.8.17  |
| Python Required     | 3.11+    |
| CUDA Support        | 12.x     |

## Usage Examples

### Monitor System Performance
```cmd
python system_monitor.py
```

### Run Performance Benchmarks
```cmd
python benchmark.py
```

### Custom Python Scripts
See [PYTHON_SETUP_WINDOWS10.md](PYTHON_SETUP_WINDOWS10.md) for example scripts and customization options.

## Contributing

Contributions are welcome! This project aims to provide:
- Robust build automation
- Efficient UX/UI tools
- Hardware-optimized performance
- Clear documentation

## License

See LICENSE file for details.

## Support

For detailed setup instructions, troubleshooting, and optimization guides, see:
- [Quick Start Guide](QUICKSTART.md)
- [Complete Python Setup](PYTHON_SETUP_WINDOWS10.md)
