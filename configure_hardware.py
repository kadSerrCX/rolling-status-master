#!/usr/bin/env python3
"""
Hardware Configuration and Optimization Script
For Windows 10 with NVIDIA RTX 3050 and AMD Ryzen 5 5600X

This script helps configure Python environment for optimal performance.
"""

import os
import sys
import platform
import subprocess


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"{text:^80}")
    print("=" * 80)


def print_section(text):
    """Print formatted section"""
    print("\n" + "-" * 80)
    print(text)
    print("-" * 80)


def check_python_version():
    """Check Python version"""
    print_section("Python Version Check")
    
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("⚠️  WARNING: Python 3.11+ recommended for best performance")
        print("   Current version may work but upgrade is suggested")
        return False
    else:
        print("✅ Python version is compatible")
        return True


def check_platform():
    """Check platform information"""
    print_section("Platform Information")
    
    print(f"System: {platform.system()}")
    print(f"Release: {platform.release()}")
    print(f"Version: {platform.version()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    if platform.system() != "Windows":
        print("\n⚠️  WARNING: This script is optimized for Windows 10")
        print("   Some features may not work on other platforms")
        return False
    else:
        print("\n✅ Running on Windows")
        return True


def check_cpu():
    """Check CPU information"""
    print_section("CPU Configuration - AMD Ryzen 5 5600X")
    
    try:
        import psutil
        
        cores = psutil.cpu_count(logical=False)
        threads = psutil.cpu_count(logical=True)
        
        print(f"Physical Cores: {cores}")
        print(f"Logical Threads: {threads}")
        
        if threads == 12:
            print("✅ Detected 12 threads (matches Ryzen 5 5600X)")
        else:
            print(f"⚠️  Expected 12 threads for Ryzen 5 5600X, detected {threads}")
        
        freq = psutil.cpu_freq()
        if freq:
            print(f"\nCPU Frequency:")
            print(f"  Current: {freq.current:.2f} MHz")
            print(f"  Min: {freq.min:.2f} MHz")
            print(f"  Max: {freq.max:.2f} MHz")
        
        # Set environment variables for multi-threading
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
        
        print("\n✅ Multi-threading environment variables set:")
        print(f"   OMP_NUM_THREADS={threads}")
        print(f"   MKL_NUM_THREADS={threads}")
        print(f"   NUMEXPR_NUM_THREADS={threads}")
        
        return True
        
    except ImportError:
        print("❌ psutil not installed")
        print("   Install with: pip install psutil")
        return False


def check_gpu():
    """Check GPU availability and configuration"""
    print_section("GPU Configuration - NVIDIA RTX 3050")
    
    # Check for CUDA
    cuda_available = False
    
    try:
        result = subprocess.run(
            ["nvidia-smi"], 
            capture_output=True, 
            text=True, 
            timeout=10  # Increased timeout for slower systems
        )
        
        if result.returncode == 0:
            print("✅ NVIDIA driver detected")
            print("\nGPU Information:")
            # Parse nvidia-smi output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'RTX' in line or 'NVIDIA' in line:
                    print(f"  {line.strip()}")
            cuda_available = True
        else:
            print("❌ nvidia-smi failed to run")
            
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        print("   NVIDIA drivers may not be installed")
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
    
    # Check for PyTorch CUDA
    try:
        import torch
        
        print(f"\n✅ PyTorch installed (version {torch.__version__})")
        
        if torch.cuda.is_available():
            print("✅ CUDA is available in PyTorch")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
            
            # Get GPU properties
            props = torch.cuda.get_device_properties(0)
            print(f"\nGPU Properties:")
            print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
            print(f"  Multi Processors: {props.multi_processor_count}")
            print(f"  CUDA Capability: {props.major}.{props.minor}")
            
            return True
        else:
            print("⚠️  CUDA not available in PyTorch")
            print("   Reinstall PyTorch with CUDA support:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            
    except ImportError:
        print("⚠️  PyTorch not installed")
        print("   Install with CUDA support:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    return cuda_available


def check_memory():
    """Check system memory"""
    print_section("Memory Configuration")
    
    try:
        import psutil
        
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        
        print(f"Total RAM: {total_gb:.2f} GB")
        print(f"Available RAM: {mem.available / (1024**3):.2f} GB")
        print(f"Used RAM: {mem.used / (1024**3):.2f} GB")
        print(f"Memory Usage: {mem.percent}%")
        
        if total_gb < 16:
            print("\n⚠️  WARNING: Less than 16GB RAM detected")
            print("   16GB+ recommended for optimal performance")
        else:
            print("\n✅ Sufficient RAM available")
        
        return True
        
    except ImportError:
        print("❌ psutil not installed")
        return False


def check_dependencies():
    """Check important Python dependencies"""
    print_section("Python Dependencies Check")
    
    dependencies = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('torch', 'PyTorch'),
        ('psutil', 'PSUtil'),
        ('GPUtil', 'GPUtil'),
        ('lxml', 'lxml'),
        ('sqlalchemy', 'SQLAlchemy'),
    ]
    
    installed = []
    missing = []
    
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"✅ {display_name} installed")
            installed.append(display_name)
        except ImportError:
            print(f"❌ {display_name} not installed")
            missing.append(display_name)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All core dependencies installed")
        return True


def generate_config():
    """Generate configuration recommendations"""
    print_section("Configuration Recommendations")
    
    print("Recommended Windows 10 Optimizations:")
    print()
    print("1. Enable High Performance Power Plan:")
    print("   powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c")
    print()
    print("2. NVIDIA Control Panel Settings:")
    print("   - Power Management Mode: Prefer Maximum Performance")
    print("   - CUDA - GPUs: Use RTX 3050")
    print()
    print("3. Windows Visual Effects:")
    print("   - Settings > System > About > Advanced system settings")
    print("   - Performance > Adjust for best performance")
    print()
    print("4. Disable CPU Parking (Advanced):")
    print("   - Improves multi-threaded performance on Ryzen CPUs")
    print("   - Requires registry edit (see PYTHON_SETUP_WINDOWS10.md)")


def main():
    """Main function"""
    print_header("CODEC5JCHAIN HARDWARE CONFIGURATION")
    print(f"Windows 10 + NVIDIA RTX 3050 + AMD Ryzen 5 5600X")
    
    # Run all checks
    results = []
    
    results.append(("Python Version", check_python_version()))
    results.append(("Platform", check_platform()))
    results.append(("CPU", check_cpu()))
    results.append(("GPU", check_gpu()))
    results.append(("Memory", check_memory()))
    results.append(("Dependencies", check_dependencies()))
    
    # Generate recommendations
    generate_config()
    
    # Summary
    print_header("CONFIGURATION SUMMARY")
    
    print("\nChecks Performed:")
    for name, passed in results:
        status = "✅ PASS" if passed else "⚠️  NEEDS ATTENTION"
        print(f"  {name}: {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\nScore: {passed_count}/{total_count} checks passed")
    
    if passed_count == total_count:
        print("\n🎉 Your system is fully configured and ready!")
        print("   Run 'python benchmark.py' to test performance")
    else:
        print("\n⚠️  Some configuration issues detected")
        print("   Review the output above and follow recommendations")
        print("   See PYTHON_SETUP_WINDOWS10.md for detailed instructions")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nConfiguration check interrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
