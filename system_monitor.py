#!/usr/bin/env python3
"""
System Monitor for codec5jchain
Optimized for Windows 10 with NVIDIA RTX 3050 and AMD Ryzen 5 5600X

This script monitors and displays real-time system performance metrics.
"""

import os
import sys
import time
import platform
from datetime import datetime

try:
    import psutil
except ImportError:
    print("Error: psutil not installed. Run: pip install psutil")
    sys.exit(1)

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: GPUtil not installed. GPU monitoring disabled.")
    print("To enable GPU monitoring, run: pip install gputil nvidia-ml-py")


def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_cpu_info():
    """Get detailed CPU information for AMD Ryzen 5 5600X"""
    info = {
        'cores': psutil.cpu_count(logical=False),
        'threads': psutil.cpu_count(logical=True),
        'usage_total': psutil.cpu_percent(interval=1),
        'usage_per_thread': psutil.cpu_percent(interval=1, percpu=True),
        'frequency': psutil.cpu_freq(),
    }
    return info


def get_memory_info():
    """Get memory information"""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        'total': memory.total / (1024**3),
        'used': memory.used / (1024**3),
        'available': memory.available / (1024**3),
        'percent': memory.percent,
        'swap_total': swap.total / (1024**3),
        'swap_used': swap.used / (1024**3),
        'swap_percent': swap.percent,
    }


def get_gpu_info():
    """Get GPU information for NVIDIA RTX 3050"""
    if not GPU_AVAILABLE:
        return None
    
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = []
        
        for gpu in gpus:
            gpu_info.append({
                'id': gpu.id,
                'name': gpu.name,
                'load': gpu.load * 100,
                'memory_total': gpu.memoryTotal,
                'memory_used': gpu.memoryUsed,
                'memory_free': gpu.memoryFree,
                'memory_percent': gpu.memoryUtil * 100,
                'temperature': gpu.temperature,
            })
        
        return gpu_info
    except Exception as e:
        print(f"GPU monitoring error: {e}")
        return None


def get_disk_info():
    """Get disk information"""
    disks = []
    
    for partition in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            disks.append({
                'device': partition.device,
                'mountpoint': partition.mountpoint,
                'fstype': partition.fstype,
                'total': usage.total / (1024**3),
                'used': usage.used / (1024**3),
                'free': usage.free / (1024**3),
                'percent': usage.percent,
            })
        except PermissionError:
            continue
    
    return disks


def get_network_info():
    """Get network information"""
    net_io = psutil.net_io_counters()
    
    return {
        'bytes_sent': net_io.bytes_sent / (1024**2),  # MB
        'bytes_recv': net_io.bytes_recv / (1024**2),  # MB
        'packets_sent': net_io.packets_sent,
        'packets_recv': net_io.packets_recv,
    }


def format_bar(percent, width=40):
    """Create a progress bar"""
    filled = int(width * percent / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {percent:.1f}%"


def display_system_info():
    """Display all system information"""
    clear_screen()
    
    # Header
    print("=" * 80)
    print(f"{'CODEC5JCHAIN SYSTEM MONITOR':^80}")
    print(f"{'Windows 10 + NVIDIA RTX 3050 + AMD Ryzen 5 5600X':^80}")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print()
    
    # CPU Information
    print("─" * 80)
    print("AMD RYZEN 5 5600X - CPU INFORMATION")
    print("─" * 80)
    
    cpu_info = get_cpu_info()
    print(f"Cores: {cpu_info['cores']} | Threads: {cpu_info['threads']}")
    
    if cpu_info['frequency']:
        print(f"Frequency: {cpu_info['frequency'].current:.2f} MHz "
              f"(Min: {cpu_info['frequency'].min:.2f} MHz, Max: {cpu_info['frequency'].max:.2f} MHz)")
    
    print(f"\nTotal CPU Usage: {format_bar(cpu_info['usage_total'])}")
    
    print("\nPer-Thread Usage:")
    for i, usage in enumerate(cpu_info['usage_per_thread']):
        core = i // 2
        thread = i % 2
        print(f"  Core {core} Thread {thread}: {format_bar(usage, 30)}")
    
    # Memory Information
    print("\n" + "─" * 80)
    print("MEMORY INFORMATION")
    print("─" * 80)
    
    mem_info = get_memory_info()
    print(f"RAM Usage: {format_bar(mem_info['percent'])}")
    print(f"  Total: {mem_info['total']:.2f} GB")
    print(f"  Used: {mem_info['used']:.2f} GB")
    print(f"  Available: {mem_info['available']:.2f} GB")
    
    if mem_info['swap_total'] > 0:
        print(f"\nSwap Usage: {format_bar(mem_info['swap_percent'])}")
        print(f"  Total: {mem_info['swap_total']:.2f} GB")
        print(f"  Used: {mem_info['swap_used']:.2f} GB")
    
    # GPU Information
    print("\n" + "─" * 80)
    print("NVIDIA RTX 3050 - GPU INFORMATION")
    print("─" * 80)
    
    gpu_info = get_gpu_info()
    if gpu_info:
        for gpu in gpu_info:
            print(f"GPU {gpu['id']}: {gpu['name']}")
            print(f"  Load: {format_bar(gpu['load'])}")
            print(f"  Memory: {format_bar(gpu['memory_percent'])}")
            print(f"    Used: {gpu['memory_used']:.0f} MB / {gpu['memory_total']:.0f} MB")
            print(f"    Free: {gpu['memory_free']:.0f} MB")
            print(f"  Temperature: {gpu['temperature']}°C")
    else:
        print("GPU information unavailable")
        print("  - Ensure NVIDIA drivers are installed")
        print("  - Install GPU monitoring: pip install gputil nvidia-ml-py")
    
    # Disk Information
    print("\n" + "─" * 80)
    print("DISK INFORMATION")
    print("─" * 80)
    
    disks = get_disk_info()
    for disk in disks:
        print(f"\nDrive {disk['device']} ({disk['mountpoint']}) - {disk['fstype']}")
        print(f"  Usage: {format_bar(disk['percent'])}")
        print(f"  Total: {disk['total']:.2f} GB")
        print(f"  Used: {disk['used']:.2f} GB")
        print(f"  Free: {disk['free']:.2f} GB")
    
    # Network Information
    print("\n" + "─" * 80)
    print("NETWORK INFORMATION")
    print("─" * 80)
    
    net_info = get_network_info()
    print(f"Data Sent: {net_info['bytes_sent']:.2f} MB ({net_info['packets_sent']} packets)")
    print(f"Data Received: {net_info['bytes_recv']:.2f} MB ({net_info['packets_recv']} packets)")
    
    # Footer
    print("\n" + "=" * 80)
    print("Press Ctrl+C to exit | Updating every 5 seconds")
    print("=" * 80)


def main():
    """Main function"""
    print("Starting System Monitor...")
    print("Optimized for Windows 10 with NVIDIA RTX 3050 and AMD Ryzen 5 5600X")
    time.sleep(2)
    
    try:
        while True:
            display_system_info()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nSystem Monitor stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
