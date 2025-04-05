import functools
import os
import torch
import platform
import cpuinfo
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
print = functools.partial(print, flush=True)


def get_cpu_info():
    try:
        info = cpuinfo.get_cpu_info()
        return {
            'brand': info['brand_raw'],
            'cores': info['count'],
            'frequency': info.get('hz_actual_friendly', 'Unknown')
        }
    except Exception as e:
        return {
            'brand': platform.processor() or "Unknown",
            'cores': os.cpu_count(),
            'frequency': 'Unknown'
        }


print("\n")

# CPU Information
cpu_info = get_cpu_info()
print("CPU Specifications:")
print(f"Processor: {cpu_info['brand']}")
print(f"Number of Cores: {cpu_info['cores']}")
print(f"CPU Frequency: {cpu_info['frequency']}")
print()

# GPU Information
print("GPU Specifications:")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU Model 1: {torch.cuda.get_device_name(0)}")
    print(f"GPU Model 2: {torch.cuda.get_device_name(1)}")
    print(f"GPU Memory 1: {torch.cuda.get_device_properties(
        0).total_memory / 1024**3:.2f} GB")
    print(f"GPU Memory 2: {torch.cuda.get_device_properties(
        1).total_memory / 1024**3:.2f} GB")
print("-" * 30 + "\n")
