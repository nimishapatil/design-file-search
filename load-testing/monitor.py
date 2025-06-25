#!/usr/bin/env python3
import psutil
import time

def simple_monitor():
    print("🔍 Monitoring system resources...")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print("Time     | CPU   | Memory | Available RAM | Disk  ")
    print("=" * 60)
    
    try:
        while True:
            # Get current time
            current_time = time.strftime("%H:%M:%S")
            
            # Get system stats
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Format available RAM in GB
            available_gb = memory.available / (1024**3)
            
            # Print formatted line
            print(f"{current_time} | {cpu:5.1f}% | {memory.percent:6.1f}% | {available_gb:11.1f}GB | {disk.percent:5.1f}%")
            
            # Warning thresholds
            if cpu > 90:
                print("  ⚠️  WARNING: High CPU usage!")
            if memory.percent > 85:
                print("  ⚠️  WARNING: High memory usage!")
            if available_gb < 1.0:
                print("  ⚠️  WARNING: Low available RAM!")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("📊 Monitoring stopped")
        
        # Print final summary
        final_cpu = psutil.cpu_percent()
        final_memory = psutil.virtual_memory()
        
        print(f"\nFinal stats:")
        print(f"CPU: {final_cpu:.1f}%")
        print(f"Memory: {final_memory.percent:.1f}%")
        print(f"Available RAM: {final_memory.available/(1024**3):.1f}GB")

if __name__ == "__main__":
    simple_monitor()
