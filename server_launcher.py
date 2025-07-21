#!/usr/bin/env python3
"""
Server launcher for distributed training testing.
Starts multiple server processes that can communicate via PyTorch distributed.
"""

import subprocess
import sys
import time
import os
import signal
from typing import List

def start_server(rank: int, world_size: int, port: int = 12359) -> subprocess.Popen:
    """Start a single server process."""
    cmd = [
        sys.executable, "server_instance.py",
        "--rank", str(rank),
        "--world-size", str(world_size),
        "--port", str(port)
    ]
    
    print(f"Starting server rank {rank} on port {port}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def main():
    """Launch multiple server instances for distributed training."""
    world_size = 4
    port = 12359
    processes: List[subprocess.Popen] = []
    
    print(f"🚀 Launching {world_size} server instances for distributed training")
    print("=" * 60)
    
    try:
        # Start all server processes
        for rank in range(world_size):
            proc = start_server(rank, world_size, port)
            processes.append(proc)
            time.sleep(0.5)  # Small delay between launches
        
        print(f"✅ All {world_size} servers started!")
        print("Press Ctrl+C to stop all servers")
        
        # Wait for all processes to complete
        for i, proc in enumerate(processes):
            stdout, stderr = proc.communicate()
            if stdout:
                print(f"=== Server {i} Output ===")
                print(stdout)
            if stderr:
                print(f"=== Server {i} Errors ===")
                print(stderr)
        
        print("🎉 All servers completed!")
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping all servers...")
        for proc in processes:
            proc.terminate()
        
        # Wait for clean shutdown
        for proc in processes:
            proc.wait()
        
        print("✅ All servers stopped")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        # Clean up any running processes
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()

if __name__ == "__main__":
    main()