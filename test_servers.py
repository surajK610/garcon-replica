#!/usr/bin/env python3
"""
Test script for distributed servers.
Runs various tests to verify distributed communication is working.
"""

import subprocess
import sys
import time
import os
import threading
from typing import List, Dict

def run_command(cmd: List[str], timeout: int = 30) -> Dict[str, str]:
    """Run a command and return stdout/stderr."""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'stdout': '',
            'stderr': 'Command timed out',
            'returncode': -1
        }

def test_single_server():
    """Test if a single server can start and run."""
    print("🧪 Testing single server startup...")
    
    cmd = [sys.executable, "server_instance.py", "--rank", "0", "--world-size", "1", "--epochs", "2"]
    result = run_command(cmd, timeout=20)
    
    if result['returncode'] == 0:
        print("✅ Single server test passed!")
        return True
    else:
        print(f"❌ Single server test failed!")
        print(f"Error: {result['stderr']}")
        return False

def test_distributed_servers():
    """Test multiple servers running together."""
    print("🧪 Testing distributed servers...")
    
    # Test with 2 servers first
    world_size = 2
    processes = []
    
    try:
        # Start servers
        for rank in range(world_size):
            cmd = [
                sys.executable, "server_instance.py",
                "--rank", str(rank),
                "--world-size", str(world_size),
                "--epochs", "3"
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            processes.append(proc)
            time.sleep(0.5)
        
        # Wait for completion
        success = True
        for i, proc in enumerate(processes):
            stdout, stderr = proc.communicate(timeout=30)
            if proc.returncode != 0:
                print(f"❌ Server {i} failed with return code {proc.returncode}")
                print(f"Error: {stderr}")
                success = False
            else:
                print(f"✅ Server {i} completed successfully")
        
        if success:
            print("✅ Distributed servers test passed!")
            return True
        else:
            print("❌ Distributed servers test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during distributed test: {e}")
        # Clean up
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        return False

def test_launcher():
    """Test the server launcher script."""
    print("🧪 Testing server launcher...")
    
    # Create a simple test that just starts and stops quickly
    # We'll modify the launcher to accept a test mode
    cmd = [sys.executable, "server_launcher.py"]
    
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Let it run for a few seconds then terminate
        time.sleep(5)
        proc.terminate()
        
        stdout, stderr = proc.communicate(timeout=10)
        
        if "servers started" in stdout.lower() or "launching" in stdout.lower():
            print("✅ Server launcher test passed!")
            return True
        else:
            print("❌ Server launcher test failed!")
            print(f"Output: {stdout}")
            print(f"Error: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during launcher test: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("🧪 Running distributed training server tests")
    print("=" * 50)
    
    tests = [
        ("Single Server", test_single_server),
        ("Distributed Servers", test_distributed_servers),
        ("Server Launcher", test_launcher)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        success = test_func()
        results.append((test_name, success))
        print()
    
    print("📊 Test Results:")
    print("=" * 30)
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed! Your distributed training setup is working!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return all_passed

if __name__ == "__main__":
    run_all_tests()