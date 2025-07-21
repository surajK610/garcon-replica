#!/bin/bash
# Quick script to run distributed training tests

echo "🚀 Distributed Training Test Suite"
echo "=================================="

# Make scripts executable
chmod +x server_launcher.py
chmod +x server_instance.py
chmod +x test_servers.py

echo "📋 Available test options:"
echo "1. Run test suite (recommended)"
echo "2. Launch 4 servers manually"
echo "3. Launch 2 servers manually"
echo "4. Run single server test"
echo "5. Run your original distributed_training.py"

read -p "Choose option (1-5): " choice

case $choice in
    1)
        echo "🧪 Running test suite..."
        python3 test_servers.py
        ;;
    2)
        echo "🚀 Launching 4 servers..."
        python3 server_launcher.py
        ;;
    3)
        echo "🚀 Launching 2 servers..."
        python3 server_instance.py --rank 0 --world-size 2 &
        python3 server_instance.py --rank 1 --world-size 2 &
        wait
        ;;
    4)
        echo "🧪 Running single server test..."
        python3 server_instance.py --rank 0 --world-size 1 --epochs 3
        ;;
    5)
        echo "🔥 Running original distributed training..."
        python3 distributed_training.py
        ;;
    *)
        echo "❌ Invalid choice. Please run again with 1-5."
        exit 1
        ;;
esac

echo "✅ Test completed!"