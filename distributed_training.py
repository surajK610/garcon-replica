#!/usr/bin/env python3
"""
Mac-Optimized Distributed Training Example
Supports both CPU and MPS (Metal Performance Shaders) on Apple Silicon

Features:
- Automatic device detection (MPS > CPU)
- Mac-friendly multiprocessing
- Optimized for Apple Silicon
- No CUDA dependencies

Usage:
    python mac_distributed_training.py
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
import time
import platform
from typing import Optional


def get_best_device() -> str:
    """Get the best available device on Mac."""
    if torch.backends.mps.is_available():
        print("🚀 MPS (Metal Performance Shaders) available - using GPU acceleration!")
        return "mps"
    else:
        print("💻 Using CPU (MPS not available)")
        return "cpu"


def setup_mac_multiprocessing():
    """Setup multiprocessing for Mac compatibility."""
    # Mac-specific multiprocessing setup
    if platform.system() == "Darwin":  # macOS
        mp.set_start_method('spawn', force=True)
        # Avoid potential issues with Mac's multiprocessing
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')


def setup_process_group(rank: int, world_size: int, port: int = 12359):
    """Initialize process group with Mac-friendly settings."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # Use localhost IP
    os.environ['MASTER_PORT'] = str(port)
    
    # Use gloo backend (works well on Mac)
    dist.init_process_group(
        backend='gloo',
        rank=rank,
        world_size=world_size,
        timeout=torch.distributed.default_pg_timeout
    )
    
    print(f"🔧 Process {rank}/{world_size} initialized on {platform.system()}")


class MacOptimizedMLP(nn.Module):
    """MLP optimized for Mac devices."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device: str):
        super().__init__()
        self.device = device
        
        # Create layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)
        
        # Initialize weights for better convergence
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.layers(x)


def manual_reduce_scatter(input_tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Manually implement reduce_scatter since gloo doesn't support it."""
    # Step 1: All-gather all tensors
    gather_list = [torch.zeros_like(input_tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, input_tensor)
    
    # Step 2: Each rank reduces its assigned chunk
    chunk_size = len(input_tensor) // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size
    
    result_chunk = torch.zeros(chunk_size)
    for gathered_tensor in gather_list:
        result_chunk += gathered_tensor[start_idx:end_idx]
    
    return result_chunk


def mac_collective_operations_demo(rank: int, world_size: int, device: str):
    """Demonstrate collective operations optimized for Mac."""
    print(f"📱 Rank {rank}: Running collective ops on {device}")
    
    # Create tensors on the specified device
    tensor = torch.tensor([float(rank + i) for i in range(4)], device='cpu')  # gloo requires CPU tensors
    print(f"📊 Rank {rank} initial tensor: {tensor}")
    
    dist.barrier()
    
    # ALL-REDUCE
    if rank == 0:
        print("\n🔄 ALL-REDUCE Operation")
    
    all_reduce_tensor = tensor.clone()
    dist.all_reduce(all_reduce_tensor, op=dist.ReduceOp.SUM)
    print(f"✅ Rank {rank} all_reduce result: {all_reduce_tensor}")
    
    dist.barrier()
    
    # ALL-GATHER  
    if rank == 0:
        print("\n📥 ALL-GATHER Operation")
    
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    print(f"✅ Rank {rank} all_gather result: {[t.tolist() for t in gather_list]}")
    
    dist.barrier()
    
    # BROADCAST
    if rank == 0:
        print("\n📡 BROADCAST Operation")
        broadcast_tensor = torch.tensor([100.0, 200.0, 300.0, 400.0])
    else:
        broadcast_tensor = torch.zeros(4)
    
    print(f"📊 Rank {rank} before broadcast: {broadcast_tensor}")
    dist.broadcast(broadcast_tensor, src=0)
    print(f"✅ Rank {rank} after broadcast: {broadcast_tensor}")
    
    dist.barrier()
    
    # MANUAL REDUCE-SCATTER (since gloo doesn't support it)
    if rank == 0:
        print("\n🔀 MANUAL REDUCE-SCATTER Operation")
        print("   (gloo backend doesn't support reduce_scatter, so implementing manually)")
    
    # Create input for manual reduce_scatter
    scatter_input = torch.tensor([float(rank * 4 + i) for i in range(4)])
    print(f"📊 Rank {rank} reduce_scatter input: {scatter_input}")
    
    result_chunk = manual_reduce_scatter(scatter_input, rank, world_size)
    print(f"✅ Rank {rank} manual reduce_scatter result: {result_chunk}")
    
    dist.barrier()
    
    # Demonstrate that manual reduce_scatter = reduce + scatter
    if rank == 0:
        print("\n🔍 Verifying: reduce_scatter = all_reduce + scatter")
    
    # Method 1: All-reduce then take our chunk
    verify_tensor = scatter_input.clone()
    dist.all_reduce(verify_tensor, op=dist.ReduceOp.SUM)
    our_chunk = verify_tensor[rank:rank+1]  # Each rank takes 1 element
    
    print(f"✅ Rank {rank} verification (all_reduce then scatter): {our_chunk}")
    
    dist.barrier()


def mac_data_parallel_training(rank: int, world_size: int, device: str):
    """Data parallel training optimized for Mac."""
    print(f"🎯 Rank {rank}: Starting DDP training on {device}")
    
    # Training parameters
    batch_size = 32
    input_dim = 64
    hidden_dim = 128
    output_dim = 10
    num_epochs = 3
    learning_rate = 0.001
    
    # Create model
    model = MacOptimizedMLP(input_dim, hidden_dim, output_dim, device)
    
    # Create synthetic dataset (different on each rank for data parallelism)
    torch.manual_seed(42 + rank)  # Different seed per rank
    local_batch_size = batch_size // world_size
    
    # Generate data on CPU first, then move to device if needed
    train_data = torch.randn(local_batch_size, input_dim)
    train_targets = torch.randint(0, output_dim, (local_batch_size,))
    
    if device == "mps":
        train_data = train_data.to(device)
        train_targets = train_targets.to(device)
    
    print(f"📊 Rank {rank} training data shape: {train_data.shape}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        
        # Forward pass
        outputs = model(train_data)
        loss = criterion(outputs, train_targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Synchronize gradients across ranks (KEY DDP STEP!)
        for param in model.parameters():
            if param.grad is not None:
                # Move gradient to CPU fo\
                # r gloo backend
                cpu_grad = param.grad.cpu() if device == "mps" else param.grad
                
                # Use SUM instead of AVG (gloo doesn't support AVG)
                dist.all_reduce(cpu_grad, op=dist.ReduceOp.SUM)
                
                # Manually average by dividing by world_size
                cpu_grad /= world_size
                
                # Move back to device if needed
                if device == "mps":
                    param.grad.copy_(cpu_grad.to(device))
                else:
                    param.grad.copy_(cpu_grad)
        
        # Update parameters
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            predicted = torch.argmax(outputs, dim=1)
            accuracy = (predicted == train_targets).float().mean()
        
        if epoch % 1 == 0:
            print(f"🏃 Rank {rank}, Epoch {epoch}, Loss: {loss.item():.4f}, Acc: {accuracy.item():.4f}")
    
    dist.barrier()
    
    # Verify parameter synchronization
    if rank == 0:
        print("\n🔍 Verifying parameter synchronization...")
    
    # Check first layer weights
    first_param = next(model.parameters())
    cpu_param = first_param.cpu() if device == "mps" else first_param
    
    param_list = [torch.zeros_like(cpu_param) for _ in range(world_size)]
    dist.all_gather(param_list, cpu_param)
    
    if rank == 0:
        # Check if parameters are synchronized
        all_same = all(torch.allclose(param_list[0], p, atol=1e-6) for p in param_list[1:])
        print(f"✅ Parameters synchronized across ranks: {all_same}")


def benchmark_mac_performance(rank: int, world_size: int, device: str):
    """Benchmark communication and computation on Mac."""
    if rank == 0:
        print(f"\n⚡ Benchmarking performance on {device}")
    
    # Test different tensor sizes
    sizes = [1000, 10000, 100000]
    
    for size in sizes:
        # Create tensor on CPU for communication
        tensor = torch.randn(size)
        
        # Benchmark communication (all_reduce)
        dist.barrier()
        start_time = time.time()
        
        for _ in range(5):  # Multiple iterations
            test_tensor = tensor.clone()
            dist.all_reduce(test_tensor)
        
        dist.barrier()
        comm_time = (time.time() - start_time) / 5
        
        # Benchmark computation (matrix multiplication)
        if device == "mps":
            comp_tensor = torch.randn(size // 10, size // 10).to(device)
        else:
            comp_tensor = torch.randn(size // 10, size // 10)
        
        start_time = time.time()
        for _ in range(5):
            result = torch.matmul(comp_tensor, comp_tensor.T)
        
        if device == "mps":
            torch.mps.synchronize()  # Wait for MPS operations
        
        comp_time = (time.time() - start_time) / 5
        
        if rank == 0:
            print(f"📊 Size {size:>6}: Comm {comm_time:.4f}s, Comp {comp_time:.4f}s")


def show_backend_capabilities(rank: int):
    """Show what operations are supported by different backends."""
    if rank == 0:
        print("\n📋 Backend Capabilities Summary:")
        print("┌─────────────────┬─────────┬─────────┬─────────┐")
        print("│ Operation       │  NCCL   │  Gloo   │   MPI   │")
        print("├─────────────────┼─────────┼─────────┼─────────┤")
        print("│ all_reduce      │    ✅    │    ✅    │    ✅    │")
        print("│ all_gather      │    ✅    │    ✅    │    ✅    │")
        print("│ broadcast       │    ✅    │    ✅    │    ✅    │")
        print("│ reduce_scatter  │    ✅    │    ❌    │    ✅    │")
        print("│ ReduceOp.AVG    │    ✅    │    ❌    │    ✅    │")
        print("│ ReduceOp.SUM    │    ✅    │    ✅    │    ✅    │")
        print("│ GPU Support     │    ✅    │    ❌    │    ✅    │")
        print("│ CPU Support     │    ❌    │    ✅    │    ✅    │")
        print("│ Mac MPS         │    ❌    │   ✅*   │    ❌    │")
        print("└─────────────────┴─────────┴─────────┴─────────┘")
        print("* Gloo works with MPS by moving tensors to CPU for communication")
        print("\n💡 Workarounds we're using:")
        print("  • reduce_scatter: manual implementation with all_gather")
        print("  • ReduceOp.AVG: use SUM then divide by world_size")
        print("  • MPS tensors: move to CPU for communication, back to MPS for compute")


def demonstrate_workarounds(rank: int, world_size: int):
    """Demonstrate the workarounds we use for gloo limitations."""
    if rank == 0:
        print("\n🔧 WORKAROUND DEMONSTRATIONS")
        print("="*50)
    
    dist.barrier()
    
    # 1. Manual averaging instead of ReduceOp.AVG
    if rank == 0:
        print("\n1️⃣ Manual Averaging (since gloo doesn't support ReduceOp.AVG)")
    
    test_tensor = torch.tensor([float(rank + 1)])
    print(f"📊 Rank {rank} value: {test_tensor.item()}")
    
    # Method 1: SUM then manual divide
    sum_tensor = test_tensor.clone()
    dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
    avg_tensor = sum_tensor / world_size
    print(f"✅ Rank {rank} manual average: {avg_tensor.item()}")
    
    dist.barrier()
    
    # 2. Manual reduce_scatter using primitives that work
    if rank == 0:
        print("\n2️⃣ Manual Reduce-Scatter (since gloo doesn't support it)")
    
    # Each rank has a 4-element tensor
    input_data = torch.tensor([float(rank * 4 + i) for i in range(4)])
    print(f"📊 Rank {rank} input: {input_data}")
    
    # Step 1: All-gather everyone's data
    all_data = [torch.zeros_like(input_data) for _ in range(world_size)]
    dist.all_gather(all_data, input_data)
    
    # Step 2: Each rank reduces its assigned element
    my_element_idx = rank
    my_result = sum(data[my_element_idx] for data in all_data)
    print(f"✅ Rank {rank} reduce_scatter result (element {my_element_idx}): {my_result.item()}")
    
    dist.barrier()
    
    if rank == 0:
        print("\n🎓 These workarounds teach us:")
        print("  • How complex operations are built from simple ones")
        print("  • Why backend choice matters for performance")
        print("  • How to adapt when hardware/software has limitations")


def worker_process(rank: int, world_size: int):
    """Main worker function for each process."""
    try:
        # Setup
        setup_process_group(rank, world_size)
        device = get_best_device()
        
        if rank == 0:
            print(f"\n🍎 Running on macOS with {world_size} processes")
            print(f"🔧 Device: {device}")
            print("=" * 50)
        
        # Show backend info
        show_backend_capabilities(rank)
        
        # Run demos
        mac_collective_operations_demo(rank, world_size, device)
        demonstrate_workarounds(rank, world_size)
        mac_data_parallel_training(rank, world_size, device)
        benchmark_mac_performance(rank, world_size, device)
        
        if rank == 0:
            print("\n🎉 All Mac demos completed successfully!")
            print("\n📚 Summary of what you learned:")
            print("  ✅ Collective operations (all_reduce, all_gather, broadcast)")
            print("  ✅ Data Parallel training with gradient synchronization")
            print("  ✅ Backend limitations and workarounds")
            print("  ✅ MPS acceleration on Apple Silicon")
            print("  ✅ Real distributed training concepts from CS336!")
        
    except Exception as e:
        print(f"❌ Error in rank {rank}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    """Main function optimized for Mac."""
    # Mac-specific setup
    setup_mac_multiprocessing()
    
    world_size = 4  # Simulate 4 processes
    
    print("🍎 Mac-Optimized Distributed Training")
    print(f"🖥️  System: {platform.system()} {platform.machine()}")
    print(f"🐍 Python: {platform.python_version()}")
    print(f"🔥 PyTorch: {torch.__version__}")
    
    if torch.backends.mps.is_available():
        print("⚡ MPS Support: Available")
    else:
        print("💻 MPS Support: Not available (using CPU)")
    
    print("-" * 60)
    
    try:
        # Spawn processes
        mp.spawn(
            worker_process,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
        
        print("\n✨ Mac distributed training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
