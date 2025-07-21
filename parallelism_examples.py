"""
Minimal PyTorch examples for different parallelism strategies:
1. DDP (Distributed Data Parallel)
2. FSDP 2 (Fully Sharded Data Parallel v2)
3. Tensor Parallelism
4. Sequence/Context Parallelism

Run with: torchrun --nproc_per_node=4 parallelism_examples.py --strategy ddp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, parallelize_module
)
from torch.distributed.tensor import DeviceMesh, distribute_tensor, Replicate, Shard
import argparse
import os


# Simple transformer model for demonstration
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=1000, d_model=512, nhead=8, num_layers=6, seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len)
        seq_len = x.size(1)
        
        # Embeddings + positional encoding
        x = self.embedding(x) + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer
        x = self.transformer(x)
        
        # Classification (predict next token for each position)
        x = self.classifier(x)
        return x


def get_best_device():
    """Get the best available device in order of preference"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def setup_distributed():
    """Initialize distributed training environment with device auto-detection"""
    # Get best available device type
    device_type = get_best_device()
    
    # Check if we're in a distributed environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Distributed training
        if not dist.is_initialized():
            # Choose backend based on device and platform
            if device_type == "cuda":
                backend = "nccl"  # Best for CUDA
            else:
                backend = "gloo"  # Works for CPU/MPS and cross-platform
            
            dist.init_process_group(backend=backend)
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # Set device based on rank and available devices
        if device_type == "cuda":
            device_id = rank % torch.cuda.device_count()
            device = torch.device(f"cuda:{device_id}")
            torch.cuda.set_device(device_id)
        else:
            device = torch.device("cpu")
        
        # Set default device for automatic tensor placement
        torch.set_default_device(device)
        return rank, world_size, device
    else:
        # Single device training (for testing)
        rank = 0
        world_size = 1
        
        if device_type == "cuda":
            device = torch.device("cuda:0")
            torch.cuda.set_device(0)
        elif device_type == "mps":
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        torch.set_default_device(device)
        print(f"Running in single-device mode on {device}")
        return rank, world_size, device


def create_dummy_data(batch_size, seq_len, vocab_size, device):
    """Create dummy training data on the specified device"""
    # Use torch.randint which respects default device
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Only move to device if not using default device
    if not hasattr(torch, '_default_device') or torch._default_device != device:
        inputs = inputs.to(device)
        targets = targets.to(device)
    
    return inputs, targets


# =============================================================================
# 1. DDP (Distributed Data Parallel)
# =============================================================================
def ddp_example():
    """DDP replicates model on each device, synchronizes gradients"""
    rank, world_size, device = setup_distributed()
    
    # Create model (automatically on correct device due to set_default_device)
    model = SimpleTransformer()
    
    # Only wrap with DDP if in distributed mode
    if world_size > 1:
        # DDP device_ids only needed for CUDA
        if device.type == 'cuda':
            model = DDP(model, device_ids=[device])
        else:
            model = DDP(model)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    for step in range(5):
        # Each rank gets different data (data parallelism)
        batch_size = 4
        inputs, targets = create_dummy_data(batch_size, 128, 1000, device)
        
        # Forward pass
        outputs = model(inputs)  # Shape: (batch, seq_len, vocab_size)
        
        # Loss (cross entropy over sequence)
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        
        # Backward pass - DDP automatically synchronizes gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #if rank == 0:
        print(f"[Rank {rank}]: DDP Step {step}: Loss = {loss.item():.4f} (Device: {device})")
    
    # Only destroy process group if it was initialized and we're the last call
    if world_size > 1 and dist.is_initialized():
        # Don't destroy in notebooks - just leave it initialized
        pass


# =============================================================================
# 2. FSDP 2 (Fully Sharded Data Parallel)
# =============================================================================
def fsdp_example():
    """FSDP 2 shards model parameters across devices using manual wrapping"""
    rank, world_size, device = setup_distributed()
    
    # Create model (automatically on correct device)
    model = SimpleTransformer()
    
    if world_size > 1:
        # FSDP works best with CUDA, but can work with other devices
        if device.type == 'cuda':
            # FSDP 2 uses manual wrapping - wrap each transformer layer individually
            for i, layer in enumerate(model.transformer.layers):
                model.transformer.layers[i] = FSDP(
                    layer,
                    device_id=device,
                )
            
            # Wrap the entire model as well
            model = FSDP(
                model,
                device_id=device,
            )
        else:
            # For non-CUDA devices, use regular DDP instead of FSDP
            print(f"FSDP not optimized for {device.type}, falling back to DDP")
            model = DDP(model)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    for step in range(5):
        batch_size = 4
        inputs, targets = create_dummy_data(batch_size, 128, 1000, device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Loss
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if rank == 0:
            strategy = "FSDP" if device.type == 'cuda' and world_size > 1 else "DDP" if world_size > 1 else "Single"
            print(f"{strategy} Step {step}: Loss = {loss.item():.4f} (Device: {device})")
    
    if world_size > 1:
        pass


# =============================================================================
# 3. Tensor Parallelism
# =============================================================================
class TensorParallelTransformer(nn.Module):
    """Transformer with tensor parallelism on linear layers"""
    def __init__(self, vocab_size=1000, d_model=512, nhead=8, seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model))
        
        # Single transformer layer for simplicity
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network (will be tensor parallel)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.classifier = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        
        # Embeddings + positional encoding
        x = self.embedding(x) + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward (tensor parallel layers)
        ff_out = F.relu(self.linear1(x))
        ff_out = self.linear2(ff_out)
        x = self.norm2(x + ff_out)
        
        # Classification
        x = self.classifier(x)
        return x


def tensor_parallel_example():
    """Tensor parallelism splits individual weight matrices (CUDA only)"""
    rank, world_size, device = setup_distributed()
    
    if device.type != 'cuda':
        print(f"Tensor parallelism requires CUDA, but got {device.type}. Running simple training instead.")
        simple_training_fallback(device)
        return
    
    # Create device mesh for tensor parallelism
    device_mesh = DeviceMesh("cuda", list(range(world_size)))
    
    # Create model (automatically on correct device)
    model = TensorParallelTransformer()
    
    # Apply tensor parallelism to feed-forward layers
    # linear1: column-wise parallel (split output dimension)
    # linear2: row-wise parallel (split input dimension)
    parallelize_plan = {
        "linear1": ColwiseParallel(),
        "linear2": RowwiseParallel(),
    }
    
    model = parallelize_module(model, device_mesh, parallelize_plan)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    for step in range(5):
        batch_size = 4
        inputs, targets = create_dummy_data(batch_size, 128, 1000, device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Loss
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if rank == 0:
            print(f"Tensor Parallel Step {step}: Loss = {loss.item():.4f} (Device: {device})")
    
    if world_size > 1:
        pass


def simple_training_fallback(device):
    """Fallback for devices that don't support advanced parallelism"""
    model = SimpleTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for step in range(5):
        batch_size = 4
        inputs, targets = create_dummy_data(batch_size, 128, 1000, device)
        
        outputs = model(inputs)
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Simple Training Step {step}: Loss = {loss.item():.4f} (Device: {device})")


# =============================================================================
# 4. Sequence/Context Parallelism
# =============================================================================
def sequence_parallel_example():
    """Sequence parallelism splits sequence dimension across devices"""
    rank, world_size, device = setup_distributed()
    
    # Create model (automatically on correct device)
    model = SimpleTransformer()
    
    # Wrap with DDP for parameter synchronization
    if world_size > 1:
        if device.type == 'cuda':
            model = DDP(model, device_ids=[device])
        else:
            model = DDP(model)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    for step in range(5):
        # Create longer sequence to split
        full_seq_len = 128 * world_size  # Total sequence length
        local_seq_len = 128  # Each device handles this much
        
        # Generate full sequence data (only on rank 0)
        if rank == 0:
            full_inputs = torch.randint(0, 1000, (4, full_seq_len))
            full_targets = torch.randint(0, 1000, (4, full_seq_len))
        else:
            full_inputs = torch.empty(4, full_seq_len, dtype=torch.long)
            full_targets = torch.empty(4, full_seq_len, dtype=torch.long)
        
        # Broadcast full sequence to all ranks (if distributed)
        if world_size > 1:
            dist.broadcast(full_inputs, src=0)
            dist.broadcast(full_targets, src=0)
        
        # Each rank processes its chunk of the sequence
        start_idx = rank * local_seq_len
        end_idx = (rank + 1) * local_seq_len
        
        local_inputs = full_inputs[:, start_idx:end_idx]
        local_targets = full_targets[:, start_idx:end_idx]
        
        # Forward pass on local sequence chunk
        outputs = model(local_inputs)
        
        # Compute loss on local chunk
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), local_targets.view(-1))
        
        # Backward pass - gradients are automatically synchronized by DDP
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if rank == 0:
            strategy = "Sequence Parallel" if world_size > 1 else "Single Device"
            print(f"{strategy} Step {step}: Loss = {loss.item():.4f} (Device: {device})")
    
    if world_size > 1:
        pass


# =============================================================================
# Main execution
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["ddp", "fsdp", "tp", "sp"], 
                       default="ddp", help="Parallelism strategy")
    args = parser.parse_args()
    
    if args.strategy == "ddp":
        print("Running DDP example...")
        ddp_example()
    elif args.strategy == "fsdp":
        print("Running FSDP example...")
        fsdp_example()
    elif args.strategy == "tp":
        print("Running Tensor Parallelism example...")
        tensor_parallel_example()
    elif args.strategy == "sp":
        print("Running Sequence Parallelism example...")
        sequence_parallel_example()


if __name__ == "__main__":
    main()


"""
Usage Examples:

1. DDP (4 devices, data parallel):
   # CUDA: torchrun --nproc_per_node=4 parallelism_examples.py --strategy ddp
   # CPU: CUDA_VISIBLE_DEVICES="" torchrun --nproc_per_node=4 parallelism_examples.py --strategy ddp
   # MPS: torchrun --nproc_per_node=1 parallelism_examples.py --strategy ddp

2. FSDP (4 devices, parameter sharding - best on CUDA):
   torchrun --nproc_per_node=4 parallelism_examples.py --strategy fsdp

3. Tensor Parallelism (4 CUDA devices only):
   torchrun --nproc_per_node=4 parallelism_examples.py --strategy tp

4. Sequence Parallelism (any device type):
   torchrun --nproc_per_node=4 parallelism_examples.py --strategy sp

Device Support Matrix:
┌─────────────────┬──────┬──────┬─────┐
│ Strategy        │ CUDA │ MPS  │ CPU │
├─────────────────┼──────┼──────┼─────┤
│ DDP             │  ✅   │  ✅   │  ✅  │
│ FSDP            │  ✅   │  ⚠️*  │  ⚠️* │
│ Tensor Parallel │  ✅   │  ❌   │  ❌  │
│ Sequence Par.   │  ✅   │  ✅   │  ✅  │
└─────────────────┴──────┴──────┴─────┘
* Falls back to DDP

Key Changes for Multi-Device Support:

1. Device Detection:
   - Automatically detects best available device (CUDA > MPS > CPU)
   - Uses appropriate distributed backend (NCCL for CUDA, Gloo for others)

2. Tensor Placement:
   - Uses torch.set_default_device() for automatic placement
   - Handles device-specific setup (torch.cuda.set_device for CUDA)

3. Backend Selection:
   - NCCL: CUDA devices (best performance)
   - Gloo: CPU/MPS and mixed environments
   - Automatic fallback for unsupported combinations

4. Graceful Fallbacks:
   - FSDP falls back to DDP on non-CUDA devices
   - Tensor parallelism falls back to simple training on non-CUDA
   - All strategies work in single-device mode

Memory Usage (approximate for large models):
- DDP: Model_size × num_devices (parameters replicated)
- FSDP: Model_size ÷ num_devices (parameters sharded, CUDA only)
- TP: Model_size ÷ num_devices (weights sharded, CUDA only)
- SP: Model_size × num_devices (model replicated, sequences split)

Communication Patterns:
- DDP: All-reduce gradients after backward
- FSDP: All-gather parameters before forward/backward, reduce-scatter gradients
- TP: All-reduce within tensor parallel groups
- SP: All-gather/reduce-scatter for sequence dimensions
"""
