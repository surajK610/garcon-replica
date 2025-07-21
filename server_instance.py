#!/usr/bin/env python3
"""
Individual server instance for distributed training testing.
Each server runs as a separate process with its own rank.
"""

import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import os
import time
import platform
import threading
import queue
from typing import Optional

def setup_process_group(rank: int, world_size: int, port: int = 12359):
    """Initialize process group for this server instance."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    
    # Use gloo backend (works well on Mac and Linux)
    dist.init_process_group(
        backend='gloo',
        rank=rank,
        world_size=world_size,
        timeout=torch.distributed.default_pg_timeout
    )
    
    print(f"🔧 Server {rank}/{world_size} initialized successfully")

def get_best_device() -> str:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

class SimpleModel(nn.Module):
    """Simple model for testing distributed training."""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

class DistributedServer:
    """Server instance for distributed training."""
    
    def __init__(self, rank: int, world_size: int, device: str):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.model = SimpleModel().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.running = True
        self.message_queue = queue.Queue()
        
        print(f"🤖 Server {rank} initialized on device: {device}")
    
    def sync_gradients(self):
        for param in self.model.parameters():
            if param.grad is not None:
                # Move to CPU for gloo communication
                cpu_grad = param.grad.cpu() if self.device in ['mps', 'cuda'] else param.grad
                
                # All-reduce with SUM operation
                dist.all_reduce(cpu_grad, op=dist.ReduceOp.SUM)
                
                # Average the gradients
                cpu_grad /= self.world_size
                
                # Move back to original device
                if self.device in ['mps', 'cuda']:
                    param.grad.copy_(cpu_grad.to(self.device))
                else:
                    param.grad.copy_(cpu_grad)
    
    def train_step(self, batch_size: int = 32):
        """Perform one training step."""
        # Generate synthetic data
        torch.manual_seed(42 + self.rank)  # Different seed per rank
        data = torch.randn(batch_size, 64).to(self.device)
        targets = torch.randint(0, 10, (batch_size,)).to(self.device)
        
        # Forward pass
        self.model.train()
        outputs = self.model(data)
        loss = self.criterion(outputs, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Synchronize gradients
        self.sync_gradients()
        
        # Update parameters
        self.optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            predicted = torch.argmax(outputs, dim=1)
            accuracy = (predicted == targets).float().mean()
        
        return loss.item(), accuracy.item()
    
    def collective_operations_demo(self):
        """Demonstrate collective operations."""
        print(f"📡 Server {self.rank}: Running collective operations demo")
        
        # All-reduce demo
        tensor = torch.tensor([float(self.rank + 1)])
        print(f"📊 Server {self.rank} initial value: {tensor.item()}")
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"✅ Server {self.rank} all-reduce result: {tensor.item()}")
        
        gather_tensor = torch.tensor([float(self.rank * 10)])
        gather_list = [torch.zeros_like(gather_tensor) for _ in range(self.world_size)]
        dist.all_gather(gather_list, gather_tensor)
        print(f"📥 Server {self.rank} all-gather result: {[t.item() for t in gather_list]}")
        
        if self.rank == 0:
            broadcast_tensor = torch.tensor([999.0])
        else:
            broadcast_tensor = torch.tensor([0.0])
        
        print(f"📡 Server {self.rank} before broadcast: {broadcast_tensor.item()}")
        dist.broadcast(broadcast_tensor, src=0)
        print(f"✅ Server {self.rank} after broadcast: {broadcast_tensor.item()}")
        
        dist.barrier()
    
    def run_training(self, epochs: int = 5):
        """Run training for specified epochs."""
        print(f"🏋️ Server {self.rank}: Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            loss, accuracy = self.train_step()
            
            # Synchronize before printing (cleaner output)
            dist.barrier()
            
            if epoch % 1 == 0:
                print(f"🔄 Server {self.rank}, Epoch {epoch}: Loss={loss:.4f}, Acc={accuracy:.4f}")
            
            dist.barrier()
        
        print(f"✅ Server {self.rank}: Training completed!")
    
    def parameter_sync_check(self):
        """Check if parameters are synchronized across servers."""
        print(f"🔍 Server {self.rank}: Checking parameter synchronization")
        
        first_param = next(self.model.parameters())
        cpu_param = first_param.cpu() if self.device in ['mps', 'cuda'] else first_param
        
        param_list = [torch.zeros_like(cpu_param) for _ in range(self.world_size)]
        dist.all_gather(param_list, cpu_param)
        
        if self.rank == 0:
            all_same = all(torch.allclose(param_list[0], p, atol=1e-6) for p in param_list[1:])
            print(f"✅ Parameters synchronized: {all_same}")
        
        dist.barrier()
    
    def run(self):
        """Main server loop."""
        try:
            print(f"🚀 Server {self.rank} starting main loop")
            
            self.collective_operations_demo()
            self.run_training()
            self.parameter_sync_check()
            
            print(f"🎉 Server {self.rank} completed successfully!")
            
        except Exception as e:
            print(f"❌ Server {self.rank} error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='Distributed Training Server Instance')
    parser.add_argument('--rank', type=int, required=True, help='Rank of this server')
    parser.add_argument('--world-size', type=int, required=True, help='Total number of servers')
    parser.add_argument('--port', type=int, default=12359, help='Master port')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    
    args = parser.parse_args()
    
    setup_process_group(args.rank, args.world_size, args.port)
    
    device = get_best_device()
    
    server = DistributedServer(args.rank, args.world_size, device)
    server.run()

if __name__ == "__main__":
    main()
