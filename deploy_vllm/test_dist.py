import torch
import torch.distributed as dist
import os

def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank() 
    world_size = dist.get_world_size()
    print(f"Process {rank}/{world_size} initialized!")
    
    # Simple tensor test
    x = torch.ones(2, 2).cuda(rank)
    dist.all_reduce(x)
    print(f"Rank {rank}: {x}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()