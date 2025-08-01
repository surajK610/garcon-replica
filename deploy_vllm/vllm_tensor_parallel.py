#!/usr/bin/env python3

import os
import torch
import torch.distributed as dist
from vllm import LLM, SamplingParams

# Debug info
local_rank = int(os.environ.get('LOCAL_RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))
rank = int(os.environ.get('RANK', 0))

print(f"🔧 Process {rank}/{world_size} (local_rank: {local_rank}) starting...")

# Set device BEFORE any CUDA operations
torch.cuda.set_device(local_rank)
print(f"📱 Process using GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")

# Check memory
free, total = torch.cuda.mem_get_info()
print(f"📊 GPU {local_rank}: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total")

local_model_path = "./qwen3-1.7b"

# Create prompts
prompts = [
    "Hello, my name is",
    "The capital of France is",
]

sampling_params = SamplingParams(
    temperature=0.8, 
    top_p=0.95, 
    max_tokens=30  # Keep it small for testing
)

print(f"🚀 Process {rank} initializing LLM...")

try:
    # Very conservative settings to avoid memory issues
    llm = LLM(
        model=local_model_path,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
        distributed_executor_backend="external_launcher",
        max_model_len=1024,           # Very small context
        gpu_memory_utilization=0.6,   # Very conservative
        trust_remote_code=True,
        enforce_eager=True,           # No CUDA graphs
        enable_prefix_caching=False,  # No caching
        enable_chunked_prefill=False, # No chunked prefill
        swap_space=0,                 # No swapping
        dtype="bfloat16",            # Use bfloat16 for better stability
        load_format="safetensors",   # Explicit format
        revision=None,               # No specific revision
        quantization=None,           # No quantization
        disable_sliding_window=True, # Disable sliding window
        disable_custom_all_reduce=True, # Use standard NCCL
    )
    
    print(f"✅ Process {rank} LLM initialized successfully!")

    # Generate
    print(f"🎯 Process {rank} starting generation...")
    outputs = llm.generate(prompts, sampling_params)

    # Only rank 0 prints results
    if rank == 0:
        print("\n" + "="*60)
        print("GENERATION RESULTS")
        print("="*60)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}")
            print(f"Generated: {generated_text!r}")
            print("-" * 50)
    
    print(f"🎉 Process {rank} completed successfully!")

except Exception as e:
    print(f"❌ Process {rank} failed: {e}")
    import traceback
    traceback.print_exc()
    raise

finally:
    # Cleanup
    try:
        if 'llm' in locals():
            del llm
        torch.cuda.empty_cache()
    except:
        pass