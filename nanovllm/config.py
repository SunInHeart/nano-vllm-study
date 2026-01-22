import os
from dataclasses import dataclass
from transformers import AutoConfig

@dataclass # Simplify the writing of classes used for storing data.
class Config:
    model: str # Path to the pretrained model directory
    max_num_batched_tokens: int = 16384 # Maximum total tokens that can be precessed in a single batch
    max_num_seqs: int = 512 # Maximum number of the sequences that can be processed concurrently
    max_model_len: int = 4096 # Maximum sequence length the model can handle
    gpu_memory_utilization: float = 0.9 # Fraction of GPU memory to allocate (0.9 = 90%)
    tensor_parallel_size: int = 1 # Number of GPUs to use for tensor parallelism (1-8)
    enforce_eager: bool = False # Wether to disable CUDA graphs and use eager mode
    hf_config: AutoConfig | None = None # Will store HuggingFace model config
    eos: int = -1 # End-of-sequence token ID (default -1 means not set)
    kvcache_block_size: int = 256 # Size of blocks for KV cache allocation (must be multiple of 256)
    num_kvcache_blocks: int = -1 # Total number of KV cache blocks (-1 means auto-caculate)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        # Tensor parallelism must use between 1-8 GPUs
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        # Batch capacity must be able to handle at least one full-length sequence
        assert self.max_num_batched_tokens >= self.max_model_len





