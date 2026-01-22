import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence

class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=self.rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier() # synchronize all processes
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm", create=False)
                self.loop()

    # TODO
    def exit(self):
        pass

    # TODO
    def loop(self):
        pass

    # TODO
    def read_shm(self):
        pass

    # TODO
    def write_shm(self):
        pass

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    # TODO
    def warmup_model(self):
        pass

    # TODO
    def allocate_kv_cache(self):
        pass

    # TODO
    def prepare_block_tables(self, seqs: list[Sequence]):
        pass

    # TODO
    def prepare_prefill(self, seqs: list[Sequence]):
        pass

    # TODO
    def prepare_decode(self, seqs: list[Sequence]):
        pass


    # TODO
    def prepare_decode(self, seqs: list[Sequence]):
        pass

    # TODO
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        pass

    # TODO
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        pass

    # TODO
    def capture_cudagraph(self):
        pass
    