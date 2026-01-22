from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence

class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = [-1]

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = list[Block] = [Block[i] for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64() # 创建64位的xxhash哈希器
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little")) # 将前缀加入哈希
        h.update(np.array(token_ids).tobytes()) # 将token_ids加入哈希
        return h.intdigest() # 计算整数哈希值
    
    def _allocate_block(self, block_id: int) -> Block: # allocate a block with the given block_id, return the block
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]
    
    def _deallocate_block(self, block_id: int) -> Block: # note that deacllocate don't clear the hash table
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks
    
    # for prefill
    def allocate(self, seq: Sequence): # allocate blocks for the sequece, update the block table and hash table
        assert not seq.block_table # make sure the first time to allocate blocks (block table is empty)
        h = -1 # hash of the first block, will be updated later
        cache_miss = False
        for i in range(seq.num_blocks): # seq.num_blocks is the number of blocks needed to store the sequence
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1 # compute hash only if the block is full
            block_id = self.hash_to_block_id.get(h, -1) # get the block id from hash table, if not found, use -1 # for prefix cache
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids: # block_id=-1 means cache miss, or the block is not the same as the cached block
                cache_miss = True
            if cache_miss: # if cache miss, allocate a new block
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else: # maybe hash table has the block_id but used_block_ids is cleared
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)
    
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table): # deallocate from end to start
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0: # only release if ref_count is 0
                self._deallocate_block(block_id) 
        seq.num_cached_blocks = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence): # check whether memory is enough for allocate next block
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1) # only need one more block if the last block has only one token

    def may_append(self, seq: Sequence): # allocate next block if needed
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]] # get the last block in the block table
        if len(seq) % self.block_size == 1: # has only one token in the last block
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0: # the last block is full, we need to update the hash and token_ids
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else: # has gt 1 and lt block_size tokens in the last block, do nothing
            assert last_block.hash == -1
