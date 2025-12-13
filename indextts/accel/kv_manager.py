import hashlib
import pickle  # noqa: S403
from collections import deque
from collections.abc import Iterable, MutableSequence, Sequence
from copy import copy
from typing import overload

import torch
from transformers import GPT2Model

from indextts.s2mel.modules.gpt_fast.model import KVCache


class KVCacheBlock:
    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_cnt = 0
        self._block_hash = None
        self.token_ids: Sequence[int] = []

    @property
    def block_hash(self) -> bytes | None:
        return self._block_hash

    def update(self, block_hash: bytes, token_ids: Sequence[int]) -> None:
        self._block_hash = block_hash
        self.token_ids = token_ids

    def reset(self) -> None:
        self.ref_cnt = 1
        self._block_hash = None
        self.token_ids = []


class Seq(Sequence[int]):
    def __init__(self, token_ids: MutableSequence[int], block_size: int = 256) -> None:
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1] if token_ids else 0
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table: list[int] = []
        self.block_size = block_size

    def __len__(self) -> int:
        return self.num_tokens

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice[int, int | None, int | None]) -> Sequence[int]: ...

    def __getitem__(self, index: int | slice[int, int | None, int | None]) -> int | Sequence[int]:
        return self.token_ids[index]

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def num_cached_blocks(self) -> int:
        return self.num_cached_tokens // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def get_block_tokens(self, block_idx: int) -> list[int]:
        assert 0 <= block_idx < self.num_blocks
        start = block_idx * self.block_size
        end = start + self.block_size
        return list(self.token_ids[start:end])

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1


class KVCacheManager:
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int,
        num_blocks: int,
        dtype: torch.dtype,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = dtype

        self.blocks = [KVCacheBlock(i) for i in range(num_blocks)]
        self.block_hash_to_id: dict[bytes, int] = {}
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_dtype = torch.float16 if device == "cuda" else dtype
        self.kv_cache = torch.empty(
            2,
            num_layers,
            num_blocks,
            block_size,
            num_heads,
            head_dim,
            dtype=cache_dtype,
            device=device,
        )

    @classmethod
    def compute_block_hash(cls, token_ids: Iterable[int], parent_hash: bytes | None = None) -> bytes:
        hash_input: list[bytes | int] = []
        if parent_hash is not None:
            hash_input.append(parent_hash)
        hash_input.extend(token_ids)
        input_bytes = pickle.dumps(tuple(hash_input), protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(input_bytes).digest()

    def _allocate_block(self, block_id: int) -> KVCacheBlock:
        block = self.blocks[block_id]
        assert block.ref_cnt == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_cnt == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def allocate(self, sequence: Seq) -> None:
        assert not sequence.block_table, "Sequence already has allocated blocks"

        parent_hash = None
        cache_miss = False

        for i in range(sequence.num_blocks):
            token_ids = sequence.get_block_tokens(i)
            block_hash = self.compute_block_hash(token_ids, parent_hash) if len(token_ids) == self.block_size else None
            block_id = self.block_hash_to_id.get(block_hash) if block_hash else None

            if block_id is None or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                sequence.num_cached_tokens += self.block_size
                if block_id is not None and block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_cnt += 1
                else:
                    block_id = self.free_block_ids[0]
                    block = self._allocate_block(block_id)

            if block_hash is not None:
                block.update(block_hash, token_ids)
                self.block_hash_to_id[block_hash] = block_id
                parent_hash = block_hash

            sequence.block_table.append(block_id)

    def deallocate(self, sequence: Seq) -> None:
        for block_id in reversed(sequence.block_table):
            block = self.blocks[block_id]
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self._deallocate_block(block_id)

        sequence.num_cached_tokens = 0
        sequence.block_table.clear()

    def append_to_seq(self, sequence: Seq) -> None:
        block_table = sequence.block_table
        last_block = self.blocks[block_table[-1]]

        if len(sequence) % self.block_size == 1:
            assert last_block.block_hash is not None
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(sequence) % self.block_size == 0:
            assert last_block.block_hash is None
            token_ids = sequence.get_block_tokens(sequence.num_blocks - 1)
            parent_hash = self.blocks[block_table[-2]].block_hash if len(block_table) > 1 else None
            block_hash = self.compute_block_hash(token_ids, parent_hash)
            last_block.update(block_hash, token_ids)
            self.block_hash_to_id[block_hash] = last_block.block_id
        else:
            assert last_block.block_hash is None

    def remove_seq(self, sequence: Seq) -> None:
        self.deallocate(sequence)

    def wire_kv_cache_to_model(self, model: GPT2Model) -> None:
        layer_id = 0
        for module in model.modules():
            if isinstance(module, KVCache):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
