import torch
from abc import ABC, abstractmethod
from typing import Optional, Union
from ..utils import add_start_docstrings
from .beam_constraints import Constraint

PROCESS_INPUTS_DOCSTRING = ...
FINALIZE_INPUTS_DOCSTRING = ...

class BeamScorer(ABC):
    @abstractmethod
    @add_start_docstrings(PROCESS_INPUTS_DOCSTRING)
    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        **kwargs,
    ) -> tuple[torch.Tensor]: ...
    @abstractmethod
    @add_start_docstrings(FINALIZE_INPUTS_DOCSTRING)
    def finalize(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        max_length: int,
        **kwargs,
    ) -> torch.LongTensor: ...

class BeamSearchScorer(BeamScorer):
    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = ...,
        do_early_stopping: Optional[Union[bool, str]] = ...,
        num_beam_hyps_to_keep: Optional[int] = ...,
        num_beam_groups: Optional[int] = ...,
        max_length: Optional[int] = ...,
    ) -> None: ...
    @property
    def is_done(self) -> bool: ...
    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[Union[int, torch.Tensor]] = ...,
        eos_token_id: Optional[Union[int, list[int], torch.Tensor]] = ...,
        beam_indices: Optional[torch.LongTensor] = ...,
        group_index: Optional[int] = ...,
        decoder_prompt_len: Optional[int] = ...,
    ) -> dict[str, torch.Tensor]: ...
    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[Union[int, torch.Tensor]] = ...,
        eos_token_id: Optional[Union[int, list[int], torch.Tensor]] = ...,
        beam_indices: Optional[torch.LongTensor] = ...,
        decoder_prompt_len: Optional[int] = ...,
    ) -> tuple[torch.LongTensor]: ...

class ConstrainedBeamSearchScorer(BeamScorer):
    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        constraints: list[Constraint],
        device: torch.device,
        length_penalty: Optional[float] = ...,
        do_early_stopping: Optional[Union[bool, str]] = ...,
        num_beam_hyps_to_keep: Optional[int] = ...,
        num_beam_groups: Optional[int] = ...,
        max_length: Optional[int] = ...,
    ) -> None: ...
    @property
    def is_done(self) -> bool: ...
    def make_constraint_states(self, n):  # -> list[ConstraintListState]:
        ...
    def check_completes_constraints(self, sequence):  # -> bool:
        ...
    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        scores_for_all_vocab: torch.FloatTensor,
        pad_token_id: Optional[Union[int, torch.Tensor]] = ...,
        eos_token_id: Optional[Union[int, list[int], torch.Tensor]] = ...,
        beam_indices: Optional[torch.LongTensor] = ...,
        decoder_prompt_len: Optional[int] = ...,
    ) -> tuple[torch.Tensor]: ...
    def step_sentence_constraint(
        self,
        batch_idx: int,
        input_ids: torch.LongTensor,
        vocab_scores: torch.FloatTensor,
        sent_beam_scores: torch.FloatTensor,
        sent_beam_tokens: torch.LongTensor,
        sent_beam_indices: torch.LongTensor,
        push_progress: bool = ...,
    ):  # -> tuple[FloatTensor, LongTensor, LongTensor]:
        ...
    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[Union[int, torch.Tensor]] = ...,
        eos_token_id: Optional[Union[int, list[int], torch.Tensor]] = ...,
        beam_indices: Optional[torch.LongTensor] = ...,
        decoder_prompt_len: Optional[int] = ...,
    ) -> tuple[torch.LongTensor]: ...

class BeamHypotheses:
    def __init__(
        self, num_beams: int, length_penalty: float, early_stopping: bool, max_length: Optional[int] = ...
    ) -> None: ...
    def __len__(self):  # -> int:

        ...
    def add(
        self,
        hyp: torch.LongTensor,
        sum_logprobs: float,
        beam_indices: Optional[torch.LongTensor] = ...,
        generated_len: Optional[int] = ...,
    ):  # -> None:

        ...
    def is_done(self, best_sum_logprobs: float, cur_len: int, decoder_prompt_len: Optional[int] = ...) -> bool: ...
