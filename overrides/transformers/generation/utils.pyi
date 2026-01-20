import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
from _typeshed import Incomplete
from transformers.cache_utils import Cache as Cache
from transformers.cache_utils import DynamicCache as DynamicCache
from transformers.cache_utils import EncoderDecoderCache as EncoderDecoderCache
from transformers.cache_utils import HybridChunkedCache as HybridChunkedCache
from transformers.cache_utils import OffloadedCache as OffloadedCache
from transformers.cache_utils import OffloadedHybridCache as OffloadedHybridCache
from transformers.cache_utils import QuantizedCacheConfig as QuantizedCacheConfig
from transformers.configuration_utils import PretrainedConfig as PretrainedConfig
from transformers.dynamic_module_utils import check_python_requirements as check_python_requirements
from transformers.dynamic_module_utils import get_cached_module_file as get_cached_module_file
from transformers.dynamic_module_utils import get_class_in_module as get_class_in_module
from transformers.dynamic_module_utils import resolve_trust_remote_code as resolve_trust_remote_code
from transformers.generation.beam_constraints import DisjunctiveConstraint as DisjunctiveConstraint
from transformers.generation.beam_constraints import PhrasalConstraint as PhrasalConstraint
from transformers.generation.beam_search import BeamScorer as BeamScorer
from transformers.generation.beam_search import BeamSearchScorer as BeamSearchScorer
from transformers.generation.beam_search import ConstrainedBeamSearchScorer as ConstrainedBeamSearchScorer
from transformers.generation.candidate_generator import AssistantVocabTranslatorCache as AssistantVocabTranslatorCache
from transformers.generation.candidate_generator import AssistedCandidateGenerator as AssistedCandidateGenerator
from transformers.generation.candidate_generator import (
    AssistedCandidateGeneratorDifferentTokenizers as AssistedCandidateGeneratorDifferentTokenizers,
)
from transformers.generation.candidate_generator import CandidateGenerator as CandidateGenerator
from transformers.generation.candidate_generator import EarlyExitCandidateGenerator as EarlyExitCandidateGenerator
from transformers.generation.candidate_generator import PromptLookupCandidateGenerator as PromptLookupCandidateGenerator
from transformers.generation.candidate_generator import (
    UniversalSpeculativeDecodingGenerator as UniversalSpeculativeDecodingGenerator,
)
from transformers.generation.configuration_utils import (
    NEED_SETUP_CACHE_CLASSES_MAPPING as NEED_SETUP_CACHE_CLASSES_MAPPING,
)
from transformers.generation.configuration_utils import QUANT_BACKEND_CLASSES_MAPPING as QUANT_BACKEND_CLASSES_MAPPING
from transformers.generation.configuration_utils import GenerationConfig as GenerationConfig
from transformers.generation.configuration_utils import GenerationMode as GenerationMode
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor as EncoderNoRepeatNGramLogitsProcessor,
)
from transformers.generation.logits_process import (
    EncoderRepetitionPenaltyLogitsProcessor as EncoderRepetitionPenaltyLogitsProcessor,
)
from transformers.generation.logits_process import EpsilonLogitsWarper as EpsilonLogitsWarper
from transformers.generation.logits_process import EtaLogitsWarper as EtaLogitsWarper
from transformers.generation.logits_process import ExponentialDecayLengthPenalty as ExponentialDecayLengthPenalty
from transformers.generation.logits_process import ForcedBOSTokenLogitsProcessor as ForcedBOSTokenLogitsProcessor
from transformers.generation.logits_process import ForcedEOSTokenLogitsProcessor as ForcedEOSTokenLogitsProcessor
from transformers.generation.logits_process import HammingDiversityLogitsProcessor as HammingDiversityLogitsProcessor
from transformers.generation.logits_process import InfNanRemoveLogitsProcessor as InfNanRemoveLogitsProcessor
from transformers.generation.logits_process import LogitNormalization as LogitNormalization
from transformers.generation.logits_process import LogitsProcessorList as LogitsProcessorList
from transformers.generation.logits_process import MinLengthLogitsProcessor as MinLengthLogitsProcessor
from transformers.generation.logits_process import (
    MinNewTokensLengthLogitsProcessor as MinNewTokensLengthLogitsProcessor,
)
from transformers.generation.logits_process import MinPLogitsWarper as MinPLogitsWarper
from transformers.generation.logits_process import NoBadWordsLogitsProcessor as NoBadWordsLogitsProcessor
from transformers.generation.logits_process import NoRepeatNGramLogitsProcessor as NoRepeatNGramLogitsProcessor
from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor as PrefixConstrainedLogitsProcessor
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor as RepetitionPenaltyLogitsProcessor
from transformers.generation.logits_process import SequenceBiasLogitsProcessor as SequenceBiasLogitsProcessor
from transformers.generation.logits_process import (
    SuppressTokensAtBeginLogitsProcessor as SuppressTokensAtBeginLogitsProcessor,
)
from transformers.generation.logits_process import SuppressTokensLogitsProcessor as SuppressTokensLogitsProcessor
from transformers.generation.logits_process import TemperatureLogitsWarper as TemperatureLogitsWarper
from transformers.generation.logits_process import TopKLogitsWarper as TopKLogitsWarper
from transformers.generation.logits_process import TopPLogitsWarper as TopPLogitsWarper
from transformers.generation.logits_process import TypicalLogitsWarper as TypicalLogitsWarper
from transformers.generation.logits_process import (
    UnbatchedClassifierFreeGuidanceLogitsProcessor as UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from transformers.generation.stopping_criteria import ConfidenceCriteria as ConfidenceCriteria
from transformers.generation.stopping_criteria import EosTokenCriteria as EosTokenCriteria
from transformers.generation.stopping_criteria import MaxLengthCriteria as MaxLengthCriteria
from transformers.generation.stopping_criteria import MaxTimeCriteria as MaxTimeCriteria
from transformers.generation.stopping_criteria import StoppingCriteria as StoppingCriteria
from transformers.generation.stopping_criteria import StoppingCriteriaList as StoppingCriteriaList
from transformers.generation.stopping_criteria import StopStringCriteria as StopStringCriteria
from transformers.generation.streamers import BaseStreamer as BaseStreamer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled as is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module as is_fsdp_managed_module
from transformers.modeling_outputs import CausalLMOutputWithPast as CausalLMOutputWithPast
from transformers.modeling_outputs import Seq2SeqLMOutput as Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel as PreTrainedModel
from transformers.pytorch_utils import isin_mps_friendly as isin_mps_friendly
from transformers.tokenization_utils import ExtensionsTrie as ExtensionsTrie
from transformers.tokenization_utils_base import PreTrainedTokenizerBase as PreTrainedTokenizerBase
from transformers.utils import ModelOutput as ModelOutput
from transformers.utils import is_accelerate_available as is_accelerate_available
from transformers.utils import is_hqq_available as is_hqq_available
from transformers.utils import is_optimum_quanto_available as is_optimum_quanto_available
from transformers.utils import is_torchdynamo_exporting as is_torchdynamo_exporting
from transformers.utils import logging as logging

logger: Incomplete
ALL_CACHE_NAMES: Incomplete

@dataclass
class GenerateDecoderOnlyOutput(ModelOutput):
    sequences: torch.Tensor
    scores: tuple[torch.Tensor] | None = ...
    logits: tuple[torch.Tensor] | None = ...
    attentions: tuple[tuple[torch.Tensor]] | None = ...
    hidden_states: tuple[tuple[torch.Tensor]] | None = ...
    past_key_values: tuple[tuple[tuple[torch.Tensor]]] | None = ...

@dataclass
class GenerateEncoderDecoderOutput(ModelOutput):
    sequences: torch.Tensor
    scores: tuple[torch.Tensor] | None = ...
    logits: tuple[torch.Tensor] | None = ...
    encoder_attentions: tuple[torch.Tensor] | None = ...
    encoder_hidden_states: tuple[torch.Tensor] | None = ...
    decoder_attentions: tuple[tuple[torch.Tensor]] | None = ...
    cross_attentions: tuple[tuple[torch.Tensor]] | None = ...
    decoder_hidden_states: tuple[tuple[torch.Tensor]] | None = ...
    past_key_values: tuple[tuple[tuple[torch.Tensor]]] | None = ...

@dataclass
class GenerateBeamDecoderOnlyOutput(ModelOutput):
    sequences: torch.Tensor
    sequences_scores: torch.Tensor | None = ...
    scores: tuple[torch.Tensor] | None = ...
    logits: tuple[torch.Tensor] | None = ...
    beam_indices: torch.Tensor | None = ...
    attentions: tuple[tuple[torch.Tensor]] | None = ...
    hidden_states: tuple[tuple[torch.Tensor]] | None = ...
    past_key_values: tuple[tuple[tuple[torch.Tensor]]] | None = ...

@dataclass
class GenerateBeamEncoderDecoderOutput(ModelOutput):
    sequences: torch.Tensor
    sequences_scores: torch.Tensor | None = ...
    scores: tuple[torch.Tensor] | None = ...
    logits: tuple[torch.Tensor] | None = ...
    beam_indices: torch.Tensor | None = ...
    encoder_attentions: tuple[torch.Tensor] | None = ...
    encoder_hidden_states: tuple[torch.Tensor] | None = ...
    decoder_attentions: tuple[tuple[torch.Tensor]] | None = ...
    cross_attentions: tuple[tuple[torch.Tensor]] | None = ...
    decoder_hidden_states: tuple[tuple[torch.Tensor]] | None = ...
    past_key_values: tuple[tuple[tuple[torch.Tensor]]] | None = ...

GreedySearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
ContrastiveSearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
SampleDecoderOnlyOutput = GenerateDecoderOnlyOutput
ContrastiveSearchEncoderDecoderOutput = GenerateEncoderDecoderOutput
GreedySearchEncoderDecoderOutput = GenerateEncoderDecoderOutput
SampleEncoderDecoderOutput = GenerateEncoderDecoderOutput
BeamSearchDecoderOnlyOutput = GenerateBeamDecoderOnlyOutput
BeamSampleDecoderOnlyOutput = GenerateBeamDecoderOnlyOutput
BeamSearchEncoderDecoderOutput = GenerateBeamEncoderDecoderOutput
BeamSampleEncoderDecoderOutput = GenerateBeamEncoderDecoderOutput
type GreedySearchOutput = GreedySearchEncoderDecoderOutput | GreedySearchDecoderOnlyOutput
type SampleOutput = SampleEncoderDecoderOutput | SampleDecoderOnlyOutput
type BeamSearchOutput = BeamSearchEncoderDecoderOutput | BeamSearchDecoderOnlyOutput
type BeamSampleOutput = BeamSampleEncoderDecoderOutput | BeamSampleDecoderOnlyOutput
type ContrastiveSearchOutput = ContrastiveSearchEncoderDecoderOutput | ContrastiveSearchDecoderOnlyOutput
type GenerateNonBeamOutput = GenerateDecoderOnlyOutput | GenerateEncoderDecoderOutput
type GenerateBeamOutput = GenerateBeamDecoderOnlyOutput | GenerateBeamEncoderDecoderOutput
type GenerateOutput = GenerateNonBeamOutput | GenerateBeamOutput

class GenerationMixin:
    def load_custom_generate(
        self,
        pretrained_model_name_or_path: str | os.PathLike | None = None,
        trust_remote_code: bool | None = None,
        **kwargs,
    ) -> Callable: ...
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ): ...
    def compute_transition_scores(
        self,
        sequences: torch.Tensor,
        scores: tuple[torch.Tensor],
        beam_indices: torch.Tensor | None = None,
        normalize_logits: bool = False,
    ) -> torch.Tensor: ...
    def generate(
        self,
        inputs: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        logits_processor: LogitsProcessorList | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]] | None = None,
        synced_gpus: bool | None = None,
        assistant_model: PreTrainedModel | None = None,
        streamer: BaseStreamer | None = None,
        negative_prompt_ids: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        use_model_defaults: bool | None = None,
        custom_generate: str | None = None,
        **kwargs,
    ) -> GenerateOutput | torch.Tensor: ...
    def heal_tokens(
        self, input_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase | None = None
    ) -> torch.Tensor: ...

def stack_model_outputs(model_outputs: list[ModelOutput], config: PretrainedConfig) -> ModelOutput: ...
