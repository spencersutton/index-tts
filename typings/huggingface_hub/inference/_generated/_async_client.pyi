from collections.abc import AsyncIterable
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from huggingface_hub.inference._common import ContentT
from huggingface_hub.inference._generated.types import (
    AudioClassificationOutputElement,
    AudioClassificationOutputTransform,
    AudioToAudioOutputElement,
    AutomaticSpeechRecognitionOutput,
    ChatCompletionInputGrammarType,
    ChatCompletionInputMessage,
    ChatCompletionInputStreamOptions,
    ChatCompletionInputTool,
    ChatCompletionInputToolChoiceClass,
    ChatCompletionInputToolChoiceEnum,
    ChatCompletionOutput,
    ChatCompletionStreamOutput,
    DocumentQuestionAnsweringOutputElement,
    FillMaskOutputElement,
    ImageClassificationOutputElement,
    ImageClassificationOutputTransform,
    ImageSegmentationOutputElement,
    ImageSegmentationSubtask,
    ImageToImageTargetSize,
    ImageToTextOutput,
    ImageToVideoTargetSize,
    ObjectDetectionOutputElement,
    Padding,
    QuestionAnsweringOutputElement,
    SummarizationOutput,
    SummarizationTruncationStrategy,
    TableQuestionAnsweringOutputElement,
    TextClassificationOutputElement,
    TextClassificationOutputTransform,
    TextGenerationInputGrammarType,
    TextGenerationOutput,
    TextGenerationStreamOutput,
    TextToSpeechEarlyStoppingEnum,
    TokenClassificationAggregationStrategy,
    TokenClassificationOutputElement,
    TranslationOutput,
    TranslationTruncationStrategy,
    VisualQuestionAnsweringOutputElement,
    ZeroShotClassificationOutputElement,
    ZeroShotImageClassificationOutputElement,
)
from huggingface_hub.inference._providers import PROVIDER_OR_POLICY_T
from PIL.Image import Image

if TYPE_CHECKING: ...
logger = ...
MODEL_KWARGS_NOT_USED_REGEX = ...

class AsyncInferenceClient:
    def __init__(
        self,
        model: str | None = ...,
        *,
        provider: PROVIDER_OR_POLICY_T | None = ...,
        token: str | None = ...,
        timeout: float | None = ...,
        headers: dict[str, str] | None = ...,
        cookies: dict[str, str] | None = ...,
        trust_env: bool = ...,
        proxies: Any | None = ...,
        bill_to: str | None = ...,
        base_url: str | None = ...,
        api_key: str | None = ...,
    ) -> None: ...
    async def __aenter__(self):  # -> Self:
        ...
    async def __aexit__(self, exc_type, exc_value, traceback):  # -> None:
        ...
    def __del__(self) -> None:  # -> None:
        ...
    async def close(self):  # -> None:

        ...
    async def audio_classification(
        self,
        audio: ContentT,
        *,
        model: str | None = ...,
        top_k: int | None = ...,
        function_to_apply: AudioClassificationOutputTransform | None = ...,
    ) -> list[AudioClassificationOutputElement]: ...
    async def audio_to_audio(self, audio: ContentT, *, model: str | None = ...) -> list[AudioToAudioOutputElement]: ...
    async def automatic_speech_recognition(
        self, audio: ContentT, *, model: str | None = ..., extra_body: dict | None = ...
    ) -> AutomaticSpeechRecognitionOutput: ...
    @overload
    async def chat_completion(
        self,
        messages: list[dict | ChatCompletionInputMessage],
        *,
        model: str | None = ...,
        stream: Literal[False] = ...,
        frequency_penalty: float | None = ...,
        logit_bias: list[float] | None = ...,
        logprobs: bool | None = ...,
        max_tokens: int | None = ...,
        n: int | None = ...,
        presence_penalty: float | None = ...,
        response_format: ChatCompletionInputGrammarType | None = ...,
        seed: int | None = ...,
        stop: list[str] | None = ...,
        stream_options: ChatCompletionInputStreamOptions | None = ...,
        temperature: float | None = ...,
        tool_choice: ChatCompletionInputToolChoiceClass | ChatCompletionInputToolChoiceEnum | None = ...,
        tool_prompt: str | None = ...,
        tools: list[ChatCompletionInputTool] | None = ...,
        top_logprobs: int | None = ...,
        top_p: float | None = ...,
        extra_body: dict | None = ...,
    ) -> ChatCompletionOutput: ...
    @overload
    async def chat_completion(
        self,
        messages: list[dict | ChatCompletionInputMessage],
        *,
        model: str | None = ...,
        stream: Literal[True] = ...,
        frequency_penalty: float | None = ...,
        logit_bias: list[float] | None = ...,
        logprobs: bool | None = ...,
        max_tokens: int | None = ...,
        n: int | None = ...,
        presence_penalty: float | None = ...,
        response_format: ChatCompletionInputGrammarType | None = ...,
        seed: int | None = ...,
        stop: list[str] | None = ...,
        stream_options: ChatCompletionInputStreamOptions | None = ...,
        temperature: float | None = ...,
        tool_choice: ChatCompletionInputToolChoiceClass | ChatCompletionInputToolChoiceEnum | None = ...,
        tool_prompt: str | None = ...,
        tools: list[ChatCompletionInputTool] | None = ...,
        top_logprobs: int | None = ...,
        top_p: float | None = ...,
        extra_body: dict | None = ...,
    ) -> AsyncIterable[ChatCompletionStreamOutput]: ...
    @overload
    async def chat_completion(
        self,
        messages: list[dict | ChatCompletionInputMessage],
        *,
        model: str | None = ...,
        stream: bool = ...,
        frequency_penalty: float | None = ...,
        logit_bias: list[float] | None = ...,
        logprobs: bool | None = ...,
        max_tokens: int | None = ...,
        n: int | None = ...,
        presence_penalty: float | None = ...,
        response_format: ChatCompletionInputGrammarType | None = ...,
        seed: int | None = ...,
        stop: list[str] | None = ...,
        stream_options: ChatCompletionInputStreamOptions | None = ...,
        temperature: float | None = ...,
        tool_choice: ChatCompletionInputToolChoiceClass | ChatCompletionInputToolChoiceEnum | None = ...,
        tool_prompt: str | None = ...,
        tools: list[ChatCompletionInputTool] | None = ...,
        top_logprobs: int | None = ...,
        top_p: float | None = ...,
        extra_body: dict | None = ...,
    ) -> ChatCompletionOutput | AsyncIterable[ChatCompletionStreamOutput]: ...
    async def chat_completion(
        self,
        messages: list[dict | ChatCompletionInputMessage],
        *,
        model: str | None = ...,
        stream: bool = ...,
        frequency_penalty: float | None = ...,
        logit_bias: list[float] | None = ...,
        logprobs: bool | None = ...,
        max_tokens: int | None = ...,
        n: int | None = ...,
        presence_penalty: float | None = ...,
        response_format: ChatCompletionInputGrammarType | None = ...,
        seed: int | None = ...,
        stop: list[str] | None = ...,
        stream_options: ChatCompletionInputStreamOptions | None = ...,
        temperature: float | None = ...,
        tool_choice: ChatCompletionInputToolChoiceClass | ChatCompletionInputToolChoiceEnum | None = ...,
        tool_prompt: str | None = ...,
        tools: list[ChatCompletionInputTool] | None = ...,
        top_logprobs: int | None = ...,
        top_p: float | None = ...,
        extra_body: dict | None = ...,
    ) -> ChatCompletionOutput | AsyncIterable[ChatCompletionStreamOutput]: ...
    async def document_question_answering(
        self,
        image: ContentT,
        question: str,
        *,
        model: str | None = ...,
        doc_stride: int | None = ...,
        handle_impossible_answer: bool | None = ...,
        lang: str | None = ...,
        max_answer_len: int | None = ...,
        max_question_len: int | None = ...,
        max_seq_len: int | None = ...,
        top_k: int | None = ...,
        word_boxes: list[list[float] | str] | None = ...,
    ) -> list[DocumentQuestionAnsweringOutputElement]: ...
    async def feature_extraction(
        self,
        text: str,
        *,
        normalize: bool | None = ...,
        prompt_name: str | None = ...,
        truncate: bool | None = ...,
        truncation_direction: Literal["Left", "Right"] | None = ...,
        model: str | None = ...,
    ) -> np.ndarray: ...
    async def fill_mask(
        self, text: str, *, model: str | None = ..., targets: list[str] | None = ..., top_k: int | None = ...
    ) -> list[FillMaskOutputElement]: ...
    async def image_classification(
        self,
        image: ContentT,
        *,
        model: str | None = ...,
        function_to_apply: ImageClassificationOutputTransform | None = ...,
        top_k: int | None = ...,
    ) -> list[ImageClassificationOutputElement]: ...
    async def image_segmentation(
        self,
        image: ContentT,
        *,
        model: str | None = ...,
        mask_threshold: float | None = ...,
        overlap_mask_area_threshold: float | None = ...,
        subtask: ImageSegmentationSubtask | None = ...,
        threshold: float | None = ...,
    ) -> list[ImageSegmentationOutputElement]: ...
    async def image_to_image(
        self,
        image: ContentT,
        prompt: str | None = ...,
        *,
        negative_prompt: str | None = ...,
        num_inference_steps: int | None = ...,
        guidance_scale: float | None = ...,
        model: str | None = ...,
        target_size: ImageToImageTargetSize | None = ...,
        **kwargs,
    ) -> Image: ...
    async def image_to_video(
        self,
        image: ContentT,
        *,
        model: str | None = ...,
        prompt: str | None = ...,
        negative_prompt: str | None = ...,
        num_frames: float | None = ...,
        num_inference_steps: int | None = ...,
        guidance_scale: float | None = ...,
        seed: int | None = ...,
        target_size: ImageToVideoTargetSize | None = ...,
        **kwargs,
    ) -> bytes: ...
    async def image_to_text(self, image: ContentT, *, model: str | None = ...) -> ImageToTextOutput: ...
    async def object_detection(
        self, image: ContentT, *, model: str | None = ..., threshold: float | None = ...
    ) -> list[ObjectDetectionOutputElement]: ...
    async def question_answering(
        self,
        question: str,
        context: str,
        *,
        model: str | None = ...,
        align_to_words: bool | None = ...,
        doc_stride: int | None = ...,
        handle_impossible_answer: bool | None = ...,
        max_answer_len: int | None = ...,
        max_question_len: int | None = ...,
        max_seq_len: int | None = ...,
        top_k: int | None = ...,
    ) -> QuestionAnsweringOutputElement | list[QuestionAnsweringOutputElement]: ...
    async def sentence_similarity(
        self, sentence: str, other_sentences: list[str], *, model: str | None = ...
    ) -> list[float]: ...
    async def summarization(
        self,
        text: str,
        *,
        model: str | None = ...,
        clean_up_tokenization_spaces: bool | None = ...,
        generate_parameters: dict[str, Any] | None = ...,
        truncation: SummarizationTruncationStrategy | None = ...,
    ) -> SummarizationOutput: ...
    async def table_question_answering(
        self,
        table: dict[str, Any],
        query: str,
        *,
        model: str | None = ...,
        padding: Padding | None = ...,
        sequential: bool | None = ...,
        truncation: bool | None = ...,
    ) -> TableQuestionAnsweringOutputElement: ...
    async def tabular_classification(self, table: dict[str, Any], *, model: str | None = ...) -> list[str]: ...
    async def tabular_regression(self, table: dict[str, Any], *, model: str | None = ...) -> list[float]: ...
    async def text_classification(
        self,
        text: str,
        *,
        model: str | None = ...,
        top_k: int | None = ...,
        function_to_apply: TextClassificationOutputTransform | None = ...,
    ) -> list[TextClassificationOutputElement]: ...
    @overload
    async def text_generation(
        self,
        prompt: str,
        *,
        details: Literal[True],
        stream: Literal[True],
        model: str | None = ...,
        adapter_id: str | None = ...,
        best_of: int | None = ...,
        decoder_input_details: bool | None = ...,
        do_sample: bool | None = ...,
        frequency_penalty: float | None = ...,
        grammar: TextGenerationInputGrammarType | None = ...,
        max_new_tokens: int | None = ...,
        repetition_penalty: float | None = ...,
        return_full_text: bool | None = ...,
        seed: int | None = ...,
        stop: list[str] | None = ...,
        stop_sequences: list[str] | None = ...,
        temperature: float | None = ...,
        top_k: int | None = ...,
        top_n_tokens: int | None = ...,
        top_p: float | None = ...,
        truncate: int | None = ...,
        typical_p: float | None = ...,
        watermark: bool | None = ...,
    ) -> AsyncIterable[TextGenerationStreamOutput]: ...
    @overload
    async def text_generation(
        self,
        prompt: str,
        *,
        details: Literal[True],
        stream: Literal[False] | None = ...,
        model: str | None = ...,
        adapter_id: str | None = ...,
        best_of: int | None = ...,
        decoder_input_details: bool | None = ...,
        do_sample: bool | None = ...,
        frequency_penalty: float | None = ...,
        grammar: TextGenerationInputGrammarType | None = ...,
        max_new_tokens: int | None = ...,
        repetition_penalty: float | None = ...,
        return_full_text: bool | None = ...,
        seed: int | None = ...,
        stop: list[str] | None = ...,
        stop_sequences: list[str] | None = ...,
        temperature: float | None = ...,
        top_k: int | None = ...,
        top_n_tokens: int | None = ...,
        top_p: float | None = ...,
        truncate: int | None = ...,
        typical_p: float | None = ...,
        watermark: bool | None = ...,
    ) -> TextGenerationOutput: ...
    @overload
    async def text_generation(
        self,
        prompt: str,
        *,
        details: Literal[False] | None = ...,
        stream: Literal[True],
        model: str | None = ...,
        adapter_id: str | None = ...,
        best_of: int | None = ...,
        decoder_input_details: bool | None = ...,
        do_sample: bool | None = ...,
        frequency_penalty: float | None = ...,
        grammar: TextGenerationInputGrammarType | None = ...,
        max_new_tokens: int | None = ...,
        repetition_penalty: float | None = ...,
        return_full_text: bool | None = ...,
        seed: int | None = ...,
        stop: list[str] | None = ...,
        stop_sequences: list[str] | None = ...,
        temperature: float | None = ...,
        top_k: int | None = ...,
        top_n_tokens: int | None = ...,
        top_p: float | None = ...,
        truncate: int | None = ...,
        typical_p: float | None = ...,
        watermark: bool | None = ...,
    ) -> AsyncIterable[str]: ...
    @overload
    async def text_generation(
        self,
        prompt: str,
        *,
        details: Literal[False] | None = ...,
        stream: Literal[False] | None = ...,
        model: str | None = ...,
        adapter_id: str | None = ...,
        best_of: int | None = ...,
        decoder_input_details: bool | None = ...,
        do_sample: bool | None = ...,
        frequency_penalty: float | None = ...,
        grammar: TextGenerationInputGrammarType | None = ...,
        max_new_tokens: int | None = ...,
        repetition_penalty: float | None = ...,
        return_full_text: bool | None = ...,
        seed: int | None = ...,
        stop: list[str] | None = ...,
        stop_sequences: list[str] | None = ...,
        temperature: float | None = ...,
        top_k: int | None = ...,
        top_n_tokens: int | None = ...,
        top_p: float | None = ...,
        truncate: int | None = ...,
        typical_p: float | None = ...,
        watermark: bool | None = ...,
    ) -> str: ...
    @overload
    async def text_generation(
        self,
        prompt: str,
        *,
        details: bool | None = ...,
        stream: bool | None = ...,
        model: str | None = ...,
        adapter_id: str | None = ...,
        best_of: int | None = ...,
        decoder_input_details: bool | None = ...,
        do_sample: bool | None = ...,
        frequency_penalty: float | None = ...,
        grammar: TextGenerationInputGrammarType | None = ...,
        max_new_tokens: int | None = ...,
        repetition_penalty: float | None = ...,
        return_full_text: bool | None = ...,
        seed: int | None = ...,
        stop: list[str] | None = ...,
        stop_sequences: list[str] | None = ...,
        temperature: float | None = ...,
        top_k: int | None = ...,
        top_n_tokens: int | None = ...,
        top_p: float | None = ...,
        truncate: int | None = ...,
        typical_p: float | None = ...,
        watermark: bool | None = ...,
    ) -> str | TextGenerationOutput | AsyncIterable[str] | AsyncIterable[TextGenerationStreamOutput]: ...
    async def text_generation(
        self,
        prompt: str,
        *,
        details: bool | None = ...,
        stream: bool | None = ...,
        model: str | None = ...,
        adapter_id: str | None = ...,
        best_of: int | None = ...,
        decoder_input_details: bool | None = ...,
        do_sample: bool | None = ...,
        frequency_penalty: float | None = ...,
        grammar: TextGenerationInputGrammarType | None = ...,
        max_new_tokens: int | None = ...,
        repetition_penalty: float | None = ...,
        return_full_text: bool | None = ...,
        seed: int | None = ...,
        stop: list[str] | None = ...,
        stop_sequences: list[str] | None = ...,
        temperature: float | None = ...,
        top_k: int | None = ...,
        top_n_tokens: int | None = ...,
        top_p: float | None = ...,
        truncate: int | None = ...,
        typical_p: float | None = ...,
        watermark: bool | None = ...,
    ) -> str | TextGenerationOutput | AsyncIterable[str] | AsyncIterable[TextGenerationStreamOutput]: ...
    async def text_to_image(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = ...,
        height: int | None = ...,
        width: int | None = ...,
        num_inference_steps: int | None = ...,
        guidance_scale: float | None = ...,
        model: str | None = ...,
        scheduler: str | None = ...,
        seed: int | None = ...,
        extra_body: dict[str, Any] | None = ...,
    ) -> Image: ...
    async def text_to_video(
        self,
        prompt: str,
        *,
        model: str | None = ...,
        guidance_scale: float | None = ...,
        negative_prompt: list[str] | None = ...,
        num_frames: float | None = ...,
        num_inference_steps: int | None = ...,
        seed: int | None = ...,
        extra_body: dict[str, Any] | None = ...,
    ) -> bytes: ...
    async def text_to_speech(
        self,
        text: str,
        *,
        model: str | None = ...,
        do_sample: bool | None = ...,
        early_stopping: bool | TextToSpeechEarlyStoppingEnum | None = ...,
        epsilon_cutoff: float | None = ...,
        eta_cutoff: float | None = ...,
        max_length: int | None = ...,
        max_new_tokens: int | None = ...,
        min_length: int | None = ...,
        min_new_tokens: int | None = ...,
        num_beam_groups: int | None = ...,
        num_beams: int | None = ...,
        penalty_alpha: float | None = ...,
        temperature: float | None = ...,
        top_k: int | None = ...,
        top_p: float | None = ...,
        typical_p: float | None = ...,
        use_cache: bool | None = ...,
        extra_body: dict[str, Any] | None = ...,
    ) -> bytes: ...
    async def token_classification(
        self,
        text: str,
        *,
        model: str | None = ...,
        aggregation_strategy: TokenClassificationAggregationStrategy | None = ...,
        ignore_labels: list[str] | None = ...,
        stride: int | None = ...,
    ) -> list[TokenClassificationOutputElement]: ...
    async def translation(
        self,
        text: str,
        *,
        model: str | None = ...,
        src_lang: str | None = ...,
        tgt_lang: str | None = ...,
        clean_up_tokenization_spaces: bool | None = ...,
        truncation: TranslationTruncationStrategy | None = ...,
        generate_parameters: dict[str, Any] | None = ...,
    ) -> TranslationOutput: ...
    async def visual_question_answering(
        self, image: ContentT, question: str, *, model: str | None = ..., top_k: int | None = ...
    ) -> list[VisualQuestionAnsweringOutputElement]: ...
    async def zero_shot_classification(
        self,
        text: str,
        candidate_labels: list[str],
        *,
        multi_label: bool | None = ...,
        hypothesis_template: str | None = ...,
        model: str | None = ...,
    ) -> list[ZeroShotClassificationOutputElement]: ...
    async def zero_shot_image_classification(
        self,
        image: ContentT,
        candidate_labels: list[str],
        *,
        model: str | None = ...,
        hypothesis_template: str | None = ...,
        labels: list[str] = ...,
    ) -> list[ZeroShotImageClassificationOutputElement]: ...
    async def get_endpoint_info(self, *, model: str | None = ...) -> dict[str, Any]: ...
    async def health_check(self, model: str | None = ...) -> bool: ...
    @property
    def chat(self) -> ProxyClientChat: ...

class _ProxyClient:
    def __init__(self, client: AsyncInferenceClient) -> None: ...

class ProxyClientChat(_ProxyClient):
    @property
    def completions(self) -> ProxyClientChatCompletions: ...

class ProxyClientChatCompletions(_ProxyClient):
    @property
    def create(self):  # -> Callable[..., Any]:
        ...
