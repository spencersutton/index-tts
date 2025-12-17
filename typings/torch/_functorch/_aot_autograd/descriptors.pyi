import dataclasses

@dataclasses.dataclass(frozen=True)
class AOTInput:
    def expr(self) -> str: ...
    def is_param(self) -> bool: ...
    def is_buffer(self) -> bool: ...
    def is_tangent(self) -> bool: ...

@dataclasses.dataclass(frozen=True)
class DifferentiableAOTInput(AOTInput): ...

@dataclasses.dataclass(frozen=True)
class AOTOutput:
    def expr(self) -> str: ...
    def is_grad(self) -> bool: ...

@dataclasses.dataclass(frozen=True)
class DifferentiableAOTOutput(AOTOutput): ...

@dataclasses.dataclass(frozen=True)
class ParamAOTInput(DifferentiableAOTInput):
    target: str
    def expr(self) -> str: ...
    def is_param(self) -> bool: ...
    def is_buffer(self) -> bool: ...

@dataclasses.dataclass(frozen=True)
class BufferAOTInput(DifferentiableAOTInput):
    target: str
    def expr(self) -> str: ...
    def is_param(self) -> bool: ...
    def is_buffer(self) -> bool: ...

@dataclasses.dataclass(frozen=True)
class DummyAOTInput(AOTInput):
    idx: int
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class PlainAOTInput(DifferentiableAOTInput):
    idx: int
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class SubclassGetAttrAOTInput(AOTInput):
    base: AOTInput
    attr: str
    def expr(self) -> str: ...
    def is_param(self) -> bool: ...
    def is_buffer(self) -> bool: ...
    def is_tangent(self) -> bool: ...

@dataclasses.dataclass(frozen=True)
class SubclassSizeAOTInput(AOTInput):
    base: AOTInput
    idx: int
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class SubclassStrideAOTInput(AOTInput):
    base: AOTInput
    idx: int
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class ViewBaseAOTInput(DifferentiableAOTInput):
    base_of: AOTInput
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class SyntheticBaseAOTInput(DifferentiableAOTInput):
    base_of: AOTInput
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class PhiloxForwardSeedAOTInput(AOTInput):
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class PhiloxForwardBaseOffsetAOTInput(AOTInput):
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class PhiloxBackwardSeedAOTInput(AOTInput):
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class PhiloxBackwardBaseOffsetAOTInput(AOTInput):
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class ForwardTokenAOTInput(AOTInput):
    idx: int
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class BackwardTokenAOTInput(AOTInput):
    idx: int
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class TangentAOTInput(DifferentiableAOTInput):
    output: DifferentiableAOTOutput
    def __post_init__(self) -> None: ...
    def expr(self) -> str: ...
    def is_tangent(self) -> bool: ...

@dataclasses.dataclass(frozen=True)
class PlainAOTOutput(DifferentiableAOTOutput):
    idx: int
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class InputMutationAOTOutput(DifferentiableAOTOutput):
    mutated_input: AOTInput
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class IntermediateBaseAOTOutput(DifferentiableAOTOutput):
    base_of: AOTOutput
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class MetadataMutationAOTOutput(DifferentiableAOTOutput):
    idx: int
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class GradAOTOutput(DifferentiableAOTOutput):
    grad_of: DifferentiableAOTInput
    def __post_init__(self) -> None: ...
    def expr(self) -> str: ...
    def is_grad(self) -> bool: ...

@dataclasses.dataclass(frozen=True)
class PhiloxUpdatedForwardOffsetAOTOutput(AOTOutput):
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class PhiloxUpdatedBackwardOffsetAOTOutput(AOTOutput):
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class ForwardTokenAOTOutput(AOTOutput):
    idx: int
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class BackwardTokenAOTOutput(AOTOutput):
    idx: int
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class SubclassGetAttrAOTOutput(AOTOutput):
    base: AOTOutput
    attr: str
    def expr(self) -> str: ...
    def is_grad(self) -> bool: ...

@dataclasses.dataclass(frozen=True)
class SubclassSizeAOTOutput(AOTOutput):
    base: AOTOutput
    idx: int
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class SubclassStrideAOTOutput(AOTOutput):
    base: AOTOutput
    idx: int
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class DummyAOTOutput(AOTOutput):
    idx: int
    def expr(self) -> str: ...

@dataclasses.dataclass(frozen=True)
class SavedForBackwardsAOTOutput(AOTOutput):
    idx: int
    def expr(self) -> str: ...
