from dataclasses import dataclass

from torchaudio.models import SquimObjective, SquimSubjective

@dataclass
class SquimObjectiveBundle:
    _path: str
    _sample_rate: float
    def get_model(self) -> SquimObjective: ...
    @property
    def sample_rate(self): ...

SQUIM_OBJECTIVE = ...

@dataclass
class SquimSubjectiveBundle:
    _path: str
    _sample_rate: float
    def get_model(self) -> SquimSubjective: ...
    @property
    def sample_rate(self): ...

SQUIM_SUBJECTIVE = ...
