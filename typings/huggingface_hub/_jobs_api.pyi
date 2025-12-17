from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from huggingface_hub._space_api import SpaceHardware

class JobStage(StrEnum):
    COMPLETED = ...
    CANCELED = ...
    ERROR = ...
    DELETED = ...
    RUNNING = ...

@dataclass
class JobStatus:
    stage: JobStage
    message: str | None

@dataclass
class JobOwner:
    id: str
    name: str
    type: str

@dataclass
class JobInfo:
    id: str
    created_at: datetime | None
    docker_image: str | None
    space_id: str | None
    command: list[str] | None
    arguments: list[str] | None
    environment: dict[str, Any] | None
    secrets: dict[str, Any] | None
    flavor: SpaceHardware | None
    status: JobStatus
    owner: JobOwner
    endpoint: str
    url: str
    def __init__(self, **kwargs) -> None: ...

@dataclass
class JobSpec:
    docker_image: str | None
    space_id: str | None
    command: list[str] | None
    arguments: list[str] | None
    environment: dict[str, Any] | None
    secrets: dict[str, Any] | None
    flavor: SpaceHardware | None
    timeout: int | None
    tags: list[str] | None
    arch: str | None
    def __init__(self, **kwargs) -> None: ...

@dataclass
class LastJobInfo:
    id: str
    at: datetime
    def __init__(self, **kwargs) -> None: ...

@dataclass
class ScheduledJobStatus:
    last_job: LastJobInfo | None
    next_job_run_at: datetime | None
    def __init__(self, **kwargs) -> None: ...

@dataclass
class ScheduledJobInfo:
    id: str
    created_at: datetime | None
    job_spec: JobSpec
    schedule: str | None
    suspend: bool | None
    concurrency: bool | None
    status: ScheduledJobStatus
    owner: JobOwner
    def __init__(self, **kwargs) -> None: ...
