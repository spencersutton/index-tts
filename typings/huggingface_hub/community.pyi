from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypedDict

"""
Data structures to interact with Discussions and Pull Requests on the Hub.

See [the Discussions and Pull Requests guide](https://huggingface.co/docs/hub/repositories-pull-requests-discussions)
for more information on Pull Requests, Discussions, and the community tab.
"""
type DiscussionStatus = Literal["open", "closed", "merged", "draft"]

@dataclass
class Discussion:
    title: str
    status: DiscussionStatus
    num: int
    repo_id: str
    repo_type: str
    author: str
    is_pull_request: bool
    created_at: datetime
    endpoint: str
    @property
    def git_reference(self) -> str | None: ...
    @property
    def url(self) -> str: ...

@dataclass
class DiscussionWithDetails(Discussion):
    events: list[DiscussionEvent]
    conflicting_files: list[str] | bool | None
    target_branch: str | None
    merge_commit_oid: str | None
    diff: str | None

class DiscussionEventArgs(TypedDict):
    id: str
    type: str
    created_at: datetime
    author: str
    _event: dict

@dataclass
class DiscussionEvent:
    id: str
    type: str
    created_at: datetime
    author: str
    _event: dict

@dataclass
class DiscussionComment(DiscussionEvent):
    content: str
    edited: bool
    hidden: bool
    @property
    def rendered(self) -> str: ...
    @property
    def last_edited_at(self) -> datetime: ...
    @property
    def last_edited_by(self) -> str: ...
    @property
    def edit_history(self) -> list[dict]: ...
    @property
    def number_of_edits(self) -> int: ...

@dataclass
class DiscussionStatusChange(DiscussionEvent):
    new_status: str

@dataclass
class DiscussionCommit(DiscussionEvent):
    summary: str
    oid: str

@dataclass
class DiscussionTitleChange(DiscussionEvent):
    old_title: str
    new_title: str

def deserialize_event(event: dict) -> DiscussionEvent: ...
