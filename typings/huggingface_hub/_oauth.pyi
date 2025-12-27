import datetime
from dataclasses import dataclass
from typing import Literal

import fastapi

from .utils import experimental

logger = ...

@dataclass
class OAuthOrgInfo:
    sub: str
    name: str
    preferred_username: str
    picture: str
    is_enterprise: bool
    can_pay: bool | None = ...
    role_in_org: str | None = ...
    security_restrictions: list[Literal["ip", "token-policy", "mfa", "sso"]] | None = ...

@dataclass
class OAuthUserInfo:
    sub: str
    name: str
    preferred_username: str
    email_verified: bool | None
    email: str | None
    picture: str
    profile: str
    website: str | None
    is_pro: bool
    can_pay: bool | None
    orgs: list[OAuthOrgInfo] | None

@dataclass
class OAuthInfo:
    access_token: str
    access_token_expires_at: datetime.datetime
    user_info: OAuthUserInfo
    state: str | None
    scope: str

@experimental
def attach_huggingface_oauth(app: fastapi.FastAPI, route_prefix: str = ...):  # -> None:
    ...
def parse_huggingface_oauth(request: fastapi.Request) -> OAuthInfo | None: ...
