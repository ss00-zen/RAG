from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set


APP_ROLE_ADMIN = "admin"
APP_ROLE_USER = "user"

PERM_CHAT_READ = "chat:read"
PERM_UPLOAD_WRITE = "upload:write"
PERM_CLASSIFIED_READ = "classified:read"
PERM_UNCLASSIFIED_READ = "unclassified:read"


@dataclass
class AuthContext:
    user_id: str
    username: Optional[str] = None
    provider: Optional[str] = None
    roles: Set[str] = field(default_factory=set)
    permissions: Set[str] = field(default_factory=set)
    claims: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "provider": self.provider,
            "roles": sorted(self.roles),
            "permissions": sorted(self.permissions),
            "claims": self.claims,
        }


def _normalize_role_name(role: Any) -> Optional[str]:
    if role is None:
        return None

    role = str(role).strip().lower()
    if not role:
        return None

    if role.startswith("org:"):
        role = role.split(":", 1)[1]

    return role


def _extract_roles(claims: Dict[str, Any]) -> Set[str]:
    roles: Set[str] = set()

    # Keycloak realm roles
    for role in claims.get("realm_access", {}).get("roles", []):
        normalized = _normalize_role_name(role)
        if normalized:
            roles.add(normalized)

    # Keycloak client roles
    resource_access = claims.get("resource_access", {})
    if isinstance(resource_access, dict):
        for client_data in resource_access.values():
            if isinstance(client_data, dict):
                for role in client_data.get("roles", []):
                    normalized = _normalize_role_name(role)
                    if normalized:
                        roles.add(normalized)

    # Clerk org role
    org_claim = claims.get("o") or {}
    if isinstance(org_claim, dict):
        org_role = _normalize_role_name(org_claim.get("rol"))
        if org_role:
            roles.add(org_role)

    # Clerk custom roles
    for role in claims.get("app_roles", []):
        normalized = _normalize_role_name(role)
        if normalized:
            roles.add(normalized)

    # SuperTokens / generic roles
    for role in claims.get("roles", []):
        normalized = _normalize_role_name(role)
        if normalized:
            roles.add(normalized)

    return roles


def _extract_permissions(claims: Dict[str, Any]) -> Set[str]:
    permissions: Set[str] = set()

    # Clerk org permissions
    org_claim = claims.get("o") or {}
    if isinstance(org_claim, dict):
        org_permissions = org_claim.get("per", [])
        if isinstance(org_permissions, list):
            for perm in org_permissions:
                normalized = str(perm).strip().lower()
                if normalized:
                    permissions.add(normalized)

    # Generic/custom permissions
    for perm in claims.get("app_permissions", []):
        normalized = str(perm).strip().lower()
        if normalized:
            permissions.add(normalized)

    for perm in claims.get("permissions", []):
        normalized = str(perm).strip().lower()
        if normalized:
            permissions.add(normalized)

    return permissions


def _default_permissions_from_roles(roles: Set[str]) -> Set[str]:
    if APP_ROLE_ADMIN in roles:
        return {
            PERM_CHAT_READ,
            PERM_UPLOAD_WRITE,
            PERM_CLASSIFIED_READ,
            PERM_UNCLASSIFIED_READ,
        }

    return {
        PERM_CHAT_READ,
        PERM_UNCLASSIFIED_READ,
    }


def normalize_user(claims: Dict[str, Any], provider: Optional[str] = None) -> AuthContext:
    if not claims:
        raise RuntimeError("Missing claims for auth normalization")

    user_id = claims.get("sub") or claims.get("user_id")
    if not user_id:
        raise RuntimeError("Token missing user identifier (sub/user_id)")

    username = (
        claims.get("preferred_username")
        or claims.get("username")
        or claims.get("email")
    )

    roles = _extract_roles(claims)
    if APP_ROLE_ADMIN not in roles:
        roles.add(APP_ROLE_USER)

    permissions = _extract_permissions(claims)
    permissions.update(_default_permissions_from_roles(roles))

    return AuthContext(
        user_id=user_id,
        username=username,
        provider=provider,
        roles=roles,
        permissions=permissions,
        claims=claims,
    )


def is_admin(ctx: AuthContext) -> bool:
    return APP_ROLE_ADMIN in ctx.roles


def can_upload(ctx: AuthContext) -> bool:
    return PERM_UPLOAD_WRITE in ctx.permissions


def can_view_classified(ctx: AuthContext) -> bool:
    return PERM_CLASSIFIED_READ in ctx.permissions


def role_from_auth(ctx: AuthContext) -> str:
    return APP_ROLE_ADMIN if is_admin(ctx) else APP_ROLE_USER
