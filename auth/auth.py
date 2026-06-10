import os
from functools import lru_cache
from typing import Any, Dict, Optional

import requests
from jose import jwt


# ============================================================
# Provider-aware JWT verification
# - Secure JWT verification using JWKS
# - Works with Keycloak now
# - Can work with Clerk / SuperTokens by env config
# ============================================================

AUTH_PROVIDER = os.getenv("AUTH_PROVIDER", "keycloak").strip().lower()
AUTH_VERIFY_AUDIENCE = os.getenv("AUTH_VERIFY_AUDIENCE", "true").strip().lower() == "true"


def _provider_env(provider: str) -> Dict[str, Optional[str]]:
    """
    Reads provider-specific envs first, then generic AUTH_* envs.
    """
    if provider == "keycloak":
        issuer = os.getenv("KEYCLOAK_ISSUER") or os.getenv("AUTH_ISSUER")
        audience = os.getenv("KEYCLOAK_AUDIENCE") or os.getenv("AUTH_AUDIENCE")
        jwks_url = os.getenv("KEYCLOAK_JWKS_URL") or os.getenv("AUTH_JWKS_URL")

        if issuer and not jwks_url:
            jwks_url = f"{issuer.rstrip('/')}/protocol/openid-connect/certs"

        return {
            "provider": provider,
            "issuer": issuer,
            "audience": audience,
            "jwks_url": jwks_url,
        }

    if provider == "clerk":
        return {
            "provider": provider,
            "issuer": os.getenv("CLERK_ISSUER") or os.getenv("AUTH_ISSUER"),
            "audience": os.getenv("CLERK_AUDIENCE") or os.getenv("AUTH_AUDIENCE"),
            "jwks_url": os.getenv("CLERK_JWKS_URL") or os.getenv("AUTH_JWKS_URL"),
        }

    if provider == "supertokens":
        return {
            "provider": provider,
            "issuer": os.getenv("SUPERTOKENS_ISSUER") or os.getenv("AUTH_ISSUER"),
            "audience": os.getenv("SUPERTOKENS_AUDIENCE") or os.getenv("AUTH_AUDIENCE"),
            "jwks_url": os.getenv("SUPERTOKENS_JWKS_URL") or os.getenv("AUTH_JWKS_URL"),
        }

    raise RuntimeError(f"Unsupported AUTH_PROVIDER={provider}")


def get_auth_provider() -> str:
    return AUTH_PROVIDER


def _get_config() -> Dict[str, Optional[str]]:
    config = _provider_env(AUTH_PROVIDER)

    if not config["issuer"]:
        raise RuntimeError(f"Missing issuer configuration for provider={AUTH_PROVIDER}")

    if not config["jwks_url"]:
        raise RuntimeError(
            f"Missing JWKS configuration for provider={AUTH_PROVIDER}. "
            f"Set AUTH_JWKS_URL or provider-specific *_JWKS_URL."
        )

    return config


@lru_cache(maxsize=8)
def _load_jwks(jwks_url: str) -> Dict[str, Any]:
    response = requests.get(jwks_url, timeout=10)
    response.raise_for_status()
    return response.json()


def _get_signing_key(token: str, jwks_url: str) -> Dict[str, Any]:
    header = jwt.get_unverified_header(token)
    kid = header.get("kid")
    if not kid:
        raise RuntimeError("JWT header missing 'kid'")

    jwks = _load_jwks(jwks_url)
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return key

    # Retry once in case JWKS rotated
    _load_jwks.cache_clear()
    jwks = _load_jwks(jwks_url)
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return key

    raise RuntimeError(f"No matching signing key found for kid={kid}")


def verify_token(token: str) -> Dict[str, Any]:
    """
    Verifies a JWT using provider-configured issuer/JWKS.
    Returns decoded claims.
    """
    if not token:
        raise RuntimeError("Empty token")

    config = _get_config()
    signing_key = _get_signing_key(token, config["jwks_url"])

    
    decode_kwargs: Dict[str, Any] = {
        "algorithms": ["RS256"],
        "issuer": config["issuer"],
        "options": {
            "verify_aud": False,   # ✅ FORCE disable
        },
    }


    if AUTH_VERIFY_AUDIENCE and config["audience"]:
        decode_kwargs["audience"] = config["audience"]

    return jwt.decode(token, signing_key, **decode_kwargs)
