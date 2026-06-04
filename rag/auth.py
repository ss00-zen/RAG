import requests
from jose import jwt

KEYCLOAK_ISSUER = "http://localhost:8080/realms/rag_application"
JWKS_URL = f"{KEYCLOAK_ISSUER}/protocol/openid-connect/certs"
AUDIENCE = "backend-api"

jwks = requests.get(JWKS_URL).json()


def verify_token(token: str):
    header = jwt.get_unverified_header(token)
    key = next(k for k in jwks["keys"] if k["kid"] == header["kid"])

    return jwt.decode(
        token,
        key,
        algorithms=["RS256"],
        audience=AUDIENCE,
        issuer=KEYCLOAK_ISSUER,
    )
