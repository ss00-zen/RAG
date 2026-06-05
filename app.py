import streamlit as st
import requests
from jose import jwt

from rag.logger import logger

st.set_page_config(page_title="Enterprise RAG Demo")

# ✅ FIXED URL
KEYCLOAK_URL = "http://localhost:8080"
REALM = "rag_application"
CLIENT_ID = "backend-api"
CLIENT_SECRET = "Hdi2qBsE8OsFV4GV8LfxriI7fs7BUM9k"  # 🔐 rotate this

TOKEN_URL = f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/token"
KEYCLOAK_ISSUER = f"{KEYCLOAK_URL}/realms/{REALM}"
JWKS_URL = f"{KEYCLOAK_ISSUER}/protocol/openid-connect/certs"
LOGOUT_URL = f"{KEYCLOAK_ISSUER}/protocol/openid-connect/logout"

@st.cache_resource
def load_jwks():
    return requests.get(JWKS_URL, timeout=10).json()

jwks = load_jwks()

def verify_token(token: str):
    header = jwt.get_unverified_header(token)
    key = next(k for k in jwks["keys"] if k["kid"] == header["kid"])
    return jwt.decode(
        token,
        key,
        algorithms=["RS256"],
        options={"verify_aud": False},
        issuer=KEYCLOAK_ISSUER,
    )

# ✅ FIXED TOKEN REQUEST (matches curl)
def token_request(payload: dict):
    payload["client_id"] = CLIENT_ID
    payload["client_secret"] = CLIENT_SECRET

    return requests.post(
        TOKEN_URL,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )

def refresh_access_token(refresh_token):
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    res = token_request(payload)
    if res.status_code == 200:
        data = res.json()
        return data["access_token"], data.get("refresh_token")
    return None, None

def logout():
    if st.session_state.refresh_token:
        try:
            requests.post(
                LOGOUT_URL,
                data={
                    "client_id": CLIENT_ID,
                    "client_secret": CLIENT_SECRET,
                    "refresh_token": st.session_state.refresh_token,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10,
            )
        except Exception:
            pass

    # ✅ CRITICAL: clear session
    st.session_state.user = None
    st.session_state.access_token = None
    st.session_state.refresh_token = None

# session state
if "user" not in st.session_state:
    st.session_state.user = None
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = None

# login UI
st.sidebar.title("🔐 Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button("Login"):
    payload = {
        "grant_type": "password",
        "username": username.strip(),
        "password": password.strip(),
    }

    res = token_request(payload)
    logger.info("Login attempt for user %s, status=%s", username, res.status_code)
    logger.debug("Login response: %s", res.text)

    st.write("TOKEN_URL:", TOKEN_URL)
    st.write("LOGIN_STATUS:", res.status_code)
    st.write("LOGIN_RESPONSE:", res.text)

    if res.status_code == 200:
        tokens = res.json()
        token = tokens["access_token"]
        refresh = tokens.get("refresh_token")

        try:
            user = verify_token(token)
            st.session_state.access_token = token
            st.session_state.refresh_token = refresh
            st.session_state.user = user
            st.sidebar.success(f"✅ {user.get('preferred_username')}")
            logger.info("Login successful for user %s", user.get('preferred_username'))
            st.rerun()
        except Exception as e:
            logger.exception("Token verification failed for user %s", username)
            st.sidebar.error(f"❌ Token verification failed: {str(e)}")
    else:
        st.sidebar.error(f"❌ Login failed")


# ✅ Logout button
if st.session_state.user:
    if st.sidebar.button("🚪 Logout"):
        logout()
        st.sidebar.success("Logged out")
        st.rerun()


# auto-refresh
# ✅ Always validate token on every run
if st.session_state.access_token:
    try:
        # try validating current token
        verify_token(st.session_state.access_token)
    except Exception:
        # token expired → refresh
        if st.session_state.refresh_token:
            logger.info("Refreshing access token for user %s", st.session_state.user.get('preferred_username') if st.session_state.user else 'unknown')
            new_access, new_refresh = refresh_access_token(st.session_state.refresh_token)

            if new_access:
                st.session_state.access_token = new_access
                st.session_state.refresh_token = new_refresh
                logger.info("Access token refreshed successfully")
            else:
                logger.warning("Access token refresh failed; logging out")
                # refresh also failed → logout
                logout()
                st.rerun()


# UI
if not st.session_state.user:
    st.title("📚 Enterprise RAG Demo")
    st.warning("Please login")
    st.stop()

user = st.session_state.user
roles = user.get("realm_access", {}).get("roles", [])
is_admin = "admin" in roles

st.title("📚 Enterprise RAG Demo")
st.success(f"Logged in as: {user.get('preferred_username')}")

if is_admin:
    st.markdown("""
### 👈 Use the sidebar to navigate:

- **Upload** → Upload & process documents  
- **Chat** → Ask questions
""")
else:
    st.markdown("""
### 👈 Use the sidebar to navigate:

- **Chat** → Ask questions
""")