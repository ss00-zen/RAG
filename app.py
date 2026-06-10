import os
import requests
import streamlit as st
from dotenv import load_dotenv

from rag.logger import logger
from auth.auth import verify_token, get_auth_provider
from auth.auth_context import normalize_user, is_admin, role_from_auth

load_dotenv(override=True)

st.set_page_config(page_title="Enterprise RAG Demo")

AUTH_PROVIDER = get_auth_provider()


# ==============================
# ✅ KEYCLOAK CONFIG (only used when AUTH_PROVIDER=keycloak)
# ==============================
KEYCLOAK_BASE_URL = os.getenv("KEYCLOAK_BASE_URL", "http://localhost:8080").rstrip("/")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "rag_application")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "backend-api")
KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET", "")

TOKEN_URL = f"{KEYCLOAK_BASE_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token"


# ==============================
# ✅ HELPERS
# ==============================
def token_request(payload):
    return requests.post(
        TOKEN_URL,
        data={
            "client_id": KEYCLOAK_CLIENT_ID,
            "client_secret": KEYCLOAK_CLIENT_SECRET,
            **payload,
        },
        timeout=15,
    )


def refresh_access_token(refresh_token):
    res = token_request({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token
    })
    if res.status_code == 200:
        data = res.json()
        return data["access_token"], data.get("refresh_token")
    return None, None


def logout():
    st.session_state.clear()


def save_authenticated_session(access_token: str, refresh_token: str = None):
    claims = verify_token(access_token)
    st.session_state.user = claims
    st.session_state.access_token = access_token
    st.session_state.refresh_token = refresh_token


# ==============================
# ✅ SESSION
# ==============================
for key in ["user", "access_token", "refresh_token"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ==============================
# ✅ LOGIN UI
# ==============================
st.sidebar.title("🔐 Login")
st.sidebar.caption(f"Provider: {AUTH_PROVIDER}")

if AUTH_PROVIDER == "keycloak":
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        res = token_request({
            "grant_type": "password",
            "username": username.strip(),
            "password": password.strip(),
        })

        if res.status_code == 200:
            tokens = res.json()
            save_authenticated_session(
                access_token=tokens["access_token"],
                refresh_token=tokens.get("refresh_token"),
            )
            st.sidebar.success("✅ Login successful")
            st.rerun()
        else:
            logger.error("Keycloak login failed: %s", res.text)
            st.sidebar.error("❌ Login failed")

else:
    st.sidebar.info("Paste a valid access/session token for this provider.")
    raw_token = st.sidebar.text_area("JWT token", height=180)

    if st.sidebar.button("Use Token"):
        try:
            save_authenticated_session(access_token=raw_token.strip())
            st.sidebar.success("✅ Token accepted")
            st.rerun()
        except Exception as e:
            logger.exception("Token login failed")
            st.sidebar.error(f"❌ Token verification failed: {str(e)}")


# ==============================
# ✅ LOGOUT
# ==============================
if st.session_state.user:
    if st.sidebar.button("🚪 Logout"):
        logout()
        st.rerun()


# ==============================
# ✅ MAIN LANDING PAGE
# ==============================
if not st.session_state.user:
    st.title("📚 Enterprise RAG Demo")
    st.warning("Please login")
    st.stop()

auth_ctx = normalize_user(
    st.session_state.user,
    provider=AUTH_PROVIDER
)

current_role = role_from_auth(auth_ctx)
admin = is_admin(auth_ctx)

st.title("📚 Enterprise RAG Demo")
st.success(f"Logged in as: {auth_ctx.username or auth_ctx.user_id}")
st.info(f"Role: {'Admin' if admin else 'User'}")
st.caption(f"Provider: {AUTH_PROVIDER}")


# ==============================
# ✅ INSTRUCTIONS ONLY
# ==============================
if admin:
    st.markdown(
        """
## 👈 Use the sidebar to navigate:

### Admin capabilities:
- 📤 **Upload Page**
  - Upload documents
  - Mark documents as classified

- 💬 **Chat Page**
  - Ask questions from your knowledge base
  - Access classified + public data

---
"""
    )
else:
    st.markdown(
        """
## 👈 Use the sidebar to navigate:

### User capabilities:
- 💬 **Chat Page**
  - Ask questions from the knowledge base
  - Only non-classified data available

---
"""
    )

st.warning("⚠️ Please use the sidebar to switch between pages.")