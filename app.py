import streamlit as st
import requests
from jose import jwt
from dotenv import load_dotenv

from rag.logger import logger

load_dotenv(override=True)

st.set_page_config(page_title="Enterprise RAG Demo")


# ==============================
# ✅ AUTH CONFIG
# ==============================
KEYCLOAK_URL = "http://localhost:8080"
REALM = "rag_application"
CLIENT_ID = "backend-api"
CLIENT_SECRET = "Hdi2qBsE8OsFV4GV8LfxriI7fs7BUM9k"

TOKEN_URL = f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/token"
KEYCLOAK_ISSUER = f"{KEYCLOAK_URL}/realms/{REALM}"
JWKS_URL = f"{KEYCLOAK_ISSUER}/protocol/openid-connect/certs"
LOGOUT_URL = f"{KEYCLOAK_ISSUER}/protocol/openid-connect/logout"


# ==============================
# ✅ TOKEN HELPERS
# ==============================
@st.cache_resource
def load_jwks():
    return requests.get(JWKS_URL, timeout=10).json()


jwks = load_jwks()


def verify_token(token):
    header = jwt.get_unverified_header(token)
    key = next(k for k in jwks["keys"] if k["kid"] == header["kid"])
    return jwt.decode(
        token,
        key,
        algorithms=["RS256"],
        options={"verify_aud": False},
        issuer=KEYCLOAK_ISSUER,
    )


def token_request(payload):
    return requests.post(
        TOKEN_URL,
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
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
        user = verify_token(tokens["access_token"])

        st.session_state.user = user
        st.session_state.access_token = tokens["access_token"]
        st.session_state.refresh_token = tokens.get("refresh_token")

        st.sidebar.success(f"✅ {user.get('preferred_username')}")
        st.rerun()
    else:
        st.sidebar.error("❌ Login failed")


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

user = st.session_state.user
roles = user.get("realm_access", {}).get("roles", [])
is_admin = "admin" in roles

st.title("📚 Enterprise RAG Demo")

st.success(f"Logged in as: {user.get('preferred_username')}")
st.info(f"Role: {'Admin' if is_admin else 'User'}")


# ==============================
# ✅ INSTRUCTIONS ONLY
# ==============================
if is_admin:
    st.markdown(
        """
## 👈 Use the sidebar to navigate:

### Admin capabilities:
- 📤 **Upload Page**
  - Upload documents (PDF/Markdown)
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
