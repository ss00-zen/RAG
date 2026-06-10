import requests

SECRET_KEY = "sk_test_IHTPXyVB7u69WwRTuBUI5Lws1IIR1rlWa5lsi3ToPl"   # your Clerk secret key
TEMPLATE_NAME = "temp_jwt"
USER_ID = "user_3Ew9rM68g93QA7wdLiKRUKMmTnc"   # admin user

response = requests.post(
    f"https://api.clerk.com/v1/jwt_templates/{TEMPLATE_NAME}/tokens",
    headers={
        "Authorization": f"Bearer {SECRET_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "user_id": USER_ID
    }
)

print("STATUS:", response.status_code)
print("RESPONSE:", response.text)
