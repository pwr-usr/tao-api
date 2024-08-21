import json
import requests





def login(ngc_api_key: str, host_url: str):
    data = json.dumps({"ngc_api_key": ngc_api_key})
    response = requests.post(f"{host_url}/api/v1/login", data=data)
    assert response.status_code in (200, 201)
    assert "user_id" in response.json().keys()
    user_id = response.json()["user_id"]
    print("User ID", user_id)
    assert "token" in response.json().keys()
    token = response.json()["token"]
    print("JWT", token)

    # Set base URL
    base_url = f"{host_url}/api/v1/users/{user_id}"
    print("API Calls will be forwarded to", base_url)

    headers = {"Authorization": f"Bearer {token}"}