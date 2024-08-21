import requests
import json
from functools import wraps
import time
from IPython.display import clear_output

def check_response(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        response = func(*args, **kwargs)
        assert response.status_code in (200, 201), f"API call failed with status code {response.status_code}"
        return response
    return wrapper

@check_response
def post_request(url, data=None, files=None, headers=None):
    return requests.post(url, data=data, files=files, headers=headers)

@check_response
def get_request(url, headers=None, params=None):
    return requests.get(url, headers=headers, params=params)

@check_response
def patch_request(url, data, headers):
    return requests.patch(url, data=data, headers=headers)

@check_response
def delete_request(url, headers):
    return requests.delete(url, headers=headers)

def get_user_token(base_url, ngc_api_key):
    data = json.dumps({"ngc_api_key": ngc_api_key})
    response = post_request(f"{base_url}/api/v1/login", data=data)
    assert "user_id" in response.json().keys()
    assert "token" in response.json().keys()
    return response.json()["user_id"], response.json()["token"]

def monitor_job(base_url, headers, experiment_id, job_id):
    endpoint = f"{base_url}/experiments/{experiment_id}/jobs/{job_id}"
    while True:
        clear_output(wait=True)
        response = get_request(endpoint, headers=headers)
        if "error_desc" in response.json().keys() and response.json()["error_desc"] in ("Job trying to retrieve not found", "No AutoML run found"):
            print("Job is being created")
            time.sleep(5)
            continue
        print(response)
        print(json.dumps(response.json(), sort_keys=True, indent=4))
        assert "status" in response.json().keys() and response.json().get("status") != "Error"
        if response.json().get("status") in ["Done", "Error", "Canceled"] or response.status_code not in (200, 201):
            break
        time.sleep(5)
    return response.json().get("status")