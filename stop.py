from http.client import responses

import requests, json
from config import *
import api_utils

base_url, headers, user_id = api_utils.get_url_headers(HOST_URL, NGC_API_KEY)

endpoint = f"{base_url}/experiments"
response = requests.get(endpoint, headers=headers)

for exp in json.loads(response.content):
    endpoint_exp = f"{base_url}/experiments/{exp['id']}/jobs"
    response_exp = requests.get(endpoint_exp, headers=headers)
    for job in json.loads(response_exp.content):
        endpoint_job = f"{base_url}/experiments/{exp['id']}/jobs/{job['id']}:cancel"
        response = requests.post(endpoint_job, headers=headers)
        print(f"Job with id: {job['id']} stopped.")
