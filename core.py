import json
import requests
from main import HOST_URL


def create_tao_dataset(ds_type: str, ds_format: str):
    data = json.dumps({"type": ds_type, "format": ds_format})

    # TODO: remove hard code
    endpoint = f"{HOST_URL}/datasets"

    response = requests.post(endpoint, data=data, headers=headers)
    assert response.status_code in (200, 201)

    print(response)
    print(response.json())
    assert "id" in response.json().keys()
    return response.json()["id"]