import requests
import json
from config import DS_TYPE, DS_FORMAT, MODEL_NAME, ENCODE_KEY, CHECKPOINT_CHOOSE_METHOD, PRETRAINED_MODEL_MAP, NO_PTM_MODELS
from functools import wraps
import time
from IPython.display import clear_output


def get_url_headers(host_url, ngc_api_key):
    # Exchange NGC_API_KEY for JWT
    data = json.dumps({"ngc_api_key": ngc_api_key})
    print(data)
    response = requests.post(f"{host_url}/api/v1/login", data=data)
    print("Response Code:", response.status_code)
    print("Response JSON:", response.json())

    user_id = response.json()["user_id"]
    token = response.json()["token"]
    # Set base URL
    base_url = f"{host_url}/api/v1/users/{user_id}"
    print("API Calls will be forwarded to", base_url)

    headers = {"Authorization": f"Bearer {token}"}
    return base_url, headers


def create_dataset(base_url, headers):
    data = json.dumps({"type": DS_TYPE, "format": DS_FORMAT})
    endpoint = f"{base_url}/datasets"
    response = requests.post(endpoint, data=data, headers=headers)
    assert response.status_code in (200, 201)
    return response.json()["id"]

def update_dataset(base_url, headers, dataset_id, name, description):
    dataset_information = {"name": name, "description": description}
    data = json.dumps(dataset_information)
    endpoint = f"{base_url}/datasets/{dataset_id}"
    response = requests.patch(endpoint, data=data, headers=headers)
    assert response.status_code in (200, 201)

def upload_dataset(base_url, headers, dataset_id, dataset_path):
    with open(dataset_path, "rb") as f:
        files = [("file", f)]
        endpoint = f"{base_url}/datasets/{dataset_id}:upload"
        response = requests.post(endpoint, files=files, headers=headers)
        assert response.status_code in (200, 201)
        assert "message" in response.json().keys() and response.json()["message"] == "Server received file and upload process started"

# def upload_dataset(base_url, headers, dataset_id, dataset_path):
#     files = [("file", open(dataset_path, "rb"))]  # This line is now enclosed in a 'with' statement
#     endpoint = f"{base_url}/datasets/{dataset_id}:upload"
#     response = requests.post(endpoint, files=files, headers=headers)
#     assert response.status_code in (200, 201)
#     assert "message" in response.json().keys() and response.json()["message"] == "Server received file and upload process started"

def create_experiment_and_associate_datasets(base_url, headers, train_dataset_id, eval_dataset_id, test_dataset_id):
    # Create experiment
    data = json.dumps({
        "network_arch": MODEL_NAME,
        "encryption_key": ENCODE_KEY,
        "checkpoint_choose_method": CHECKPOINT_CHOOSE_METHOD
    })
    endpoint = f"{base_url}/experiments"
    response = requests.post(endpoint, data=data, headers=headers)
    assert response.status_code in (200, 201)
    experiment_id = response.json()["id"]

    # List experiments
    params = {"network_arch": MODEL_NAME}
    response = requests.get(endpoint, params=params, headers=headers)
    assert response.status_code in (200, 201)
    print("model id\t\t\t     network architecture")
    for rsp in response.json():
        print(rsp["name"], rsp["id"], rsp["network_arch"])

    # Associate datasets with experiment
    dataset_information = {
        "train_datasets": [train_dataset_id],
        "eval_dataset": eval_dataset_id,
        "inference_dataset": test_dataset_id,
        "calibration_dataset": train_dataset_id
    }
    data = json.dumps(dataset_information)
    endpoint = f"{base_url}/experiments/{experiment_id}"
    response = requests.patch(endpoint, data=data, headers=headers)
    assert response.status_code in (200, 201)

    # Get pretrained model for classification
    if MODEL_NAME not in NO_PTM_MODELS:
        params = {"network_arch": MODEL_NAME}
        response = requests.get(f"{base_url}/experiments", params=params, headers=headers)
        assert response.status_code in (200, 201)

        ptm_id = None
        for rsp in response.json():
            if rsp["ngc_path"].endswith(PRETRAINED_MODEL_MAP[MODEL_NAME]):
                ptm_id = rsp["id"]
                break

        if ptm_id:
            ptm_information = {"base_experiment": [ptm_id]}
            data = json.dumps(ptm_information)
            response = requests.patch(f"{base_url}/experiments/{experiment_id}", data=data, headers=headers)
            assert response.status_code in (200, 201)

    # Record information in JSON file
    info = {
        "experiment_id": experiment_id,
        "train_dataset_id": train_dataset_id,
        "eval_dataset_id": eval_dataset_id,
        "test_dataset_id": test_dataset_id,
        "model_name": MODEL_NAME,
        "pretrained_model_id": ptm_id if MODEL_NAME not in NO_PTM_MODELS else None
    }

    with open('experiment_info.json', 'w') as f:
        json.dump(info, f, indent=4)

    return experiment_id


def get_automl_specs(base_url, headers, experiment_id):
    endpoint = f"{base_url}/experiments/{experiment_id}/specs/train/schema"
    response = requests.get(endpoint, headers=headers)
    assert response.status_code in (200, 201)
    assert "automl_default_parameters" in response.json().keys()
    return response.json()["automl_default_parameters"]