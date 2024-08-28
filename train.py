from config import AUTOML_ENABLED, AUTOML_INFORMATION, MODEL_NAME, NUM_EPOCH, NUM_GPU
import json, requests
import time
from IPython.display import clear_output

def set_automl_params(base_url, headers, experiment_id):
    if AUTOML_ENABLED:
        # Choose any metric that is present in the kpi dictionary present in the model's status.json.
        # Example status.json for each model can be found in the respective section in NVIDIA TAO DOCS here: https://docs.nvidia.com/tao/tao-toolkit/text/model_zoo/cv_models/index.html

        data = json.dumps(AUTOML_INFORMATION)

        endpoint = f"{base_url}/experiments/{experiment_id}"

        response = requests.patch(endpoint, data=data, headers=headers)
        assert response.status_code in (200, 201)

        print(json.dumps(response.json(), sort_keys=True, indent=4))
        return response

def set_train_specs(base_url, headers, experiment_id, class_names):
    # Get default spec schema
    endpoint = f"{base_url}/experiments/{experiment_id}/specs/train/schema"

    response = requests.get(endpoint, headers=headers)
    assert response.status_code in (200, 201)

    # print(response)
    # print(response.json()) ## Uncomment for verbose schema
    train_specs = response.json()["default"]
    if MODEL_NAME == "classification_tf1":
        train_specs["train_config"]["n_epochs"] = NUM_EPOCH
    elif MODEL_NAME == "classification_tf2":
        train_specs["train"]["num_epochs"] = NUM_EPOCH
    train_specs["dataset"]["num_classes"] = len(class_names)
    train_specs["gpus"] = NUM_GPU
    # print(json.dumps(train_specs, sort_keys=True, indent=4))
    return train_specs

def training_run(base_url, headers, experiment_id, train_specs, job_map):
    parent = None
    action = "train"
    data = json.dumps({"parent_job_id": parent, "action": action, "specs": train_specs})

    endpoint = f"{base_url}/experiments/{experiment_id}/jobs"

    response = requests.post(endpoint, data=data, headers=headers)
    assert response.status_code in (200, 201)
    assert response.json()

    # print(response)
    # print(response.json())

    job_map["train_" + MODEL_NAME] = response.json()
    return job_map

def training_monitor(base_url, headers, experiment_id, job_map, ):
    job_id = job_map["train_" + MODEL_NAME]
    endpoint = f"{base_url}/experiments/{experiment_id}/jobs/{job_id}"

    while True:
        clear_output(wait=True)
        response = requests.get(endpoint, headers=headers)

        if "error_desc" in response.json().keys() and response.json()["error_desc"] in (
        "Job trying to retrieve not found", "No AutoML run found"):
            print("Job is being created")
            time.sleep(5)
            continue
        # assert response.status_code in (200, 201)
        print(response)
        print(json.dumps(response.json(), sort_keys=True, indent=4))
        assert "status" in response.json().keys() and response.json().get("status") != "Error"
        if response.json().get("status") in ["Done", "Error", "Canceled"] or response.status_code not in (200, 201):
            break
        time.sleep(5)


