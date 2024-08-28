import requests, json

from config import MODEL_NAME

def edit_export_schema(base_url, headers, experiment_id):
    endpoint = f"{base_url}/experiments/{experiment_id}/specs/export/schema"

    response = requests.get(endpoint, headers=headers)
    assert response.status_code in (200, 201)

    print(response)
    # print(response.json()) ## Uncomment for verbose schema
    assert "default" in response.json().keys()
    export_specs = response.json()["default"]
    print(json.dumps(export_specs, sort_keys=True, indent=4))
    return export_specs

def run_export(base_url, headers, experiment_id, job_map):
    export_specs = edit_export_schema(base_url, headers, experiment_id)
    parent = job_map["train_" + MODEL_NAME]
    action = "export"
    data = json.dumps({"parent_job_id": parent, "action": action, "specs": export_specs})
    endpoint = f"{base_url}/experiments/{experiment_id}/jobs"
    response = requests.post(endpoint, data=data, headers=headers)
    assert response.status_code in (200, 201)
    assert response.json()
    print("Export Status Code:", response.status_code)
    job_map["export_" + MODEL_NAME] = response.json()
    return job_map