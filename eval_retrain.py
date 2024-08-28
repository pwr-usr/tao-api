import requests, json, time
from config import MODEL_NAME, RETRAIN_EPOCH, NUM_GPU

def evaluate(base_url, headers, experiment_id, class_names, job_map):
    # Get model handler parameters
    endpoint = f"{base_url}/experiments/{experiment_id}"
    print(f"Evaluate get model handler endpoint {endpoint}")
    response = requests.get(endpoint, headers=headers)
    assert response.status_code in (200, 201)
    assert response.json()

    print(f"Evaluate get model handler response {response.json()}")

    model_parameters = response.json()
    update_checkpoint_choosing = {}
    update_checkpoint_choosing["checkpoint_epoch_number"] = model_parameters["checkpoint_epoch_number"]
    # Change the method by which checkpoint from the parent action is chosen, when parent action is a train/retrain action.
    # Example for evaluate action below, can be applied in the same way for other actions too
    update_checkpoint_choosing[
        "checkpoint_choose_method"] = "best_model"  # Choose between best_model/latest_model/from_epoch_number
    # If from_epoch_number is chosen then assign the epoch number to the dictionary key in the format 'from_epoch_number{train_job_id}'
    # update_checkpoint_choosing["checkpoint_epoch_number"]["from_epoch_number_28a2754e-50ef-43a8-9733-98913776dd90"] = 3
    data = json.dumps(update_checkpoint_choosing)

    print(f"Evaluate patch data: {data}")
    response = requests.patch(endpoint, data=data, headers=headers)
    assert response.status_code in (200, 201)
    # Get default spec schema
    endpoint = f"{base_url}/experiments/{experiment_id}/specs/evaluate/schema"

    response = requests.get(endpoint, headers=headers)
    assert response.status_code in (200, 201)

    assert "default" in response.json().keys()
    eval_specs = response.json()["default"]
    eval_specs["dataset"]["num_classes"] = len(class_names)
    # Run action
    parent = job_map["train_" + MODEL_NAME]
    action = "evaluate"
    data = json.dumps({"parent_job_id": parent, "action": action, "specs": eval_specs})
    print("Updated Specs", data)
    endpoint = f"{base_url}/experiments/{experiment_id}/jobs"
    response = requests.post(endpoint, data=data, headers=headers)
    assert response.status_code in (200, 201)
    assert response.json()
    print("Evaluate done")

    job_map["evaluate_" + MODEL_NAME] = response.json()

    # Monitor job status by repeatedly running this cell
    job_id = job_map["evaluate_" + MODEL_NAME]
    endpoint = f"{base_url}/experiments/{experiment_id}/jobs/{job_id}"

    while True:
        response = requests.get(endpoint, headers=headers)
        assert response.status_code in (200, 201)
        # print(response)
        # print(response.json())
        print("Eval status: ", response.json().get("status"))
        assert "status" in response.json().keys() and response.json().get("status") != "Error"
        if response.json().get("status") in ["Done", "Error", "Canceled"] or response.status_code not in (200, 201):
            break
        time.sleep(15)
    return job_map

def prune(base_url, headers, experiment_id, class_names, job_map):
    # Get default spec schema
    endpoint = f"{base_url}/experiments/{experiment_id}/specs/prune/schema"
    response = requests.get(endpoint, headers=headers)
    assert response.status_code in (200, 201)
    assert "default" in response.json().keys()
    prune_specs = response.json()["default"]
    prune_specs["dataset"]["num_classes"] = len(class_names)
    if MODEL_NAME == "classification_tf2":
        prune_specs["prune"]["byom_model_path"] = ""
    parent = job_map["train_" + MODEL_NAME]
    action = "prune"
    data = json.dumps({"parent_job_id":parent,"action":action,"specs":prune_specs})

    endpoint = f"{base_url}/experiments/{experiment_id}/jobs"

    response = requests.post(endpoint, data=data, headers=headers)
    assert response.status_code in (200, 201)
    assert response.json()
    job_map["prune_" + MODEL_NAME] = response.json()

    # Monitor job status by repeatedly running this cell (prune)
    job_id = job_map["prune_" + MODEL_NAME]
    endpoint = f"{base_url}/experiments/{experiment_id}/jobs/{job_id}"

    while True:
        response = requests.get(endpoint, headers=headers)
        assert response.status_code in (200, 201)
        # print(response)
        # print(response.json())
        assert "status" in response.json().keys() and response.json().get("status") != "Error"
        print("Prune status: ", response.json().get("status"))
        if response.json().get("status") in ["Done","Error", "Canceled"] or response.status_code not in (200,201):
            break
        time.sleep(15)

    return job_map

def retrain(base_url, headers, experiment_id, class_names, job_map):
    endpoint = f"{base_url}/experiments/{experiment_id}/specs/retrain/schema"

    response = requests.get(endpoint, headers=headers)
    assert response.status_code in (200, 201)
    #print(response.json()) ## Uncomment for verbose schema
    retrain_specs = response.json()["default"]
    if MODEL_NAME == "classification_tf1":
        retrain_specs["train_config"]["n_epochs"] = RETRAIN_EPOCH
        retrain_specs["gpus"] = NUM_GPU
    # Example for classification_tf2
    elif MODEL_NAME == "classification_tf2":
        retrain_specs["train"]["num_epochs"] = RETRAIN_EPOCH
        retrain_specs["gpus"] = NUM_GPU
    retrain_specs["dataset"]["num_classes"] = len(class_names)

    # Run retrain
    parent = job_map["prune_" + MODEL_NAME]
    action = "retrain"
    data = json.dumps({"parent_job_id":parent,"action":action,"specs":retrain_specs})
    endpoint = f"{base_url}/experiments/{experiment_id}/jobs"
    response = requests.post(endpoint, data=data, headers=headers)
    assert response.status_code in (200, 201)
    assert response.json()
    job_map["retrain_" + MODEL_NAME] = response.json()


    # Monitor job status by repeatedly running this cell (retrain)
    job_id = job_map["retrain_" + MODEL_NAME]
    endpoint = f"{base_url}/experiments/{experiment_id}/jobs/{job_id}"

    while True:
        response = requests.get(endpoint, headers=headers)
        assert response.status_code in (200, 201)
        # print(response)
        # print(response.json())
        assert "status" in response.json().keys() and response.json().get("status") != "Error"
        print("Retrain status: ", response.json().get("status"))
        if response.json().get("status") in ["Done","Error", "Canceled"] or response.status_code not in (200,201):
            break
        time.sleep(15)
    return job_map

def evaluate_after_retrain(base_url, headers, experiment_id, job_map):
    endpoint = f"{base_url}/experiments/{experiment_id}/specs/evaluate/schema"
    response = requests.get(endpoint, headers=headers)
    assert response.status_code in (200, 201)
    assert "default" in response.json().keys()
    eval_retrain_specs = response.json()["default"]
    eval_retrain_specs["dataset"]["num_classes"] = 5
    eval_retrain_specs["evaluate"]["top_k"] = 1
    # print(json.dumps(eval_retrain_specs, sort_keys=True, indent=4))

    # Run actions
    parent = job_map["retrain_" + MODEL_NAME]
    action = "evaluate"
    data = json.dumps({"parent_job_id":parent,"action":action,"specs":eval_retrain_specs})

    endpoint = f"{base_url}/experiments/{experiment_id}/jobs"

    response = requests.post(endpoint, data=data, headers=headers)
    assert response.status_code in (200, 201)
    assert response.json()
    job_map["eval_retrain_" + MODEL_NAME] = response.json()


    # Monitor job status by repeatedly running this cell (evaluate)
    job_id = job_map["eval_retrain_" + MODEL_NAME]
    endpoint = f"{base_url}/experiments/{experiment_id}/jobs/{job_id}"

    while True:
        response = requests.get(endpoint, headers=headers)
        assert response.status_code in (200, 201)
        # print(response)
        # print(response.json())
        assert "status" in response.json().keys() and response.json().get("status") != "Error"
        print("Evaluate after retrain status", response.json().get("status"))
        if response.json().get("status") in ["Done","Error", "Canceled"] or response.status_code not in (200,201):
            break
        time.sleep(15)
    return job_map