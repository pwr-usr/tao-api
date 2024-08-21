import json
from api_utils import post_request, get_request, patch_request, delete_request, monitor_job
import time
from IPython.display import clear_output


def create_dataset(base_url, headers, ds_type, ds_format):
    data = json.dumps({"type": ds_type, "format": ds_format})
    response = post_request(f"{base_url}/datasets", data=data, headers=headers)
    return response.json()["id"]


def update_dataset(base_url, headers, dataset_id, name, description):
    data = json.dumps({"name": name, "description": description})
    patch_request(f"{base_url}/datasets/{dataset_id}", data=data, headers=headers)


def upload_dataset(base_url, headers, dataset_id, file_path):
    files = [("file", open(file_path, "rb"))]
    response = post_request(f"{base_url}/datasets/{dataset_id}:upload", files=files, headers=headers)
    assert response.json()["message"] == "Server recieved file and upload process started"


def create_experiment(base_url, headers, model_name, encode_key, checkpoint_choose_method):
    data = json.dumps({
        "network_arch": model_name,
        "encryption_key": encode_key,
        "checkpoint_choose_method": checkpoint_choose_method
    })
    response = post_request(f"{base_url}/experiments", data=data, headers=headers)
    return response.json()["id"]


def assign_datasets(base_url, headers, experiment_id, train_dataset_id, eval_dataset_id, test_dataset_id):
    dataset_information = {
        "train_datasets": [train_dataset_id],
        "eval_dataset": eval_dataset_id,
        "inference_dataset": test_dataset_id,
        "calibration_dataset": train_dataset_id
    }
    data = json.dumps(dataset_information)
    patch_request(f"{base_url}/experiments/{experiment_id}", data=data, headers=headers)


def get_default_specs(base_url, headers, experiment_id, action):
    response = get_request(f"{base_url}/experiments/{experiment_id}/specs/{action}/schema", headers=headers)
    return response.json()["default"]


def run_action(base_url, headers, experiment_id, parent_job_id, action, specs):
    data = json.dumps({"parent_job_id": parent_job_id, "action": action, "specs": specs})
    response = post_request(f"{base_url}/experiments/{experiment_id}/jobs", data=data, headers=headers)
    return response.json()


def delete_experiment(base_url, headers, experiment_id):
    delete_request(f"{base_url}/experiments/{experiment_id}", headers=headers)


def delete_dataset(base_url, headers, dataset_id):
    delete_request(f"{base_url}/datasets/{dataset_id}", headers=headers)


def set_automl_config(base_url, headers, experiment_id, automl_config):
    data = json.dumps(automl_config)
    patch_request(f"{base_url}/experiments/{experiment_id}", data=data, headers=headers)


def update_checkpoint_choosing(base_url, headers, experiment_id, method, epoch_number=None):
    update_data = {"checkpoint_choose_method": method}
    if epoch_number:
        update_data["checkpoint_epoch_number"] = {f"from_epoch_number_{experiment_id}": epoch_number}
    data = json.dumps(update_data)
    patch_request(f"{base_url}/experiments/{experiment_id}", data=data, headers=headers)


def download_job(base_url, headers, experiment_id, job_id, output_path):
    endpoint = f"{base_url}/experiments/{experiment_id}/jobs/{job_id}"
    response = get_request(endpoint, headers=headers)
    expected_file_size = response.json().get("job_tar_stats", {}).get("file_size")

    endpoint = f'{base_url}/experiments/{experiment_id}/jobs/{job_id}:download'
    temptar = f'{job_id}.tar.gz'

    with tqdm(total=expected_file_size, unit='B', unit_scale=True) as progress_bar:
        while True:
            headers_download_job = dict(headers)
            if os.path.exists(temptar):
                file_size = os.path.getsize(temptar)
                print(f"File size of downloaded content until now is {file_size}")

                if file_size >= (expected_file_size - 1):
                    print("Download completed successfully.")
                    print("Untarring")
                    os.system(f'tar -xf {temptar} -C {output_path}/')
                    os.remove(temptar)
                    print(f"Results at {output_path}/{job_id}")
                    return f"{output_path}/{job_id}"

                headers_download_job['Range'] = f'bytes={file_size}-'

            with open(temptar, 'ab') as f:
                try:
                    response = get_request(endpoint, headers=headers_download_job, stream=True)
                    if response.status_code in [200, 206]:
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:
                                f.write(chunk)
                                f.flush()
                                os.fsync(f.fileno())
                            progress_bar.update(len(chunk))
                    else:
                        print(f"Failed to download file. Status code: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print("Connection interrupted during download, resuming download from breaking point")
                    time.sleep(5)
                    continue