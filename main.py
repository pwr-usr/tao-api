import os
import json
from config import *
from datasets.dataset_utils import prepare_dataset, split_tar_file
from api_utils import get_user_token
from model_actions import *


def main():
    # Setup
    user_id, token = get_user_token(HOST_URL, NGC_API_KEY)
    base_url = f"{HOST_URL}/api/v1/users/{user_id}"
    headers = {"Authorization": f"Bearer {token}"}

    # Prepare dataset
    prepare_dataset(DATA_DIR, f"{DATA_DIR}/split/images_train", f"{DATA_DIR}/split/images_val",
                    f"{DATA_DIR}/split/images_test")

    # Create tar files
    split_tar_dir = os.path.join(os.path.dirname(DATA_DIR), 'split_tar')
    os.makedirs(split_tar_dir, exist_ok=True)
    os.system(f'tar -C {DATA_DIR}/split/ -czf {split_tar_dir}/classification_train.tar.gz images_train classes.txt')
    os.system(f'tar -C {DATA_DIR}/split/ -czf {split_tar_dir}/classification_val.tar.gz images_val classes.txt')
    os.system(f'tar -C {DATA_DIR}/split/ -czf {split_tar_dir}/classification_test.tar.gz images_test classes.txt')

    # Create and upload datasets
    train_dataset_id = create_dataset(base_url, headers, "image_classification", DS_FORMAT)
    eval_dataset_id = create_dataset(base_url, headers, "image_classification", DS_FORMAT)
    test_dataset_id = create_dataset(base_url, headers, "image_classification", DS_FORMAT)

    update_dataset(base_url, headers, train_dataset_id, "Siemens Train Dataset", "My train dataset")
    update_dataset(base_url, headers, eval_dataset_id, "Siemens Eval dataset", "S eval dataset")

    for dataset_id, dataset_path in [
        (train_dataset_id, f"{split_tar_dir}/classification_train.tar.gz"),
        (eval_dataset_id, f"{split_tar_dir}/classification_val.tar.gz"),
        (test_dataset_id, f"{split_tar_dir}/classification_test.tar.gz")
    ]:
        output_dir = os.path.join(os.path.dirname(dataset_path), MODEL_NAME, dataset_id)
        split_tar_file(dataset_path, output_dir)
        for tar_file in os.listdir(output_dir):
            upload_dataset(base_url, headers, dataset_id, os.path.join(output_dir, tar_file))

    # Create experiment
    experiment_id = create_experiment(base_url, headers, MODEL_NAME, ENCODE_KEY, CHECKPOINT_CHOOSE_METHOD)

    # Assign datasets
    assign_datasets(base_url, headers, experiment_id, train_dataset_id, eval_dataset_id, test_dataset_id)

    # Set AutoML config if enabled
    if AUTOML_ENABLED:
        automl_config = {
            "automl_enabled": AUTOML_ENABLED,
            "automl_algorithm": AUTOML_ALGORITHM,
            "metric": "kpi",
            "automl_max_recommendations": 20,
            "automl_R": 15,
            "automl_nu": 4,
            "epoch_multiplier": 0.3,
            "override_automl_disabled_params": False,
            "automl_add_hyperparameters": "[]",
            "automl_remove_hyperparameters": "[]"
        }
        set_automl_config(base_url, headers, experiment_id, automl_config)

    # Run actions
    actions = ["train", "evaluate", "export", "gen_trt_engine", "inference"]
    for action in actions:
        specs = get_default_specs(base_url, headers, experiment_id, action)

        if action == "train":
            specs["dataset"]["num_classes"] = len(os.listdir(f"{DATA_DIR}/split/images_train"))
            specs["train"]["num_epochs"] = 80
            specs["gpus"] = 1
        elif action == "export":
            pass  # No changes needed for export specs
        elif action == "gen_trt_engine":
            specs["gen_trt_engine"]["tensorrt"]["data_type"] = "int8"

        parent_job_id = JOB_MAP.get(f"{'train' if action != 'train' else ''}_{MODEL_NAME}")
        job_id = run_action(base_url, headers, experiment_id, parent_job_id, action, specs)
        JOB_MAP[f"{action}_{MODEL_NAME}"] = job_id

        status = monitor_job(base_url, headers, experiment_id, job_id)
        print(f"{action.capitalize()} job completed with status: {status}")

        if DOWNLOAD_JOBS and action in ["train", "inference", "inference_trt"]:
            download_job(base_url, headers, experiment_id, job_id, WORKDIR)

    # Clean up
    delete_experiment(base_url, headers, experiment_id)
    for dataset_id in [train_dataset_id, eval_dataset_id, test_dataset_id]:
        delete_dataset(base_url, headers, dataset_id)


if __name__ == "__main__":
    main()