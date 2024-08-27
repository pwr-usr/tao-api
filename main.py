from config import *
import dataset_utils
import api_utils
import json
import train
import eval_retrain
import os

def main():
    print(HOST_URL)

    # Log in
    base_url, headers = api_utils.get_url_headers(HOST_URL, NGC_API_KEY)
    print("login successful")
    print("Work dir:", WORKDIR)

    # Split and move files
    class_names, train_dir, val_dir, test_dir = dataset_utils.make_dir_get_classes(DATA_DIR)
    dataset_utils.print_source_distribution(DATA_DIR, class_names)
    dataset_utils.prepare_dataset(class_names, DATA_DIR, train_dir, val_dir, test_dir)

    # Create tar files
    main_data_dir = os.path.join(os.path.dirname(DATA_DIR))
    print("The zipped file should be in a folder called split_tar in main_data_dir, if not existing, create it")
    train_dataset_path, eval_dataset_path, test_dataset_path = dataset_utils.process_tar(main_data_dir)

    # Upload  dataset
    train_dataset_id, eval_dataset_id, test_dataset_id = dataset_utils.create_and_upload_datasets(
        base_url, headers, train_dataset_path, eval_dataset_path, test_dataset_path, model_name="TODO")

    # Create experiment and associate datasets
    experiment_id = api_utils.create_experiment_and_associate_datasets(
        base_url, headers, train_dataset_id, eval_dataset_id, test_dataset_id)

    # Get AutoML specs if enabled
    if AUTOML_ENABLED:
        automl_specs = api_utils.get_automl_specs(base_url, headers, experiment_id)
        print(json.dumps(automl_specs, sort_keys=True, indent=4))

    # Train model
    job_map = {}
    response = train.set_automl_params(base_url, headers, experiment_id)
    train_specs = train.set_train_specs(base_url, headers, experiment_id, class_names)
    job_map = train.training_run(base_url, headers, experiment_id, train_specs, job_map)

    # Evaluate
    job_map = eval_retrain.evaluate(base_url, headers, experiment_id, class_names, job_map)

    # Prune
    job_map = eval_retrain.prune(base_url, headers, experiment_id, class_names, job_map)

    # Retrain
    job_map = eval_retrain.retrain(base_url, headers, experiment_id, class_names, job_map)

    # Evaluate after retrain
    job_map = eval_retrain.evaluate_after_retrain(base_url, headers, experiment_id, job_map)










    # Clean up
    # delete_experiment(base_url, headers, experiment_id)
    # for dataset_id in [train_dataset_id, eval_dataset_id, test_dataset_id]:
    #     delete_dataset(base_url, headers, dataset_id)


if __name__ == "__main__":
    main()