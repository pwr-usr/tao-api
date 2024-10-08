from exceptiongroup import catch

from config import *
import dataset_utils
import api_utils
import json
import train
import eval_retrain
import os
import export
from fastapi.responses import FileResponse

def train_tao_model():
    try:
        os.remove('/tmp/model.onnx')
    except Exception as e:
        pass

    print(HOST_URL)

    # Log in
    base_url, headers, user_id = api_utils.get_url_headers(HOST_URL, NGC_API_KEY)
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
    print("tar files processed")


    # Upload  dataset
    train_dataset_id, eval_dataset_id, test_dataset_id = dataset_utils.create_and_upload_datasets(
        base_url, headers, train_dataset_path, eval_dataset_path, test_dataset_path, model_name="TODO")
    print("Dataset_uploaded", train_dataset_id, eval_dataset_id, test_dataset_id)

    # Create experiment and associate datasets
    experiment_id = api_utils.create_experiment_and_associate_datasets(
        base_url, headers, train_dataset_id, eval_dataset_id, test_dataset_id)
    print(f"Experiment created, Experiment id: {experiment_id}")


    # Get AutoML specs if enabled
    if AUTOML_ENABLED:
        print("AudoML is enabled")
        automl_specs = api_utils.get_automl_specs(base_url, headers, experiment_id)
        print("The following is AutoML specs:", json.dumps(automl_specs, sort_keys=True, indent=4))

    # Train model
    job_map = {}
    response = train.set_automl_params(base_url, headers, experiment_id)
    train_specs = train.set_train_specs(base_url, headers, experiment_id, class_names)
    job_map = train.training_run(base_url, headers, experiment_id, train_specs, job_map)
    print("Training Done", job_map)

    # Evaluate
    job_map = eval_retrain.evaluate(base_url, headers, experiment_id, class_names, job_map)
    print("Eval Done", job_map)

    # Prune
    job_map = eval_retrain.prune(base_url, headers, experiment_id, class_names, job_map)
    print("Pruning Done", job_map)

    # Retrain
    job_map = eval_retrain.retrain(base_url, headers, experiment_id, class_names, job_map)
    print("Retrain Done", job_map)

    # Evaluate after retrain
    job_map = eval_retrain.evaluate_after_retrain(base_url, headers, experiment_id, job_map)
    print("Re-evaluation Done", job_map)


    # Export model
    job_map = export.run_export(base_url, headers, experiment_id, job_map)

    print(f"All work done, User Id: {user_id}\nExperiment Id: {experiment_id}\n Job map: {job_map}\n ")

    # Download model to loca
    remote_path = f'/mnt/nfs_share/default-nvtl-api-pvc/users/{user_id}/experiments/{experiment_id}/{job_map["export_" + MODEL_NAME]}/model.onnx'
    command = f"sshpass -p '111' scp {USERNAME}@{HOST}:{remote_path} /tmp"
    os.system(command)
    print(f"File downloaded to local")

    return FileResponse(path='/tmp/model.onnx', filename="model.onnx")
