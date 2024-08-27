import os


# Set constants
WORKDIR = "/home/zyw/tao-env/siemens_product_cla"
MODEL_NAME = "classification_tf2"
PRETRAINED_MODEL_MAP = {
    "classification_tf1": "pretrained_classification:resnet18",
    "classification_tf2": "pretrained_classification_tf2:efficientnet_b0"
}
AUTOML_MAX_RECOMMENDATIONS = 20
NUM_EPOCH = 40
RETRAIN_EPOCH = 40






HOST_URL = "http://192.168.1.85:31951"
NGC_API_KEY = "YTVmYWs3aDgxZ2Q2aG5oY3Yyc2RwZG9na2Q6MDFjMmUxMjMtZDNhZC00MWJlLWFmZGMtMGU3ZTc1OThjMGY3"

AUTOML_ENABLED = True
AUTOML_ALGORITHM = "bayesian"
DOWNLOAD_JOBS = True
DATASET_TO_BE_USED = "custom"
DATA_DIR = os.path.join(WORKDIR, MODEL_NAME, "source_data")
os.environ['DATA_DIR'] = DATA_DIR

JOB_MAP = {}
DS_TYPE = "image_classification"
if "classification_" in MODEL_NAME:
    DS_FORMAT = "default"
    ENCODE_KEY = "nvidia_tlt"
# elif MODEL_NAME == "multitask_classification":
#     DS_FORMAT = "custom"
#     ENCODE_KEY = "tlt_encode"
# else:
#     raise ValueError(f"Unsupported model name: {MODEL_NAME}")

PRETRAINED_MODEL_MAP = {
    "classification_tf1": "pretrained_classification:resnet18",
    "classification_tf2": "pretrained_classification_tf2:efficientnet_b0",
    "classification_pyt": "pretrained_fan_classification_imagenet:fan_hybrid_tiny",
    "multitask_classification": "pretrained_classification:resnet10"
}
AUTOML_INFORMATION = {"automl_enabled": AUTOML_ENABLED,
                      "automl_algorithm": AUTOML_ALGORITHM,
                      "metric": "kpi",
                      "automl_max_recommendations": AUTOML_MAX_RECOMMENDATIONS,  # Only for bayesian
                      "automl_R": 15,  # Only for hyperband
                      "automl_nu": 4,  # Only for hyperband
                      "epoch_multiplier": 1,  # Only for hyperband
                      # Enable this if you want to add parameters to automl_add_hyperparameters below that are disabled by TAO in the automl_enabled column of the spec csv.
                      # Warning: The parameters that are disabled are not tested by TAO, so there might be unexpected behaviour in overriding this
                      "override_automl_disabled_params": False,
                      "automl_add_hyperparameters": str([]),
                      "automl_remove_hyperparameters": str([])
                      }


NUM_GPU = 1
NO_PTM_MODELS = set([])
CHECKPOINT_CHOOSE_METHOD = "best_model"