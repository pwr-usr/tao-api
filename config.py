import os

MODEL_NAME = "classification_tf2"
WORKDIR = "/home/zyw/tao-env/siemens_product_cla"
HOST_URL = "http://192.168.1.85:31951"
NGC_API_KEY = "YTVmYWs3aDgxZ2Q2aG5oY3Yyc2RwZG9na2Q6MDFjMmUxMjMtZDNhZC00MWJlLWFmZGMtMGU3ZTc1OThjMGY3"

AUTOML_ENABLED = True
AUTOML_ALGORITHM = "bayesian"
DOWNLOAD_JOBS = True

DATASET_TO_BE_USED = "custom"
DATA_DIR = os.path.join(WORKDIR, MODEL_NAME, "source_data")
os.environ['DATA_DIR'] = DATA_DIR

JOB_MAP = {}

if "classification_" in MODEL_NAME:
    DS_FORMAT = "default"
    ENCODE_KEY = "nvidia_tlt"
elif MODEL_NAME == "multitask_classification":
    DS_FORMAT = "custom"
    ENCODE_KEY = "tlt_encode"
else:
    raise ValueError(f"Unsupported model name: {MODEL_NAME}")

CHECKPOINT_CHOOSE_METHOD = "best_model"