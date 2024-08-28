import requests, os
from config import *

"root@aes:/mnt/nfs_share/default-nvtl-api-pvc/users/a1c02cba-b62b-52f9-9e49-e3de0e5b66ab/experiments/d862a1b9-1305-4ac9-9d1f-a9e94e2b9360/841a5e14-c5a1-4267-be6c-a595b5e35f8a"

user_id = "a1c02cba-b62b-52f9-9e49-e3de0e5b66ab"
experiment_id = "d862a1b9-1305-4ac9-9d1f-a9e94e2b9360"
job_id = "841a5e14-c5a1-4267-be6c-a595b5e35f8a"
job_map = {}
job_map["export_" + MODEL_NAME] = job_id
remote_path = f'/mnt/nfs_share/default-nvtl-api-pvc/users/{user_id}/experiments/{experiment_id}/{job_map["export_" + MODEL_NAME]}/model.onnx'
command = f"sshpass -p '111' scp {USERNAME}@{HOST}:{remote_path} {WORKDIR}"
os.system(command)