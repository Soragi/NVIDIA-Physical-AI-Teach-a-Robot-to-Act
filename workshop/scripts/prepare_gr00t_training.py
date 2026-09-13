"""Download pinned NVIDIA training inputs without changing running services."""
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
token = Path.home()/'.local/state/vss-robotics-workshop/hf.token'
if token.exists():
    os.environ['HF_TOKEN'] = token.read_text().strip()
api = HfApi(token=os.environ.get('HF_TOKEN'))
data_id = 'nvidia/Arena-GR1-Manipulation-PlaceItemCloseDoor-Task'
model_id = 'nvidia/GR00T-N1.6-3B'
data = api.dataset_info(data_id, revision='2b8461bf9f45e2c9087902c6eaa823083a2e849b')
model = api.model_info(model_id, revision='d0814e7ecb19202e7c8468b46098b0b7ef3a6d61')
target = ROOT/'third_party/training-inputs'
target.mkdir(parents=True, exist_ok=True)
manifest = {'dataset':data_id,'dataset_revision':data.sha,'base_model':model_id,
            'base_model_revision':model.sha,'training_performed':False}
(target/'sources.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest),flush=True)
snapshot_download(data_id,repo_type='dataset',revision=data.sha,
    allow_patterns=['ranch_bottle_into_fridge/ranch_bottle_into_fridge_generated_100/lerobot/*','README.md'],
    local_dir=target/'dataset')
snapshot_download(model_id,revision=model.sha,
    ignore_patterns=['*.pt','*.bin','*optimizer*','*global_step*'],local_dir=target/'base-model')
print('Training inputs downloaded. No policy switched.',flush=True)
