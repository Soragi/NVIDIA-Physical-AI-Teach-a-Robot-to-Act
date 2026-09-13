"""Serve the pretrained Arena GR1 policy in its isolated Python 3.10 runtime."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
arena = ROOT / 'third_party/IsaacLab-Arena'
repo = arena / 'submodules/Isaac-GR00T'
token = Path.home() / '.local/state/vss-robotics-workshop/hf.token'
if token.exists():
    os.environ['HF_TOKEN'] = token.read_text().strip()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_MODE'] = 'disabled'
os.chdir(repo)
python = str(repo / '.venv/bin/python')
os.execv(python, [python, 'gr00t/eval/run_gr00t_server.py',
    '--model-path', str(ROOT / 'third_party/arena-model/ranch_bottle_into_fridge'),
    '--embodiment-tag', 'GR1', '--device', 'cuda',
    '--host', '127.0.0.1', '--port', '15556'])
