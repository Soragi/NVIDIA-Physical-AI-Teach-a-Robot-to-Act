"""Idempotent instructor preparation on the existing pinned Isaac deployment."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from workshop.lab.training import pause_nemotron, TrainingLab
ARENA=ROOT/'third_party/IsaacLab-Arena'
PYTHON=ARENA/'submodules/Isaac-GR00T/.venv/bin/python'

def main():
    for route in ('state','training/state'):
        try:
            with urllib.request.urlopen('http://127.0.0.1:8097/mission-api/'+route,timeout=3) as response:
                state=json.load(response)
                if state.get('busy') or state.get('restore_nemotron'):raise SystemExit('Finish or release the active training experiment before preparation.')
        except OSError:pass
    if not PYTHON.exists():raise SystemExit('Install the pinned Isaac profile first: see ISAAC_DEPLOYMENT.md.')
    if shutil.disk_usage(ROOT).free<45*1024**3:raise SystemExit('Preparation needs at least 45 GiB free.')
    lock=ROOT/'workshop/results/isaac/.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        subprocess.run([str(PYTHON),str(ROOT/'workshop/scripts/prepare_gr00t_training.py')],check=True)
        early=ROOT/'workshop/results/training/pilot-200'
        complete=early/'progress.json'
        if complete.exists() and json.loads(complete.read_text()).get('phase')=='complete' and (early/'checkpoint-200/config.json').exists():
            print('Prepared early checkpoint already exists; retained unchanged.')
            return
        if early.exists():raise SystemExit('Incomplete early checkpoint exists. Inspect it before retrying; it will not be silently reused.')
        check=subprocess.run(['docker','inspect','-f','{{.State.Running}}','nvidia-nemotron-nano-9b-v2'],capture_output=True,text=True,check=True)
        restore=check.stdout.strip()=='true'
        try:
            if restore:pause_nemotron()
            subprocess.run([str(PYTHON),str(ROOT/'workshop/scripts/gr00t_train_worker.py'),
                '--base',str(ROOT/'third_party/training-inputs/base-model'),'--output',str(early),
                '--steps','200','--seed','42'],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),check=True)
        finally:
            if restore and not TrainingLab._restore_service():raise RuntimeError('Restore Nemotron before returning the instance to attendees.')
    print('Training inputs prepared. Validate simulator outcomes before switching the attendee entry point.')

if __name__=='__main__':main()
