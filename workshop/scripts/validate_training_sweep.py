"""Record every seed for two fixed checkpoints with shared GPU exclusion."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from workshop.lab.training import pause_nemotron, TrainingLab

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',default='101,202,303,404,505,606,707,808,909,1010')
    args=p.parse_args()
    for route in ('state','training/state'):
        with urllib.request.urlopen('http://127.0.0.1:8097/mission-api/'+route,timeout=5) as response:
            state=json.load(response)
            if state['busy'] or state.get('restore_nemotron'):raise SystemExit('Finish or release the active experiment before validation.')
    with (ROOT/'workshop/results/isaac/.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=subprocess.run(['docker','inspect','-f','{{.State.Running}}','nvidia-nemotron-nano-9b-v2'],capture_output=True,text=True,check=True).stdout.strip()=='true'
        try:
            if active:pause_nemotron()
            for stage,checkpoint in [('before',args.before),('after',args.after)]:
                print('Evaluating',stage,str(checkpoint),flush=True)
                child=subprocess.Popen(['python3',str(ROOT/'workshop/scripts/evaluate_training.py'),
                    '--checkpoint',str(checkpoint),'--output',str(args.output/stage),'--seeds',args.seeds],start_new_session=True)
                try:
                    if child.wait(timeout=20000):raise RuntimeError(stage+' evaluation failed')
                except BaseException:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                    raise
        finally:
            if active and not TrainingLab._restore_service():raise RuntimeError('Restore Nemotron before returning the instance to attendees.')

if __name__=='__main__':main()
