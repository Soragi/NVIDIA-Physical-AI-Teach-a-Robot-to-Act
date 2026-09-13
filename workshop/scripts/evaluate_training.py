"""Evaluate an actual checkpoint in fresh Isaac episodes; never substitute video."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from workshop.lab.isaac_contract import scenario, atomic_json
ARENA=ROOT/'third_party/IsaacLab-Arena'
POLICY=ARENA/'submodules/Isaac-GR00T'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',default='101,202,303')
    args=p.parse_args()
    if (args.output/'evaluation.json').exists():raise ValueError('Evaluation output already contains recorded results; use a new directory.')
    # Refuse to mistake a stale policy server for the checkpoint requested here.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        try:probe.bind(('127.0.0.1',15557))
        except OSError as error:raise RuntimeError('Evaluation policy port is occupied; no checkpoint substituted.') from error
    args.output.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',WANDB_MODE='disabled',
        TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg')
    token=Path.home()/'.local/state/vss-robotics-workshop/hf.token'
    if token.exists(): env['HF_TOKEN']=token.read_text().strip()
    with (args.output/'policy.log').open('w') as log:
        server=subprocess.Popen([str(POLICY/'.venv/bin/python'),'gr00t/eval/run_gr00t_server.py',
            '--model-path',str(args.checkpoint),'--embodiment-tag','GR1','--device','cuda',
            '--host','127.0.0.1','--port','15557'],cwd=POLICY,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(300):
                if server.poll() is not None: raise RuntimeError('Policy failed; inspect policy.log')
                try:
                    with socket.create_connection(('127.0.0.1',15557),timeout=1): break
                except OSError: time.sleep(1)
            else: raise TimeoutError('Policy readiness timed out')
            results=[]
            for seed in map(int,args.seeds.split(',')):
                run_id=uuid.uuid4().hex[:12]
                folder=args.output/run_id;folder.mkdir()
                atomic_json(folder/'job.json',{'run_id':run_id,'scenario':scenario(seed),'mode':'baseline',
                    'policy_port':15557,'checkpoint':str(args.checkpoint),'policy_name':'GR00T N1.6 training evaluation'})
                with (folder/'worker.log').open('w') as workerlog:
                    process=subprocess.run([str(ARENA/'.venv/bin/python'),str(ROOT/'workshop/scripts/isaac_worker.py'),
                        '--job',str(folder/'job.json')],stdout=workerlog,stderr=subprocess.STDOUT,timeout=900)
                if (folder/'result.json').exists(): result=json.loads((folder/'result.json').read_text())
                else: result={'seed':seed,'success':False,'error':'No simulator result','returncode':process.returncode,'run_id':run_id}
                results.append(result)
                atomic_json(args.output/'evaluation.json',{'checkpoint':str(args.checkpoint),'results':results})
                print(json.dumps({'seed':seed,'success':result.get('success'),'steps':result.get('episode_steps'),'error':result.get('error')}),flush=True)
        finally:
            server.terminate()
            try: server.wait(timeout=30)
            except subprocess.TimeoutExpired: server.kill();server.wait()

if __name__=='__main__': main()
