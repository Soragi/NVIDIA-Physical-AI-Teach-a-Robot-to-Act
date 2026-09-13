"""Execute NVIDIA's unmodified GR1 evaluation before activating a new clone."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
ARENA=ROOT/'third_party/IsaacLab-Arena'
PRIVATE=Path.home()/'.local/state/vss-robotics-workshop'


def complete_evidence(folder):
    """Return complete three-episode reports; ignore partial/corrupt attempts."""
    complete=[]
    for path in folder.rglob('episode_results_rank0.jsonl') if folder.exists() else ():
        try:
            rows=[json.loads(line) for line in path.read_text().splitlines() if line]
        except (OSError,json.JSONDecodeError):
            continue
        if len(rows)>=3 and all(type(row.get('success')) is bool for row in rows):
            complete.append((path,rows))
    return complete


def ready():
    try:
        with socket.create_connection(('127.0.0.1',15556),timeout=1): return True
    except OSError: return False


def main():
    PRIVATE.mkdir(parents=True,exist_ok=True,mode=0o700)
    output=ROOT/'workshop/results/isaac-baseline'
    if any(any(row['success'] for row in rows) for _,rows in complete_evidence(output)):
        print('A complete successful three-episode baseline already exists; retained unchanged.')
        return
    policy=None
    try:
        with (PRIVATE/'isaac-baseline-policy.log').open('a') as log:
            if not ready():
                policy=subprocess.Popen([sys.executable,str(ROOT/'workshop/scripts/isaac_policy_server.py')],stdout=log,stderr=subprocess.STDOUT)
                deadline=time.monotonic()+300
                while not ready():
                    if policy.poll() is not None or time.monotonic()>deadline:
                        raise RuntimeError('Policy did not become ready; inspect the private baseline policy log.')
                    time.sleep(1)
            # Jupyter exports its inline Matplotlib backend to child processes.
            # That backend is intentionally absent from the isolated Isaac env.
            env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',
                CUDA_VISIBLE_DEVICES='1',MPLBACKEND='Agg')
            previous={path for path,_ in complete_evidence(output)}
            result=subprocess.run([str(ARENA/'.venv/bin/python'),'isaaclab_arena/evaluation/policy_runner.py',
                '--policy_type','isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy',
                '--policy_config_yaml_path','isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml',
                '--remote_host','127.0.0.1','--remote_port','15556','--num_episodes','3','--seed','7',
                '--enable_cameras','--record_camera_video','--output_base_dir',str(output),
                'put_item_in_fridge_and_close_door','--embodiment','gr1_joint','--object','ranch_dressing_hope_robolab'],
                cwd=ARENA,env=env,check=False,timeout=900)
            completed=[(path,rows) for path,rows in complete_evidence(output) if path not in previous]
            if not completed:
                result.check_returncode()
                raise RuntimeError('Isaac exited without a complete three-episode baseline report.')
            if result.returncode:
                if result.returncode == -signal.SIGSEGV:
                    print('Baseline evidence is complete; ignoring Isaac application shutdown SIGSEGV.')
                else:
                    result.check_returncode()
    finally:
        if policy is not None and policy.poll() is None:
            policy.terminate()
            policy.wait(timeout=30)


if __name__=='__main__': main()
