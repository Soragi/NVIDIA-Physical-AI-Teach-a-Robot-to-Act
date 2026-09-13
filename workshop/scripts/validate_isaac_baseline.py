"""Execute NVIDIA's unmodified GR1 evaluation before activating a new clone."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
ARENA=ROOT/'third_party/IsaacLab-Arena'
PRIVATE=Path.home()/'.local/state/vss-robotics-workshop'


def ready():
    try:
        with socket.create_connection(('127.0.0.1',15556),timeout=1): return True
    except OSError: return False


def main():
    PRIVATE.mkdir(parents=True,exist_ok=True,mode=0o700)
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
            env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',CUDA_VISIBLE_DEVICES='1')
            subprocess.run([str(ARENA/'.venv/bin/python'),'isaaclab_arena/evaluation/policy_runner.py',
                '--policy_type','isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy',
                '--policy_config_yaml_path','isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml',
                '--remote_host','127.0.0.1','--remote_port','15556','--num_episodes','3','--seed','7',
                '--enable_cameras','--record_camera_video','--output_base_dir',str(ROOT/'workshop/results/isaac-baseline'),
                'put_item_in_fridge_and_close_door','--embodiment','gr1_joint','--object','ranch_dressing_hope_robolab'],
                cwd=ARENA,env=env,check=True,timeout=900)
    finally:
        if policy is not None and policy.poll() is None:
            policy.terminate()
            policy.wait(timeout=30)


if __name__=='__main__': main()
