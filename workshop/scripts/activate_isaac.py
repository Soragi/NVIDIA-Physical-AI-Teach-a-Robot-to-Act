"""Activate a baseline-validated Isaac profile; retain the former release for rollback."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
PRIVATE=Path.home()/'.local/state/vss-robotics-workshop'


def run(args):
    subprocess.run([str(a) for a in args],check=True)


def main():
    for route in ('training/state',):
        try:
            with urllib.request.urlopen('http://127.0.0.1:8097/mission-api/'+route,timeout=2) as response:
                state=json.load(response)
                if state['busy'] or state.get('restore_nemotron'):
                    raise SystemExit('Activation blocked while an experiment holds GPU resources. Finish, stop or release it first.')
        except OSError:
            pass
    evidence=[]
    for path in (ROOT/'workshop/results/isaac-baseline').rglob('episode_results_rank0.jsonl'):
        evidence.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    if not any(row.get('success') is True for row in evidence):
        raise SystemExit('Activation blocked: record a successful official Isaac baseline first. See ISAAC_DEPLOYMENT.md.')
    PRIVATE.mkdir(parents=True,exist_ok=True,mode=0o700)
    for name,script in [('physical-ai-gr1','isaac_policy_server.py'),('physical-ai-isaac','isaac_ui_server.py')]:
        text=f'''[Unit]
Description=NVIDIA Physical AI workshop {name}
After=network-online.target

[Service]
User={os.environ.get('USER','ubuntu')}
WorkingDirectory={ROOT}
ExecStart={sys.executable} {ROOT}/workshop/scripts/{script}
Restart=on-failure
RestartSec=5
KillMode=control-group
StandardOutput=append:{PRIVATE}/{name}.log
StandardError=append:{PRIVATE}/{name}.log

[Install]
WantedBy=multi-user.target
'''
        path=PRIVATE/(name+'.service')
        path.write_text(text)
        run(['sudo','install','-m','644',path,'/etc/systemd/system/'+name+'.service'])
    # Stop only the temporary validation services and superseded policy/UI adapters.
    for name in ('physical-ai-gr1-validation','physical-ai-isaac-ui'):
        subprocess.run(['sudo','systemctl','stop',name],check=False)
    run(['sudo','systemctl','daemon-reload'])
    run(['sudo','systemctl','enable','--now','physical-ai-gr1','physical-ai-isaac'])
    run(['sudo','systemctl','restart','physical-ai-gr1','physical-ai-isaac'])
    run(['bash',ROOT/'workshop/scripts/deploy_vss_base.sh','refresh-ui'])
    deadline=time.monotonic()+120
    while True:
        try:
            for url in ('http://127.0.0.1:7777/',
                        'http://127.0.0.1:7777/mission-api/training/state'):
                with urllib.request.urlopen(url,timeout=5) as response:
                    if response.status!=200:raise OSError('Gateway not ready')
            break
        except OSError:
            if time.monotonic()>deadline:
                raise SystemExit('Port 7777 gateway/API is not ready. Inspect physical-ai-isaac and vss-workshop-ui logs.')
            time.sleep(2)
    print('Port 7777 UI and proxied training API are responding. Open this instance\'s Brev Secure Link; complete Setup H–I before running a mission.')
    print('Isaac profile activated. Verify a live browser and notebook mission before cloning.')


if __name__=='__main__':
    main()
