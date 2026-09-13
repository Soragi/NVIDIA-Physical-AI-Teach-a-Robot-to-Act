"""Run the pinned deployment using the key entered in Jupyter, without printing it."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
state = Path(os.getenv('XDG_STATE_HOME', str(Path.home()/'.local/state'))) / 'vss-robotics-workshop'
key_file = state/'ngc.key'
command = sys.argv[1] if len(sys.argv) > 1 else 'status'
if command not in ('check', 'deploy', 'status', 'stop'):
    raise SystemExit('Expected check, deploy, status, or stop')
env = os.environ.copy()
if command == 'deploy':
    key = env.get('NGC_API_KEY', '')
    if not key and key_file.exists():
        key = key_file.read_text().strip()
    if not key:
        private = state/'base.env'
        if private.exists():
            # The generated env contains a blank template value followed by the
            # machine-local value. Match Compose's last-assignment behavior.
            for line in private.read_text().splitlines():
                if line.startswith('NGC_CLI_API_KEY='):
                    key = line.split('=', 1)[1]
    if not key:
        raise SystemExit('Enter your NVIDIA API key in the notebook setup cell first. Do not paste it into chat.')
    env['NGC_API_KEY'] = key
result = subprocess.run(['bash', str(root/'workshop/scripts/deploy_vss_base.sh'), command], env=env)
if result.returncode == 0 and command == 'deploy':
    key_file.unlink(missing_ok=True)
raise SystemExit(result.returncode)
