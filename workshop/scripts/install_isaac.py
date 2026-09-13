"""Prepare the pinned Isaac profile without changing the attendee entry point."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ARENA_REV = '3032c7888b7b56e54d907e5c46e98c6dfdfb7dba'
POLICY_REV = 'e29d8fc50b0e4745120ae3fb72447986fe638aa6'
MODEL_REV = 'f5e388c35340080644e17a3b3d538150ad894bfa'
MODEL = 'nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task'
ARENA = ROOT / 'third_party/IsaacLab-Arena'
PRIVATE = Path.home() / '.local/state/vss-robotics-workshop'


def run(args, cwd=None):
    subprocess.run([str(a) for a in args], cwd=cwd, check=True)


def main():
    PRIVATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.environ['GIT_LFS_SKIP_SMUDGE'] = '1'
    uv = shutil.which('uv') or str(Path(sys.executable).parent / 'uv')
    run(['sudo', 'apt-get', 'update', '-qq'])
    run(['sudo', 'apt-get', 'install', '-y', 'libxt6', 'libxrender1', 'libxrandr2',
         'libxi6', 'libxinerama1', 'libxcursor1', 'libglu1-mesa', 'ffmpeg'])
    if not ARENA.exists():
        run(['git', 'clone', '--depth', '1', '--branch', 'release/0.3.0',
             'https://github.com/isaac-sim/IsaacLab-Arena.git', ARENA])
        run(['git','fetch','--depth','1','origin',ARENA_REV],ARENA)
        run(['git','checkout','--detach',ARENA_REV],ARENA)
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ARENA,text=True).strip()
    if revision != ARENA_REV:
        raise SystemExit('Arena revision differs from the workshop pin; preserve this checkout and use a clean directory.')
    for name, url in [('IsaacLab','https://github.com/isaac-sim/IsaacLab.git'),
                      ('Isaac-GR00T','https://github.com/NVIDIA/Isaac-GR00T.git')]:
        run(['git','config',f'submodule.submodules/{name}.url',url],ARENA)
    run(['git','submodule','update','--init','--depth','1'],ARENA)
    run([uv,'sync','--frozen','--no-default-groups','--group','isaaclab-from-wheel','--group','gr00t-client'],ARENA)
    policy = ARENA / 'submodules/Isaac-GR00T'
    lock = ROOT / 'workshop/runtime/isaac-policy.uv.lock'
    if lock.exists():
        shutil.copy2(lock,policy/'uv.lock')
    run([uv,'sync','--python','3.10',*(['--frozen'] if lock.exists() else [])],policy)
    # The upstream 2.7.1 default is CUDA 12.6 and lacks sm_120 kernels.
    run([uv,'pip','install','--python',policy/'.venv/bin/python',
         '--index-url','https://download.pytorch.org/whl/cu128',
         'torch==2.7.1+cu128','torchvision==0.22.1+cu128'])
    run([uv,'pip','install','--python',policy/'.venv/bin/python',
         '--no-build-isolation','flash-attn==2.8.3'])
    token = PRIVATE / 'hf.token'
    env = dict(os.environ)
    if token.exists():
        env['HF_TOKEN'] = token.read_text().strip()
    code = ('from huggingface_hub import snapshot_download; '
            f'snapshot_download({MODEL!r}, revision={MODEL_REV!r}, '
            'allow_patterns=["ranch_bottle_into_fridge/*"], '
            'ignore_patterns=["*/global_step*/*","*.pt"], '
            f'local_dir={str(ROOT/"third_party/arena-model")!r})')
    subprocess.run([str(policy/'.venv/bin/python'),'-c',code],env=env,check=True)
    run([ARENA/'.venv/bin/python',ROOT/'workshop/scripts/prepare_isaac_assets.py'])
    (PRIVATE/'isaac-install.json').write_text(json.dumps({'arena':ARENA_REV,'policy':POLICY_REV,'model':MODEL_REV},indent=2))
    print('Isaac profile prepared. Run baseline validation before activating the attendee UI.')


if __name__ == '__main__':
    main()
