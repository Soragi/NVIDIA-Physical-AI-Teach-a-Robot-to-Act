"""Cache and content-pin the official task assets, using NVIDIA's dependency parser.

Run with Arena's Python. --record is a maintainer operation; normal preparation
must match the checked-in manifest and fails if upstream content has changed.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import urllib.request
import urllib.error
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'third_party/isaac-assets'
MANIFEST=ROOT/'workshop/runtime/isaac-assets.json'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--record',action='store_true')
    args=parser.parse_args()
    source=next((ROOT/'third_party/IsaacLab-Arena/.venv/lib').glob('python*/site-packages/isaaclab/source/isaaclab/isaaclab/utils/assets.py'))
    spec=importlib.util.spec_from_file_location('nvidia_asset_helpers',source)
    helpers=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    roots={
        'robot':helpers.ISAAC_NUCLEUS_DIR+'/Robots/FourierIntelligence/GR-1/GR1T2_fourier_hand_6dof/GR1T2_fourier_hand_6dof.usd',
        'bottle':helpers.ISAACLAB_NUCLEUS_DIR.replace('omniverse-content-production','omniverse-content-staging')+'/Arena/assets/object_library/srl_robolab_assets/objects/hope/ranch_dressing.usd',
    }
    records={}
    queue=list(roots.values())
    expanded=set()
    while queue:
        url=queue.pop()
        if url in records: continue
        if '<UDIM>' in url:
            if url in expanded: continue
            expanded.add(url)
            for tile in range(1001,1101):
                candidate=url.replace('<UDIM>',str(tile))
                try:
                    with urllib.request.urlopen(urllib.request.Request(candidate,method='HEAD'),timeout=30): pass
                except urllib.error.HTTPError as error:
                    if error.code==404: break
                    raise
                queue.append(candidate)
            continue
        parsed=urlparse(url)
        if parsed.scheme!='https': raise RuntimeError('Unexpected asset protocol: '+url)
        path=CACHE/parsed.netloc/parsed.path.lstrip('/')
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            print('Downloading '+url,flush=True)
            with urllib.request.urlopen(url,timeout=60) as response, path.open('wb') as stream:
                shutil.copyfileobj(response,stream)
        records[url]={'file':str(path.relative_to(CACHE)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        for ref in helpers._find_asset_dependencies(str(path)):
            if ref in ('OmniPBR.mdl','OmniSurface.mdl','OmniGlass.mdl'):
                continue  # Built-in materials are pinned by the Isaac Sim runtime.
            resolved=helpers._resolve_reference_url(url,ref)
            if resolved and resolved not in records: queue.append(resolved)
    kitchen=Path.home()/'.cache/lightwheel_sdk/floorplan/robocasa-robocasakitchen-1-2'
    if not (kitchen/'scene.usd').exists():
        from lightwheel_sdk.loader import floorplan_loader
        floorplan_loader.get_usd(scene='robocasakitchen',layout_id=1,style_id=2,backend='robocasa')
    for source in sorted(kitchen.rglob('*')):
        if source.is_file():
            relative=source.relative_to(kitchen)
            path=CACHE/'kitchen'/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,path)
            records['kitchen/'+str(relative)]={'file':str(path.relative_to(CACHE)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    value={'schema_version':1,'roots':{name:records[url]['file'] for name,url in roots.items()},'files':records}
    value['roots']['kitchen']='kitchen/scene.usd'
    if args.record:
        MANIFEST.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    elif not MANIFEST.exists() or json.loads(MANIFEST.read_text())!=value:
        raise SystemExit('Asset content differs from the workshop pin. Do not activate this preparation.')
    print(f'Verified {len(records)} asset files. Manifest SHA256: '+hashlib.sha256(MANIFEST.read_bytes()).hexdigest())


if __name__=='__main__': main()
