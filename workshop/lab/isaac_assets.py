"""Use verified local copies of the official robot, bottle and kitchen assets."""
import hashlib
import json


def apply_pinned_assets(root, definition):
    manifest=root/'workshop/runtime/isaac-assets.json'
    cache=root/'third_party/isaac-assets'
    data=json.loads(manifest.read_text())
    for entry in data['files'].values():
        path=cache/entry['file']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise RuntimeError('Simulator asset missing or changed: '+entry['file'])
    definition.embodiment.scene_config.robot.spawn.usd_path=str(cache/data['roots']['robot'])
    for name,asset in definition.scene.assets.items():
        if name=='ranch_dressing_hope_robolab': asset.usd_path=str(cache/data['roots']['bottle'])
        if name.startswith('lightwheel_kitchen'): asset.usd_path=str(cache/data['roots']['kitchen'])
    return hashlib.sha256(manifest.read_bytes()).hexdigest()
