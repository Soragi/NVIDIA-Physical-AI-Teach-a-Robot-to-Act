"""Validated scenario and intervention contract; no simulator dependencies."""
import hashlib
import json
import math
from pathlib import Path
import random
import secrets

ACTIONS = ('continue', 'reobserve', 'reobserve_short', 'stop_for_review')
DEFAULT_QUESTION = ('Describe visible bottle movement and grasp progress over time. '
    'Has the target moved relative to the approaching hand? State what the images support '
    'and what remains uncertain. Do not assume a disturbance happened.')
DEFAULT_POLICY = ('Apply these ordered rules; the first matching rule wins. '
    '1. If the video report says no contact, no grasp, or a hand approaching an ungrasped bottle, choose reobserve_short. '
    'This rule applies even when the bottle appears stationary and approach looks normal. '
    '2. If the report supports a secure grasp and normal progress, choose continue. '
    '3. If visibility is insufficient, choose stop_for_review. '
    'Explain which visible finding triggered the selected rule.')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def scenario(seed=None):
    if seed is None:
        seed = secrets.randbelow(2**31)
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError('Seed must be an integer from 0 to 2147483647, or omitted.')
    rng = random.Random(seed)
    return {'seed': seed, 'initial_offset_m': [rng.uniform(-.015,.015),rng.uniform(-.005,.005)],
            'disturbance_m': [rng.choice([-1,1])*rng.uniform(.03,.04),0.0],
            'disturbance_step': 24, 'step_budget': 500, 'normal_chunk':16, 'short_chunk':4}


def validate_decision(value, binding):
    if not isinstance(value, dict) or set(value) != {'action','reason','uncertainty'}:
        raise ValueError('Decision must contain action, reason and uncertainty only.')
    if value['action'] not in ACTIONS:
        raise ValueError('Unknown intervention. No action executed.')
    if any(not isinstance(x,str) or not x.strip() or len(x)>4000 for x in value.values()):
        raise ValueError('Decision fields must be nonempty bounded strings.')
    return {**value, **binding}


def approve(decision, submitted):
    keys = ('run_id','scene_revision','evidence_sha256','decision_id')
    if submitted.get('approve') is not True or any(submitted.get(k) != decision[k] for k in keys):
        raise ValueError('Approval is stale or does not match the current evidence and decision.')
    return decision['action']


def visual_assessment(answer, success):
    """Parse only explicit model verdicts; never infer success from loose prose."""
    text=answer.strip()
    if text in ('complete','incomplete','uncertain'):
        value={'verdict':text,'evidence':answer,'uncertainty':'No separate explanation returned.'}
    else:
        try:
            value=json.loads(text.removeprefix('```json').removeprefix('```').removesuffix('```').strip())
            if not isinstance(value,dict) or value.get('verdict') not in ('complete','incomplete','uncertain'):
                raise ValueError('No explicit verdict')
        except (ValueError,TypeError):
            value={'verdict':'unparsed','evidence':answer}
    disagreement=None if value['verdict'] in ('uncertain','unparsed') else (value['verdict']=='complete') != success
    return value,disagreement
