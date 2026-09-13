"""Small, inspectable VSS and Cosmos HTTP client. Standard library only."""
import base64
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def request(url, payload=None, *, method=None, content_type='application/json', timeout=300):
    data = json.dumps(payload).encode() if isinstance(payload, dict) else payload
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={'Content-Type': content_type})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # Do not expose response bodies: upstream errors can echo full requests.
        raise RuntimeError(f'HTTP {exc.code} from {urllib.parse.urlsplit(url).path}; check service health.') from None
    except urllib.error.URLError:
        raise RuntimeError('Service unavailable; run the notebook health check before retrying.') from None


def final_text(value):
    text = str(value or '')
    for tag in ('agent-think', 'think'):
        text = re.sub(fr'<{tag}>.*?</{tag}>', '', text, flags=re.S | re.I)
        text = re.sub(fr'<{tag}>.*$', '', text, flags=re.S | re.I)
    text = re.sub(r'</?final_answer>', '', text, flags=re.I)
    if '<answer>' in text:
        text = text.split('<answer>', 1)[1].split('</answer>', 1)[0]
    if not text.strip():
        raise RuntimeError('No final answer returned. Do not count this as a successful inference.')
    return text.strip()


class LabClient:
    def __init__(self, vss='http://127.0.0.1:7777', cosmos='http://127.0.0.1:30082'):
        self.vss, self.cosmos = vss.rstrip('/'), cosmos.rstrip('/')

    def models(self):
        return request(self.cosmos + '/v1/models', timeout=10)

    def upload(self, path):
        path = Path(path)
        data = path.read_bytes()
        # Content-based name avoids overwriting another participant's upload.
        name = re.sub(r'[^A-Za-z0-9._-]', '_', path.stem) + '-' + hashlib.sha256(data).hexdigest()[:10] + '.mp4'
        library = request(self.vss + '/vst/api/v1/sensor/list', timeout=20)
        if not isinstance(library, list):
            raise RuntimeError('Video library did not return a list; check VST health.')
        existing = next((v for v in library if v.get('type') == 'sensor_file'
                         and v.get('name') == name[:-4] and v.get('sensorId')), None)
        if existing:
            return {'name': existing['name'], 'sensor_id': existing['sensorId']}
        result = request(self.vss + '/api/v1/videos-for-search/' + urllib.parse.quote(name),
                         data, method='PUT', content_type='video/mp4')
        if not result.get('sensor_id'):
            raise RuntimeError('Upload did not return a sensor ID; do not continue with an unrelated video.')
        return {'name': name[:-4], 'sensor_id': result['sensor_id']}

    def ask_vss(self, video, question):
        prompt = (f'Selected video: {video["name"]}. VST sensor ID: {video["sensor_id"]}. '
                  'Use the video understanding tool on this video. Provide approximate timestamps, '
                  'separate visible evidence from inference, and state uncertainty.\n' + question)
        start = time.monotonic()
        result = request(self.vss + '/generate', {'input_message': prompt}, timeout=600)
        return {'answer': final_text(result.get('value')), 'latency_s': round(time.monotonic()-start, 2),
                'path': 'vss-agent', 'question': question, 'sensor_id': video['sensor_id']}

    def ask_cosmos(self, path, question, *, frames=16, max_tokens=1536):
        path = Path(path)
        if not 1 <= frames <= 64:
            raise ValueError('For the lab, use 1 to 64 frames.')
        data = path.read_bytes()
        if len(data) > 20 * 1024 * 1024:
            raise ValueError('Use a short analysis clip below 20 MiB for the direct API exercise.')
        models = self.models().get('data', [])
        ids = [m['id'] for m in models if 'cosmos' in m.get('id', '').lower()]
        if len(ids) != 1:
            raise RuntimeError('Expected one served Cosmos model; inspect /v1/models before continuing.')
        payload = {'model': ids[0], 'messages': [{'role': 'user', 'content': [
            {'type': 'video_url', 'video_url': {'url': 'data:video/mp4;base64,' + base64.b64encode(data).decode()}},
            {'type': 'text', 'text': question}]}], 'temperature': 0, 'max_tokens': max_tokens,
            'media_io_kwargs': {'video': {'num_frames': frames}}}
        start = time.monotonic()
        result = request(self.cosmos + '/v1/chat/completions', payload, timeout=600)
        choice = result['choices'][0]
        if choice.get('finish_reason') == 'length':
            raise RuntimeError('Answer reached the token limit; increase max_tokens and repeat before evaluating.')
        return {'answer': final_text(choice['message'].get('content')), 'model': ids[0],
                'latency_s': round(time.monotonic()-start, 2), 'frames': frames,
                'usage': result.get('usage', {}), 'path': 'cosmos-direct', 'question': question,
                'media_sha256': hashlib.sha256(data).hexdigest()}


def save_result(result, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    return destination
