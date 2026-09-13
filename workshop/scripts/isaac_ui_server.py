"""Local-only HTTP bridge for the shared Isaac learning workspace."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from workshop.lab.evidence import EvidenceEngine
from workshop.lab.training import TrainingLab

engine = EvidenceEngine(ROOT)
training = TrainingLab(engine)
token = secrets.token_urlsafe(32)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):
        pass

    def reply(self,status,value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json')
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split('?',1)[0]
        if path in ('/mission-api/training/state','/mission-api/training/report'):
            return self.reply(200,dict(training.status(),csrf=token))
        match = re.fullmatch(r'/mission-api/training/media/([a-f0-9]{12})/(frame.jpg|episode.mp4|report.json)',path)
        if path=='/mission-api/training/demonstration':
            file=ROOT/'third_party/training-inputs/dataset/ranch_bottle_into_fridge/ranch_bottle_into_fridge_generated_100/lerobot/videos/chunk-000/observation.images.ego_view/episode_000000.mp4'
        elif match:
            file=training.media(match[1],match[2])
        else:
            return self.reply(404,{'error':'Unknown endpoint.'})
        if file is None or not file.is_file():
            return self.reply(404,{'error':'Artifact is not available yet.'})
        size = file.stat().st_size
        start,end,status = 0,size-1,200
        header = self.headers.get('Range','')
        if header:
            byte_range = re.fullmatch(r'bytes=(\d+)-(\d*)',header)
            if not byte_range:
                return self.reply(416,{'error':'Unsupported range.'})
            start = int(byte_range[1])
            end = min(int(byte_range[2]) if byte_range[2] else end,end)
            if not 0<=start<=end<size:
                return self.reply(416,{'error':'Invalid range.'})
            status = 206
        self.send_response(status)
        self.send_header('Content-Type',{'.mp4':'video/mp4','.jpg':'image/jpeg','.json':'application/json'}[file.suffix])
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(end-start+1))
        self.send_header('Accept-Ranges','bytes')
        if status==206:
            self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        try:
            with file.open('rb') as stream:
                stream.seek(start)
                remaining = end-start+1
                while remaining:
                    chunk = stream.read(min(65536,remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError,ConnectionResetError):
            pass

    def do_POST(self):
        if not secrets.compare_digest(self.headers.get('X-Mission-Token',''),token):
            return self.reply(403,{'error':'Refresh the page before submitting an action.'})
        try:
            length = int(self.headers.get('Content-Length','0'))
            if not 0<length<=16384:
                raise ValueError('Invalid request length.')
            data = json.loads(self.rfile.read(length))
            if not isinstance(data,dict):
                raise ValueError('Expected a JSON object.')
            if self.path=='/mission-api/training/create':
                result=training.create(data)
            elif self.path=='/mission-api/training/stop':
                result=training.stop()
            elif self.path in ('/mission-api/training/before','/mission-api/training/train','/mission-api/training/after'):
                result=training.start(self.path.rsplit('/',1)[1],data)
            else:
                return self.reply(404,{'error':'Unknown action.'})
            self.reply(202,result)
        except (ValueError,KeyError) as error:
            self.reply(409,{'error':str(error)})


if __name__=='__main__':
    ThreadingHTTPServer(('127.0.0.1',8097),Handler).serve_forever()
