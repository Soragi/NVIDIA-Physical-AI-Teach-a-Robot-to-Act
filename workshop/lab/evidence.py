"""Minimal evidence adapter required by the GR00T training experiment."""
import threading
import time
from pathlib import Path

from .client import LabClient, request


class EvidenceEngine:
    def __init__(self,root):
        self.root=Path(root)
        self.lock=threading.RLock()
        self.state={'busy':False}
        self.client=LabClient()

    def _evidence(self,path):
        sensor=self.client.upload(path)
        for _ in range(60):
            request(self.client.vss+'/vst/api/v1/sensor/list',timeout=10)
            try:
                timeline=request(self.client.vss+'/vst/api/v1/sensor/'+sensor['sensor_id']+'/timelines',timeout=10)
                if timeline:return {'sensor':sensor,'timeline':timeline}
            except RuntimeError as error:
                if 'HTTP 404' not in str(error):raise
            time.sleep(1)
        raise RuntimeError('VSS timeline not ready; evidence has not been substituted.')
