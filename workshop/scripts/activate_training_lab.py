"""Switch the attendee entry point only after real local training/evaluation."""
import json
from pathlib import Path
import shutil
import urllib.request

ROOT=Path(__file__).resolve().parents[2]

def main():
    profile=json.loads((ROOT/'workshop/training_profile.json').read_text())
    if profile['validation_status']!='validated':raise SystemExit('Training release validation is not complete.')
    for route in ('training/state',):
        with urllib.request.urlopen('http://127.0.0.1:8097/mission-api/'+route,timeout=5) as response:
            state=json.load(response)
            if state['busy'] or state.get('restore_nemotron'):raise SystemExit('Finish or release the active operation before switching the entry point.')
    reports=[]
    for path in (ROOT/'workshop/results/training').glob('*/report.json'):
        value=json.loads(path.read_text());comparison=value.get('comparison',{})
        if value.get('phase')=='complete' and comparison.get('weights_changed') and comparison.get('same_initial_state') and comparison.get('equal_control_budget'):
            reports.append(path)
    if not reports:raise SystemExit('Execute the complete training notebook locally before activation.')
    ui=ROOT/'workshop/runtime/services/ui/workshop'
    shutil.copy2(ui/'learn.html',ui/'index.html')
    # Start Here includes fresh-environment deployment; never overwrite it with
    # the shorter prepared-instance training notebook.
    # Prepared clones should open an empty experiment, not an instructor's video.
    request=urllib.request.Request('http://127.0.0.1:8097/mission-api/training/create',b'{}',
        headers={'Content-Type':'application/json','X-Mission-Token':state['csrf']},method='POST')
    with urllib.request.urlopen(request,timeout=10) as response:json.load(response)
    print('Training lab is the port 7777 entry point.')

if __name__=='__main__':main()
