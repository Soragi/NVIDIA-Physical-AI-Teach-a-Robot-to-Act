"""Real GR00T training and equal-condition checkpoint evaluation."""
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import threading
import time
import uuid

from .isaac_contract import atomic_json, scenario


def pause_nemotron(stop_event=None,on_wait=None):
    """Wait for NIM startup before requesting bounded container shutdown."""
    deadline=time.monotonic()+300
    while True:
        health=subprocess.run(['docker','inspect','-f',
            '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}',
            'nvidia-nemotron-nano-9b-v2'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
        if health!='starting':break
        if stop_event is not None and stop_event.is_set():raise RuntimeError('Stopped before GPU allocation.')
        if time.monotonic()>deadline:raise TimeoutError('Nemotron initialization did not finish; no robot or training process started.')
        if on_wait:on_wait('Waiting for Nemotron initialization before reserving GPU 0.')
        time.sleep(2)
    subprocess.run(['docker','stop','--time','10','nvidia-nemotron-nano-9b-v2'],capture_output=True,check=True,timeout=45)


class TrainingMission:
    """Notebook client for the same experiment used by the browser."""
    def __init__(self,url='http://127.0.0.1:8097'):
        self.url=url.rstrip('/')

    def status(self):
        from .client import request
        return request(self.url+'/mission-api/training/state')

    def _post(self,route,payload):
        import urllib.request, urllib.error
        token=self.status()['csrf']
        req=urllib.request.Request(self.url+'/mission-api/'+route,json.dumps(payload).encode(),
            headers={'Content-Type':'application/json','X-Mission-Token':token},method='POST')
        try:
            with urllib.request.urlopen(req,timeout=30) as response:return json.load(response)
        except urllib.error.HTTPError as error:
            # This local API returns controlled validation messages, not model prompts.
            if error.code in (403,409):
                try: message=json.load(error).get('error','Request rejected.')
                except (ValueError,AttributeError): message='Request rejected.'
                raise RuntimeError(message) from None
            raise RuntimeError(f'Mission API HTTP {error.code}; inspect service health.') from None

    def create(self,seed=None):return self._post('training/create',{'seed':seed})
    def before(self):return self._post('training/before',{})
    def train(self,steps=2000):return self._post('training/train',{'steps':steps})
    def after(self):return self._post('training/after',{})
    def stop(self):return self._post('training/stop',{})
    def report(self):return self.status()

    def wait(self,timeout=3600):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            value=self.status()
            if not value['busy']:
                if value['phase'] in ('error','interrupted','stopped'):
                    raise RuntimeError(value.get('error') or value['phase'])
                return value
            time.sleep(2)
        raise TimeoutError('Operation still active. Inspect status or stop explicitly.')


class TrainingLab:
    def __init__(self, engine):
        self.engine=engine
        self.root=engine.root
        self.results=self.root/'workshop/results/training'
        self.results.mkdir(parents=True,exist_ok=True)
        self.manifest=self.root/'workshop/training_profile.json'
        self.journal=self.results/'active.json'
        self.state={'phase':'idle','busy':False,'session_id':None,'error':None}
        if self.journal.exists():
            self.state=json.loads(self.journal.read_text())
            if self.state.get('busy'):
                self.state.update(phase='interrupted',busy=False,error='Service interrupted. No successful training or evaluation inferred.')
            # Release any surviving experiment reservation after an API restart.
            if self.state.get('restore_nemotron'):
                self.state['restore_nemotron']=not self._restore_service()
            self._save()
        self.process=None
        self.stop_event=threading.Event()
        engine.external_busy=lambda:self.state.get('busy',False) or self.state.get('restore_nemotron',False)

    def _save(self):
        atomic_json(self.journal,self.state)

    @staticmethod
    def _restore_service():
        try:
            return subprocess.run(['docker','start','nvidia-nemotron-nano-9b-v2'],capture_output=True,timeout=180).returncode==0
        except (OSError,subprocess.SubprocessError):return False

    def _pause_service(self):
        def update(message):
            with self.engine.lock:self.state['gpu_status']=message
        pause_nemotron(self.stop_event,update)

    def profile(self):
        if not self.manifest.exists():
            raise ValueError('Training preparation is incomplete. Ask the instructor to prepare this clone.')
        return json.loads(self.manifest.read_text())

    def _idle(self):
        if self.state.get('busy') or self.engine.state.get('busy'):
            raise ValueError('Another operation is active. Wait or stop it first.')

    def create(self,payload):
        spec=scenario(payload.get('seed'))
        profile=self.profile()
        before=self.root/profile['before_checkpoint']
        if not (before/'config.json').is_file():
            raise ValueError('The prepared early checkpoint is missing.')
        with self.engine.lock:
            self._idle()
            reservation=self.state.get('restore_nemotron',False)
            self.state={'phase':'ready','busy':False,'session_id':uuid.uuid4().hex[:12],
                'scenario':spec,'before_checkpoint':str(before),'before':None,'after':None,
                'training':None,'after_checkpoint':None,'error':None,'profile':profile,
                'restore_nemotron':reservation}
            self._save()
            return self.status()

    def status(self):
        with self.engine.lock:
            value=copy.deepcopy(self.state)
        session=value.get('session_id')
        if session:
            folder=self.results/session
            progress=folder/'train/progress.json'
            if progress.exists(): value['training']=json.loads(progress.read_text())
            stage=value.get('operation')
            if stage in ('before','after'):
                files=list((folder/stage).glob('*/progress.json'))
                if files:
                    file=max(files,key=lambda f:f.stat().st_mtime)
                    value['progress']=json.loads(file.read_text())
        return value

    def start(self,operation,payload):
        if operation not in ('train','before','after'):
            raise ValueError('Unknown training operation.')
        with self.engine.lock:
            self._idle()
            if not self.state.get('session_id'): raise ValueError('Create an experiment first.')
            if operation=='train':
                if not self.state.get('before'): raise ValueError('Run the early checkpoint first.')
                if self.state.get('training') or (self.results/self.state['session_id']/'train').exists():
                    raise ValueError('Create a new experiment to train again.')
                steps=payload.get('steps',self.profile()['default_steps'])
                if type(steps) is not int or not 100<=steps<=5000: raise ValueError('Choose 100–5000 optimizer steps.')
                if shutil.disk_usage(self.root).free<30*1024**3:
                    raise ValueError('Training requires 30 GiB free for checkpoint weights. Archive old experiments first.')
            else:
                steps=None
                if self.state.get(operation): raise ValueError('This evaluation is already recorded. Create a new experiment to repeat.')
                if operation=='after' and not self.state.get('after_checkpoint'):
                    raise ValueError('Complete real training before testing the learned checkpoint.')
                prior=self.results/self.state['session_id']/operation
                if prior.exists():
                    archived=prior.with_name(operation+'-attempt-'+uuid.uuid4().hex[:8])
                    prior.rename(archived)
                    self.state.setdefault('evaluation_attempts',[]).append({'operation':operation,
                        'folder':str(archived),'error':self.state.get('error'),'outcome':'incomplete'})
            self.state.update(busy=True,phase='preparing',operation=operation,error=None)
            self.stop_event.clear()
            self._save()
            threading.Thread(target=self._run,args=(operation,steps),daemon=True).start()
        return {'accepted':True}

    def stop(self):
        with self.engine.lock:
            if not self.state.get('busy'):
                if not self.state.get('restore_nemotron'):raise ValueError('No training experiment is active.')
                previous=self.state['phase']
                self.state.update(busy=True,phase='releasing')
                self._save()
                threading.Thread(target=self._release,args=(previous,),daemon=True).start()
                return {'accepted':True}
            self.stop_event.set()
            self.state['phase']='stopping'
            self._save()
        return {'accepted':True}

    def _release(self,previous):
        restored=self._restore_service()
        with self.engine.lock:
            self.state.update(busy=False,restore_nemotron=not restored,phase=previous if restored else 'error')
            if not restored:self.state['error']='Nemotron restart failed; retry releasing the GPU after restoring Docker.'
            self._save()

    def _run(self,operation,steps):
        folder=self.results/self.state['session_id']
        output=folder/('train' if operation=='train' else operation)
        output.mkdir(parents=True,exist_ok=True)
        restore=False
        resource_lock=None
        try:
            resource_lock=(self.root/'workshop/results/isaac/.lock').open('a')
            fcntl.flock(resource_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            # Training/evaluation uses GPU 0; Cosmos and the original policy use GPU 1.
            check=subprocess.run(['docker','inspect','-f','{{.State.Running}}','nvidia-nemotron-nano-9b-v2'],capture_output=True,text=True,check=True,timeout=10)
            running=check.stdout.strip()=='true'
            restore=running or self.state.get('restore_nemotron',False)
            with self.engine.lock:
                self.state['restore_nemotron']=restore;self._save()
            if running:self._pause_service()
            with self.engine.lock:self.state['gpu_status']='GPU 0 reserved until comparison, stop, or explicit release.'
            arena=self.root/'third_party/IsaacLab-Arena'
            if operation=='train':
                python=arena/'submodules/Isaac-GR00T/.venv/bin/python'
                args=[str(python),'-u',str(self.root/'workshop/scripts/gr00t_train_worker.py'),
                    '--base',self.state['before_checkpoint'],'--output',str(output),'--steps',str(steps),
                    '--seed',str(self.state['scenario']['seed'])]
            else:
                args=['python3','-u',str(self.root/'workshop/scripts/evaluate_training.py'),
                    '--checkpoint',self.state['before_checkpoint' if operation=='before' else 'after_checkpoint'],
                    '--output',str(output),'--seeds',str(self.state['scenario']['seed'])]
            with self.engine.lock:
                self.state['phase']='training' if operation=='train' else 'evaluating';self._save()
            with (output/'operation.log').open('w') as log:
                self.process=subprocess.Popen(args,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                stop_at=None;deadline=time.monotonic()+3600
                while self.process.poll() is None:
                    if time.monotonic()>deadline: self.stop_event.set()
                    if self.stop_event.is_set():
                        stop_at=stop_at or time.monotonic()
                        (output/'stop').touch()
                        for job in output.glob('*/job.json'): (job.parent/'stop').touch()
                        if time.monotonic()-stop_at>120:
                            os.killpg(self.process.pid,signal.SIGTERM)
                            try:self.process.wait(timeout=15)
                            except subprocess.TimeoutExpired:os.killpg(self.process.pid,signal.SIGKILL)
                    time.sleep(.5)
            if self.stop_event.is_set():
                with self.engine.lock:self.state['phase']='stopped'
                return
            if self.process.returncode: raise RuntimeError('Operation failed. Inspect '+str(output/'operation.log'))
            if operation=='train':
                training=json.loads((output/'progress.json').read_text())
                checkpoint=output/f"checkpoint-{training['training_steps']}"
                if training['phase']!='complete' or not training.get('weights_changed') or not (checkpoint/'config.json').exists():
                    raise RuntimeError('Training did not produce a verified changed checkpoint.')
                with self.engine.lock:self.state.update(training=training,after_checkpoint=str(checkpoint),phase='trained')
            else:
                evaluation=json.loads((output/'evaluation.json').read_text())
                result=evaluation['results'][0]
                if result.get('error') or result.get('stopped'):raise RuntimeError(result.get('error','Evaluation stopped.'))
                result['training_performed_during_episode']=result.pop('training_performed',False)
                result['checkpoint_training']={'prepared_task_updates':self.state['profile'].get('before_optimizer_steps',200),
                    'additional_updates':self.state['training']['training_steps'] if operation=='after' else 0,
                    'origin':'this experiment' if operation=='after' else 'prepared early checkpoint'}
                with self.engine.lock:self.state[operation]=result;self.state['phase']='reviewing_evidence'
                # Visual reasoning is evidence only; simulator predicates remain the measured outcome.
                try:
                    clip=output/result['run_id']/'episode.mp4'
                    result['cosmos']=self.engine.client.ask_cosmos(clip,
                        'Describe whether the robot grasps the bottle, places it in the fridge, and closes the door. '
                        'State visual evidence and uncertainty. Do not assume success.',frames=16)
                    result['vss']=self.engine._evidence(clip)
                except Exception as error:
                    result['evidence_error']=str(error)
                atomic_json(output/result['run_id']/'report.json',result)
                with self.engine.lock:self.state['phase']='evaluated'
                if operation=='after':
                    before=self.state['before']
                    with self.engine.lock:
                        self.state['comparison']={'same_scenario':before['scenario']==result['scenario'],
                            'same_initial_state':before['initial_state']==result['initial_state'],
                            'equal_control_budget':before['scenario']['step_budget']==result['scenario']['step_budget'],
                            'before_success':before['success'],'after_success':result['success'],
                            'improved':not before['success'] and result['success'],
                            'weights_changed':self.state['training']['weights_changed']}
                        self.state['phase']='complete'
        except Exception as error:
            with self.engine.lock:self.state.update(phase='error',error=str(error))
        finally:
            if restore and (operation=='after' or self.stop_event.is_set() or self.state['phase']=='error'):
                restored=self._restore_service()
                with self.engine.lock:
                    self.state['restore_nemotron']=not restored
                    if not restored:self.state.update(phase='error',error='Nemotron restart failed; restore the VSS agent dependency before continuing.')
            with self.engine.lock:
                if self.stop_event.is_set():self.state['phase']='stopped'
                self.state['busy']=False;self._save()
                atomic_json(folder/'report.json',self.state)
            if resource_lock:resource_lock.close()

    def media(self,run_id,name):
        matches=list(self.results.glob(f'*/before*/{run_id}/{name}'))+list(self.results.glob(f'*/after*/{run_id}/{name}'))
        return matches[0] if matches else None
