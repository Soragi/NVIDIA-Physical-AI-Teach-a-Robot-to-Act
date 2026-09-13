"""Boundary tests: no GPUs, fake model predictions, or substituted success."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from workshop.lab.training import TrainingLab
from workshop.lab.isaac_contract import atomic_json


class TrainingBoundaries(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.engine=SimpleNamespace(root=self.root,lock=threading.RLock(),state={'busy':False})
        atomic_json(self.root/'workshop/training_profile.json',{'before_checkpoint':'early','default_steps':2000})
        atomic_json(self.root/'early/config.json',{})
        self.lab=TrainingLab(self.engine)

    def tearDown(self):self.tmp.cleanup()

    def test_same_recorded_scene_and_fresh_session(self):
        one=self.lab.create({'seed':101});two=self.lab.create({'seed':101})
        self.assertEqual(one['scenario'],two['scenario'])
        self.assertNotEqual(one['session_id'],two['session_id'])
        self.assertIsNone(two['after_checkpoint'])

    def test_reject_training_before_baseline_and_after_before_training(self):
        self.lab.create({'seed':101})
        for operation in ('train','after'):
            with self.assertRaises(ValueError):self.lab.start(operation,{})

    def test_reject_overlapping_external_operation(self):
        self.engine.state['busy']=True
        with self.assertRaises(ValueError):self.lab.create({})
        self.engine.state['busy']=False
        self.lab.create({})
        self.lab.state['busy']=True
        self.assertTrue(self.engine.external_busy())
        with self.assertRaises(ValueError):self.lab.start('before',{})

    def test_malformed_steps_and_seeds(self):
        for seed in (True,-1,'101'):
            with self.assertRaises(ValueError):self.lab.create({'seed':seed})
        self.lab.create({})
        self.lab.state['before']={'success':False}
        for steps in (True,0,5001,1.5,'2000'):
            with self.assertRaises(ValueError):self.lab.start('train',{'steps':steps})

    def test_interrupted_training_cannot_be_presented_as_completed(self):
        self.lab.create({})
        self.lab.state.update(busy=True,phase='training')
        self.lab._save()
        recovered=TrainingLab(self.engine)
        self.assertFalse(recovered.status()['busy'])
        self.assertEqual(recovered.status()['phase'],'interrupted')
        self.assertIsNone(recovered.status()['after_checkpoint'])

    def test_no_reuse_of_partial_training_output(self):
        state=self.lab.create({});self.lab.state['before']={'success':False}
        (self.lab.results/state['session_id']/'train').mkdir(parents=True)
        with self.assertRaises(ValueError):self.lab.start('train',{'steps':200})

    def test_stop_is_explicit_and_does_not_create_success(self):
        self.lab.create({})
        with self.assertRaises(ValueError):self.lab.stop()
        self.lab.state['busy']=True
        self.lab.stop()
        self.assertTrue(self.lab.stop_event.is_set())
        self.assertEqual(self.lab.state['phase'],'stopping')
        self.assertIsNone(self.lab.state['after'])

    def test_status_does_not_expose_mutable_state(self):
        self.lab.create({'seed':101})
        status=self.lab.status();status['scenario']['seed']=5
        self.assertEqual(self.lab.state['scenario']['seed'],101)

    def test_failed_model_process_restores_paused_service(self):
        self.lab.create({'seed':101})
        self.lab.state.update(busy=True,operation='before')
        (self.root/'workshop/results/isaac').mkdir()
        commands=[]
        def command(args,**kw):
            commands.append(args)
            return SimpleNamespace(stdout='true\n',returncode=0)
        with patch('workshop.lab.training.subprocess.run',side_effect=command),patch('workshop.lab.training.subprocess.Popen',side_effect=OSError('Model runtime unavailable')):
            self.lab._run('before',None)
        self.assertEqual(self.lab.state['phase'],'error')
        self.assertFalse(self.lab.state['busy'])
        self.assertIsNone(self.lab.state['before'])
        self.assertIn(['docker','start','nvidia-nemotron-nano-9b-v2'],commands)
        self.assertIn('Model runtime unavailable',self.lab.state['error'])

    def test_retry_keeps_failed_evaluation_evidence(self):
        state=self.lab.create({'seed':101})
        prior=self.lab.results/state['session_id']/'before'
        prior.mkdir(parents=True);(prior/'operation.log').write_text('model unavailable')
        self.lab.state.update(phase='error',error='model unavailable')
        with patch('workshop.lab.training.threading.Thread.start'):
            self.lab.start('before',{})
        archived=Path(self.lab.state['evaluation_attempts'][0]['folder'])
        self.assertEqual((archived/'operation.log').read_text(),'model unavailable')
        self.assertFalse(prior.exists())

    def test_occupied_policy_port_cannot_substitute_existing_model(self):
        from workshop.scripts import evaluate_training
        with patch('sys.argv',['evaluate_training','--checkpoint',str(self.root/'early'),'--output',str(self.root/'new-evaluation')]),patch.object(evaluate_training.socket,'socket') as socket_mock,patch.object(evaluate_training.subprocess,'Popen') as process:
            socket_mock.return_value.__enter__.return_value.bind.side_effect=OSError('Address in use')
            with self.assertRaisesRegex(RuntimeError,'no checkpoint substituted'):
                evaluate_training.main()
            process.assert_not_called()

    def test_gpu_reservation_spans_all_three_operations(self):
        state=self.lab.create({'seed':101})
        self.engine.client=SimpleNamespace(ask_cosmos=lambda *a,**kw:{'answer':'unit-test fixture'})
        self.engine._evidence=lambda path:{}
        (self.root/'workshop/results/isaac').mkdir()
        folder=self.lab.results/state['session_id']
        common={'stopped':False,'scenario':state['scenario'],'initial_state':{'pose':[0]}}
        for stage,success,run_id in [('before',False,'111111111111'),('after',True,'222222222222')]:
            atomic_json(folder/stage/'evaluation.json',{'results':[{**common,'run_id':run_id,'success':success}]})
        atomic_json(folder/'train/progress.json',{'phase':'complete','training_steps':2000,'weights_changed':True})
        atomic_json(folder/'train/checkpoint-2000/config.json',{})
        calls=[];running=True
        def command(args,**kw):
            nonlocal running
            calls.append(args)
            if args[1]=='stop':running=False
            if args[1]=='start':running=True
            return SimpleNamespace(stdout=('true' if running else 'false')+'\n',returncode=0)
        process=SimpleNamespace(poll=lambda:0,returncode=0)
        with patch('workshop.lab.training.subprocess.run',side_effect=command),patch('workshop.lab.training.subprocess.Popen',return_value=process):
            for stage in ['before','train']:
                self.lab._run(stage,2000)
                self.assertTrue(self.lab.state['restore_nemotron'])
                self.assertTrue(self.engine.external_busy())
                self.assertFalse(any(c[1]=='start' for c in calls))
            self.lab._run('after',None)
        self.assertEqual(sum(c[1]=='stop' for c in calls),1)
        self.assertEqual(sum(c[1]=='start' for c in calls),1)
        self.assertFalse(self.lab.state['restore_nemotron'])
        self.assertFalse(self.engine.external_busy())
        self.assertEqual(self.lab.state['phase'],'complete')

    def test_idle_release_preserves_completed_training(self):
        self.lab.create({'seed':101})
        self.lab.state.update(phase='trained',restore_nemotron=True,after_checkpoint='my-trained-checkpoint')
        with patch.object(self.lab,'_restore_service',return_value=True),patch('workshop.lab.training.threading.Thread.start',lambda thread:thread.run()):
            self.lab.stop()
        self.assertFalse(self.lab.state['busy'])
        self.assertFalse(self.lab.state['restore_nemotron'])
        self.assertEqual(self.lab.state['after_checkpoint'],'my-trained-checkpoint')

    def test_stop_during_nim_initialization_starts_no_work(self):
        self.lab.stop_event.set()
        with patch('workshop.lab.training.subprocess.run',return_value=SimpleNamespace(stdout='starting\n',returncode=0)) as command:
            with self.assertRaisesRegex(RuntimeError,'Stopped before GPU allocation'):
                self.lab._pause_service()
            self.assertEqual(command.call_count,1)

if __name__=='__main__':unittest.main()
