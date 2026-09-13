"""Actual GR00T supervised fine-tuning, with workshop-readable progress."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ARENA = ROOT/'third_party/IsaacLab-Arena'
POLICY = ARENA/'submodules/Isaac-GR00T'
sys.path.insert(0,str(POLICY))
os.environ.update(WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false')
token = Path.home()/'.local/state/vss-robotics-workshop/hf.token'
if token.exists():
    os.environ['HF_TOKEN'] = token.read_text().strip()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--dataset',type=Path,default=ROOT/'third_party/training-inputs/dataset/ranch_bottle_into_fridge/ranch_bottle_into_fridge_generated_100/lerobot')
    args=p.parse_args()
    if not 1 <= args.steps <= 30000 or not 1 <= args.batch <= 32:
        raise ValueError('Invalid training budget.')
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    progress={'phase':'loading','training_steps':0,'max_steps':args.steps,'loss_history':[],
              'base_checkpoint':str(args.base),'dataset':str(args.dataset),'training_performed':False,
              'training_seed':args.seed,'batch_size':args.batch,
              'trainable_components':['top 4 language layers','projector','diffusion action head','vision-language normalization'],
              'optimizer_restarted':True}
    def save(**kw):
        progress.update(kw,elapsed_s=round(time.monotonic()-started,2))
        temporary=args.output/'progress.tmp.json'
        temporary.write_text(json.dumps(progress,indent=2))
        temporary.replace(args.output/'progress.json')
    save()
    from transformers import TrainerCallback
    from gr00t.configs.base_config import get_default_config
    import gr00t.experiment.experiment as experiment
    config_path=ARENA/'isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py'
    spec=importlib.util.spec_from_file_location('workshop_gr1_config',config_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def weight_probe(model):
        name,param=next((n,p) for n,p in model.named_parameters() if p.requires_grad)
        return {'parameter':name,'sha256':hashlib.sha256(param.detach().float().cpu().numpy().tobytes()).hexdigest()}
    class Progress(TrainerCallback):
        def on_train_begin(self,args,state,control,model=None,**kw):
            save(phase='training',weight_before=weight_probe(model),
                 trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        def on_log(self,args,state,control,logs=None,**kw):
            if logs and 'loss' in logs:
                progress['loss_history'].append({'step':state.global_step,**logs})
            save(training_steps=state.global_step,training_performed=state.global_step>0)
        def on_step_end(self,training_args,state,control,**kw):
            if (args.output/'stop').exists():
                control.should_training_stop=True
                control.should_save=True
        def on_train_end(self,args,state,control,model=None,**kw):
            after=weight_probe(model)
            save(phase='saving',training_steps=state.global_step,weight_after=after,
                 weights_changed=after['sha256']!=progress['weight_before']['sha256'])
    Original=experiment.Gr00tTrainer
    class WorkshopTrainer(Original):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            # Workshop continuations intentionally start a fresh optimizer.
            # Avoid retaining tens of GiB of optimizer state per attendee run.
            self.args.save_only_model=True
            self.add_callback(Progress())
    experiment.Gr00tTrainer=WorkshopTrainer
    c=get_default_config().load_dict({'data':{'download_cache':False,'datasets':[{
        'dataset_paths':[str(args.dataset)],'mix_ratio':1.0,'embodiment_tag':'gr1'}]}})
    c.load_config_path=None
    c.model.tune_llm=False;c.model.tune_visual=False
    c.model.tune_top_llm_layers=4
    c.model.tune_projector=True;c.model.tune_diffusion_model=True
    c.model.load_bf16=False;c.model.reproject_vision=False;c.model.eagle_collator=True
    c.model.model_name='nvidia/Eagle-Block2A-2B-v2'
    c.model.backbone_trainable_params_fp32=True;c.model.use_relative_action=True
    c.training.start_from_checkpoint=str(args.base)
    c.training.output_dir=str(args.output)
    c.training.optim='adamw_torch';c.training.global_batch_size=args.batch
    c.training.dataloader_num_workers=2;c.training.learning_rate=1e-4
    c.training.max_steps=args.steps;c.training.num_gpus=1;c.training.use_wandb=False
    c.training.save_steps=args.steps;c.training.save_total_limit=1
    c.training.logging_steps=10
    c.training.seed=args.seed
    c.data.shard_size=1024;c.data.episode_sampling_rate=0.1;c.data.num_shards_per_epoch=100
    try:
        experiment.run(c)
        save(phase='stopped' if (args.output/'stop').exists() else 'complete')
    except BaseException as error:
        save(phase='error',error=str(error))
        raise

if __name__=='__main__':
    main()
