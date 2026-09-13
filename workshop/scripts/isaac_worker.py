"""One persistent GR1 physics episode; interventions never reset the environment."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ARENA = ROOT/'third_party/IsaacLab-Arena'
sys.path[:0] = [str(ROOT),str(ARENA)]
os.chdir(ARENA)
os.environ.update(OMNI_KIT_ACCEPT_EULA='YES',ACCEPT_EULA='Y',
    CUDA_VISIBLE_DEVICES='1',MPLBACKEND='Agg')
from workshop.lab.isaac_contract import atomic_json, digest, ACTIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job',type=Path,required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    folder = args.job.parent
    spec = job['scenario']
    started = time.monotonic()
    def update(**values):
        progress.update(values)
        progress['wall_time_s'] = round(time.monotonic()-started,2)
        atomic_json(folder/'progress.json',progress)
    progress = {'run_id':job['run_id'],'phase':'loading','step':0,'scene_revision':0,
                'action_chunk':16,'scenario':spec,'mode':job['mode']}
    update()
    import torch
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True,enable_cameras=True,device='cuda:0').app
    import isaaclab_arena_environments  # register NVIDIA's official assets and environments
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.gr1_put_and_close_door_environment import (
        GR1PutAndCloseDoorEnvironment, GR1PutAndCloseDoorEnvironmentCfg)
    from isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy import (
        Gr00tRemoteClosedloopPolicy, Gr00tRemoteClosedloopPolicyCfg)
    from isaaclab_arena.video.camera_observation_video_recorder import _to_uint8
    from PIL import Image
    from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
    factory = GR1PutAndCloseDoorEnvironment()
    definition = factory.build(GR1PutAndCloseDoorEnvironmentCfg(enable_cameras=True,embodiment='gr1_joint'))
    from workshop.lab.isaac_assets import apply_pinned_assets
    asset_revision = apply_pinned_assets(ROOT,definition)
    builder = ArenaEnvBuilder(definition,ArenaEnvBuilderCfg(seed=42,placement_seed=42,resolve_on_reset=False))
    env = builder.make_registered()
    base = env.unwrapped
    base.episode_recorder.set_output_path(str(folder/'episodes.jsonl'))
    from workshop.lab.isaac_policy_client import install_policy_timeouts
    install_policy_timeouts()
    policy = Gr00tRemoteClosedloopPolicy(Gr00tRemoteClosedloopPolicyCfg(
        policy_config_yaml_path='isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml',
        remote_host='127.0.0.1',remote_port=job.get('policy_port',15556),policy_device='cuda:0'))
    obs,_ = env.reset(seed=spec['seed'])
    policy.reset()
    policy.set_task_description(None)
    bottle = base.scene['ranch_dressing_hope_robolab']
    pose = bottle.data.root_pose_w.clone()
    pose[0,0] = 4.05 + spec['initial_offset_m'][0]
    pose[0,1] = -.58 + spec['initial_offset_m'][1]
    bottle.write_root_pose_to_sim(pose)
    bottle.write_root_velocity_to_sim(torch.zeros_like(bottle.data.root_vel_w))

    def observe():
        base.sim.forward()
        for _ in range(3):
            base.sim.render()
        base.scene.update(base.physics_dt)
        for sensor in base.scene.sensors.values():
            if hasattr(sensor,'is_initialized') and sensor.is_initialized:
                sensor.update(base.physics_dt,force_recompute=True)
        return base.observation_manager.compute()

    def world():
        return {'bottle_pose':bottle.data.root_pose_w.detach().cpu().tolist(),
                'joints':base.scene['robot'].data.joint_pos.detach().cpu().tolist()}

    obs = observe()
    initial = world()
    video = FFMPEG_VideoWriter(str(folder/'episode.mp4'),(512,512),50,codec='libx264',preset='ultrafast')
    recent = []
    action_hash = __import__('hashlib').sha256()
    stopped = False
    success = False
    intervention = None
    inference_s = 0.0
    executed_steps = 0
    policy_io = []
    last_info = {}
    try:
        update(phase='executing',initial_state=initial)
        for step in range(spec['step_budget']):
            if (folder/'stop').exists():
                stopped = True
                break
            if step == spec['disturbance_step'] and job['mode'] != 'baseline':
                before = world()
                pose = bottle.data.root_pose_w.clone()
                pose[0,:2] += torch.tensor(spec['disturbance_m'],device=pose.device)
                bottle.write_root_pose_to_sim(pose)
                bottle.write_root_velocity_to_sim(torch.zeros_like(bottle.data.root_vel_w))
                obs = observe()
                after = world()
                update(scene_revision=1,disturbance={'before':before,'after':after,'step':step})
                # Capture the fresh camera observation without consuming a control step.
                fresh = _to_uint8(obs['camera_obs']['robot_pov_cam_rgb'][0])
                Image.fromarray(fresh).save(folder/'frame.tmp.jpg')
                (folder/'frame.tmp.jpg').replace(folder/'frame.jpg')
                recent.extend([fresh]*10)  # Explicit paused observation, not ten physics steps.
                if job['mode'] == 'supervised':
                    clip = FFMPEG_VideoWriter(str(folder/'evidence.mp4'),(512,512),10,codec='libx264',preset='ultrafast')
                    for frame in recent:
                        clip.write_frame(frame)
                    clip.close()
                    frozen = world()
                    evidence_hash = __import__('hashlib').sha256((folder/'evidence.mp4').read_bytes()).hexdigest()
                    update(phase='awaiting_review',evidence_sha256=evidence_hash,paused_state=frozen,step=step,
                           evidence_playback_fps=10,simulation_control_hz=50)
                    deadline = time.monotonic()+1800
                    while not (folder/'command.json').exists():
                        if (folder/'stop').exists() or time.monotonic()>deadline:
                            stopped = True
                            break
                        time.sleep(.1)
                    if stopped:
                        break
                    command = json.loads((folder/'command.json').read_text())
                    if command['run_id'] != job['run_id'] or command['scene_revision'] != 1 or command['evidence_sha256'] != evidence_hash:
                        raise ValueError('Worker rejected a stale intervention.')
                    action = command['action']
                    if action not in ACTIONS:
                        raise ValueError('Unknown intervention.')
                    if action == 'stop_for_review':
                        stopped = True
                        break
                    if action in ('reobserve','reobserve_short'):
                        # Scheduler masks were created during inference and must be
                        # mutated under the same PyTorch inference context.
                        with torch.inference_mode():
                            policy.reset()
                            if action == 'reobserve_short':
                                policy._chunking_state.action_chunk_length = spec['short_chunk']
                    unchanged = digest(frozen) == digest(world())
                    if not unchanged:
                        raise RuntimeError('Scene changed while awaiting approval.')
                    intervention = {'action':action,'step':step,'scene_preserved':unchanged,
                                    'simulation_time_s':step/50,'evidence_clip_time_s':step/10,
                                    'before_state_hash':digest(frozen),'after_state_hash':digest(world()),
                                    'chunk_before':16,'chunk_after':policy._chunking_state.action_chunk_length}
                    update(phase='executing',intervention=intervention,action_chunk=policy._chunking_state.action_chunk_length)
            frame = _to_uint8(obs['camera_obs']['robot_pov_cam_rgb'][0])
            video.write_frame(frame)
            recent.append(frame)
            recent = recent[-40:]
            if step % 4 == 0:
                Image.fromarray(frame).save(folder/'frame.tmp.jpg')
                (folder/'frame.tmp.jpg').replace(folder/'frame.jpg')
                update(step=step,phase='executing')
            with torch.inference_mode():
                infer_start = time.monotonic()
                action = policy.get_action(env,obs)
                inference_s += time.monotonic()-infer_start
                if step in (0,spec['disturbance_step']):
                    policy_io.append({'step':step,'instruction':policy.task_description,
                        'camera_shape':list(frame.shape),
                        'joint_positions':obs['policy']['robot_joint_pos'].detach().cpu().tolist(),
                        'executed_action':action.detach().cpu().tolist(),
                        'predicted_chunk':policy._chunking_state.current_action_chunk.detach().cpu().tolist()})
                action_hash.update(action.detach().cpu().numpy().tobytes())
                obs,_,terminated,truncated,last_info = env.step(action)
                executed_steps += 1
            if terminated.any() or truncated.any():
                rows = [json.loads(line) for line in (folder/'episodes.jsonl').read_text().splitlines() if line]
                success = bool(rows[-1]['success'])
                break
        record = {'schema_version':2,'run_id':job['run_id'],'scenario':spec,'mode':job['mode'],
                  'success':success,'stopped':stopped,'episode_steps':executed_steps,
                  'action_sha256':action_hash.hexdigest(),'intervention':intervention,
                  'policy_io':policy_io,
                  'subtask_ever_succeeded':last_info.get('subtask_success_state'),
                  'success_definition':'Arena sequential termination requires current bottle placement and door closure. Auxiliary progress scores are diagnostic.',
                  'initial_state':initial,'wall_time_s':round(time.monotonic()-started,2),
                  'inference_s':round(inference_s,3),'inference_calls':policy._chunking_state._n_fetch_calls,
                  'versions':{'arena':'3032c7888b7b56e54d907e5c46e98c6dfdfb7dba',
                              'assets_sha256':asset_revision,
                              'gr00t':'e29d8fc50b0e4745120ae3fb72447986fe638aa6',
                              'checkpoint':job.get('checkpoint','f5e388c35340080644e17a3b3d538150ad894bfa')},
                  'simulator':'NVIDIA Isaac Sim 6.0','policy':job.get('policy_name','GR00T N1.6 Arena GR1 pretrained'),
                  'physical_command_sent':False,'training_performed':False}
        if (folder/'episodes.jsonl').exists():
            record['episode_metrics'] = [json.loads(line) for line in (folder/'episodes.jsonl').read_text().splitlines() if line]
        atomic_json(folder/'result.json',record)
        update(phase='stopped' if stopped else 'complete',step=executed_steps,success=success)
    except BaseException as error:
        # Isaac shutdown can terminate Python before an outer exception handler runs.
        import traceback
        atomic_json(folder/'error.json',{'error':str(error),'type':type(error).__name__,'traceback':traceback.format_exc()})
        raise
    finally:
        video.close()
        policy.close()
        env.close()
        app.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        import traceback
        traceback.print_exc()
        folder = Path(sys.argv[sys.argv.index('--job')+1]).parent
        atomic_json(folder/'error.json',{'error':str(error)})
        raise
