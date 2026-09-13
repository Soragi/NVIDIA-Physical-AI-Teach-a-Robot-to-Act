# Physical AI Masterclass: learn to act

This lab performs real supervised GR00T fine-tuning and evaluates the resulting checkpoint in Isaac Sim. This public repository contains only that training experiment and its evidence workflow.

The experiment is **early policy → demonstrations → gradient updates → new checkpoint → equal-budget robot test**. A short run may fail. The UI reports actual task outcomes instead of promising improvement or substituting a successful recording.

## Instructor preparation

Use the pinned environment in [ISAAC_DEPLOYMENT.md](ISAAC_DEPLOYMENT.md) on the existing Brev instance. The tested hardware is two RTX PRO 6000 Blackwell Server GPUs with approximately 96 GB each. GPU 1 runs the original GR00T/Cosmos services and Isaac rendering. The training workflow reserves GPU 0 across baseline, training and retest, then restores Nemotron after comparison, stop, or explicit release. Keeping the reservation between stages avoids a race with NIM startup. No new cloud instance is created by these scripts.

Open **`00_START_HERE_ROBOTICS_VSS.ipynb`** inside the cloned repository. It now covers the entire deployment and learning experiment; no second notebook is required.

1. **Setup A–C:** locate the repository, select `RUN_SETUP=True`, verify the existing Brev hardware/container runtime, install build tools and pinned uv, and enter NVIDIA/NGC plus HF credentials through hidden prompts.
2. **Setup D–G:** deploy VSS/Cosmos/Nemotron, install the pinned Isaac/GR00T environment, run the official three-episode baseline, and activate the shared policy/API services. At least one successful official baseline is required.
3. **Setup H–I:** download the foundation weights and 100 demonstrations, create the early checkpoint using 200 optimizer updates, and verify package/tests/API readiness. Existing complete inputs are retained; incomplete training output requires inspection.
4. **Learning sections 1–6:** run the unchanged 90-minute before/demonstrations/train/retest/compare experiment in the same notebook. Its `/learn.html` UI is available during validation.
5. **Finish:** activate the port 7777 learning home page after the local comparison proves changed weights and equal conditions. This clears the displayed experiment but retains evidence. Start Here keeps its complete setup cells.

A cold deployment takes additional time beyond the 90-minute learning lab. Downloads and compilation should normally be completed before that timed portion. The notebook checks for 180 GiB repository headroom before a cold Isaac/training install, 45 GiB for preparation on an installed profile, and 30 GiB is required for each training experiment. The base installer separately checks 300 GiB Docker/data headroom. Use the same image's working driver, Docker, NVIDIA Container Toolkit, passwordless sudo and Python 3.12 Jupyter kernel; this notebook does not provision the cloud infrastructure.

On an already prepared instance set `RUN_SETUP=False`; Setup I still verifies readiness. Keep `UPDATE_CREDENTIALS=False` unless replacing private keys. Do not rerun setup while an experiment is active. A complete first-time deployment on a newly provisioned cloud VM remains a separate validation requirement; existing-host checks do not prove cold download availability.

Pinned inputs are in `workshop/training_profile.json`; the download manifest is in `third_party/training-inputs/sources.json`. The worker uses the Arena GR1 arms-only observation/action configuration and the same pinned Isaac-GR00T source as inference.

## Attendee flow — 90 minutes

| Stage | Time | Developer task |
|---|---:|---|
| Understand | 10 min | Distinguish GR00T action generation, training, Isaac simulation and Cosmos reasoning. |
| Verify | 10 min | Open the notebook, check credentials, model files and the mission API. |
| Test early policy | 15 min | Choose a scene seed; inspect live camera, joint inputs, action chunks and simulator outcomes. |
| Inspect demonstrations | 10 min | Review the labeled training video and the dataset's state/action structure. |
| Train | 25 min | Run actual optimizer updates and inspect loss, weight hashes and the saved checkpoint. |
| Retest and compare | 20 min | Execute the new policy in the recorded scene and export all evidence. |

Continue in `00_START_HERE_ROBOTICS_VSS.ipynb`. The browser uses the same API and experiment state. Do not launch a second operation from another browser or notebook while one is active.

After activation, `00_START_HERE_ROBOTICS_VSS.ipynb` retains both deployment and training. The port 7777 entry point remains the five-stage learning UI.

Activation opens an empty experiment while retaining validation reports on disk. Clone this idle prepared state so attendees do not start on an instructor's completed video. The first test generates and records a fresh scene seed unless the attendee supplies one.

Remove instructor credentials from the distribution copy before sharing a VM template, then configure team credentials through the hidden notebook prompts on each clone. Keeping secrets outside Git does not exclude them from a whole-VM snapshot.

The browser intentionally contains only the deployed experiment: **Test → Demonstrations → Train → Retest → Compare**. Installation commands belong in the notebook and instructor guide. The demonstration video is explicitly labeled; live before/after recordings have distinct run IDs and URLs.

To leave an experiment between stages, click **Release training GPU** or call `lab.stop()` while idle. This restores the Nemotron dependency used by VSS without deleting completed training or evaluation evidence.

## What changes during training

The early checkpoint is not a robot without prior knowledge: it inherits foundation pretraining and a short task adaptation. Live training continues its weights using demonstration batches, with a fresh optimizer. The top four language layers, projector, diffusion action head and vision-language normalization are trainable; the visual encoder remains frozen.

An **optimizer step** updates weights from a training batch. A **control step** applies an action in the simulator. With action chunk 16, the controller applies 16 predicted actions before requesting a new inference. A 500-control-step episode does not mean 500 training updates.

The demonstrated adapter also separates dimensions: 54 raw simulator joint states map to the dataset's 26-value policy convention, and the adapter emits 36 simulator action values. Those are vector dimensions; 16 is the number of actions executed per inference chunk.

Every accepted training run saves a checkpoint and a before/after hash of a trainable parameter. The subsequent evaluation loads that checkpoint explicitly on a separate policy endpoint. It does not switch to the original specialized task checkpoint.

The report's top-level `training` object records weight updates; each robot episode is inference only. `checkpoint_training` identifies the preparation and live updates behind its policy. Older raw worker records use `training_performed: false` for the episode itself, not for the complete learning workflow.

## Evidence and evaluation

- Use identical scene parameters, action chunks and control-step budgets for both checkpoints.
- Record each scenario seed, actual initial simulator state, checkpoint path, training configuration, loss trace, action hash, inference count and task result.
- Task completion uses simulator predicates. Cosmos provides a separate visual assessment; VSS indexes the clip. Evidence-service errors remain explicit and never change a failed robot result to success.
- Success is Arena's sequential termination, which requires current bottle placement and door closure. The auxiliary progress fraction in this pinned alpha release can remain incomplete; it is diagnostic and is not used as the completion criterion.
- The 100 demonstration episodes are training data. Fresh simulator tests do not establish held-out demonstration accuracy or generalization to new hardware, objects or geometry.
- Predeclare evaluation seeds and publish all results. Additional seeds can produce similar motion because the task and workspace are intentionally bounded.
- A stopped, interrupted or failed training run is not registered as the attendee's completed checkpoint. A new experiment is required to retry.
- An evaluation-only service failure can be retried from the existing trained checkpoint. The previous attempt is archived and listed in the report; retrying does not silently erase evidence or repeat training.

Standalone evaluation for an instructor, while no other operation is active:

```bash
python3 workshop/scripts/evaluate_training.py \
  --checkpoint /absolute/path/to/checkpoint \
  --output /absolute/path/to/new/evaluation-directory \
  --seeds 101,202,303
```

The standalone evaluator assumes GPU 0 is available. The browser/notebook coordinator manages Nemotron's GPU allocation automatically; the standalone evaluator does not.

## Interfaces

`TrainingMission` in `workshop/lab/training.py` exposes `create(seed)`, `before()`, `train(steps)`, `after()`, `status()`, `wait()` and `stop()`. HTTP actions live under `/mission-api/training/` and require the existing per-service CSRF token. Allowed checkpoint selection is derived from the prepared profile and the current experiment's completed training output, not an arbitrary browser path.

Reports are saved under `workshop/results/training/<session-id>/`. Completed recordings are separate from the live `frame.jpg` stream. Operation locks prevent overlapping training and checkpoint evaluation through the shared API.

## Developer value and boundaries

The three-computer architecture connects training, simulation and robot deployment. This workshop demonstrates training and simulated robot inference on Brev. Physical hardware integration remains future work.

GR00T supplies reusable robot intelligence; task demonstrations adapt it; Isaac provides controlled evaluation; Cosmos and VSS help review operational evidence. Nemotron supports the VSS agent and is not the weight optimizer.

Review foundation model, dataset and robot asset licenses separately for production. The original specialized Arena checkpoint has research/evaluation restrictions. This workshop sends no physical robot commands, performs no training from scratch, and includes no Wandelbots integration.

Sources: [NVIDIA Arena training workflow](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/example_workflows/sequential_static_manipulation/step_4_policy_training.html), [GR00T N1.6 foundation model](https://huggingface.co/nvidia/GR00T-N1.6-3B), [matching Arena demonstrations](https://huggingface.co/datasets/nvidia/Arena-GR1-Manipulation-PlaceItemCloseDoor-Task).
