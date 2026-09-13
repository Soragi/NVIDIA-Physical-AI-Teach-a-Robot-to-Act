# Deploy the NVIDIA Isaac workshop

Prepare one Brev `g7e.12xlarge` instance with two RTX PRO 6000 Blackwell Server GPUs (96 GB each), Ubuntu 22.04, Docker with NVIDIA Container Toolkit, and Jupyter Python 3.12. Downloads and compilation happen before the 90-minute class. Use the existing instance or an authorized clone; the scripts do not create cloud instances.

## Prepare a clean checkout

1. Clone the repository and run the notebook from its repository root.
2. Open `00_START_HERE_ROBOTICS_VSS.ipynb` and follow Setup A–I, then the learning experiment and Finish cell. This is the complete single-notebook path. Its hidden prompts save the NVIDIA/NGC and HF credentials privately. The terminal commands below describe the same underlying deployment stages.
3. Install `uv==0.12.13` in the Jupyter environment. Run `python3 workshop/scripts/deploy_lab.py deploy` to prepare VSS, Cosmos and Nemotron. This base installer has its own storage/driver checks and needs model/container entitlements.
4. Run `python3 workshop/scripts/install_isaac.py`. It installs the isolated simulator and policy environments, rendering libraries, pretrained checkpoint and Blackwell CUDA overlay.
5. Run `python3 workshop/scripts/validate_isaac_baseline.py`. It starts a temporary policy if necessary and runs the official unmodified three-episode evaluation. Inspect the recorded predicates and videos; a valid deployment can still have failed episodes.
6. Run `python3 workshop/scripts/activate_isaac.py`. Activation requires at least one recorded successful baseline. It installs the persistent policy and API services and refreshes the nginx UI container.
7. Open Brev's authenticated Secure Link for port 7777 and validate the five-stage training UI. Keep every backend service port private.

Continue with [TRAINING_LAB.md](TRAINING_LAB.md). Its preparation adds foundation weights, demonstrations and an early task checkpoint without replacing the pinned simulator.

## Reproducibility and memory

| Component | Pin |
|---|---|
| Arena release 0.3.0 | `3032c7888b7b56e54d907e5c46e98c6dfdfb7dba` |
| Arena Isaac Lab submodule | `af1bab4dc173ba69b08fab779c14ead61d13fd33` |
| GR00T policy code | `e29d8fc50b0e4745120ae3fb72447986fe638aa6` |
| GR1 checkpoint | `f5e388c35340080644e17a3b3d538150ad894bfa` |
| Simulator environment | Arena's frozen `uv.lock`, wheel group, Python 3.12 |
| Policy environment | Saved `workshop/runtime/isaac-policy.uv.lock`, Python 3.10 |
| Blackwell overlay | PyTorch `2.7.1+cu128`, torchvision `0.22.1+cu128`, flash-attn `2.8.3` |
| Task assets | SHA256 manifest `workshop/runtime/isaac-assets.json` (126 files) |

The upstream policy's default CUDA 12.6 wheel lacks `sm_120` kernels. The installer explicitly selects CUDA 12.8. Arena is alpha software; the pinned release and verified behavior are the compatibility boundary.

GPU 0 hosts Nemotron until training/evaluation reserves it. GPU 1 hosts Cosmos, Isaac rendering and the original GR00T evaluation policy. Retain Cosmos `NIM_GPU_MEMORY_UTILIZATION=0.60` and `NIM_MAX_MODEL_LEN=16384`; the previous generic NIM cache settings were not read by that launcher. Read actual headroom during inference before increasing budgets.

The official scene uses NVIDIA-distributed GR1, bottle and kitchen assets, including third-party asset provenance. Model licenses and asset terms are separate from application code. The GR1 checkpoint is licensed for research/evaluation; do not imply the same checkpoint is approved for partner production deployment.

The installer caches robot/bottle dependencies and the official kitchen under `third_party/isaac-assets`, checks their content against the published manifest, and the mission worker uses verified local copies. Built-in materials follow the pinned simulator runtime. The pinned Arena release references a staging asset bucket; preparation fails if those upstream files change. Preserve the prepared asset cache when cloning. `--record` on the asset helper is a maintainer operation that creates a new pin, not a way for attendees to bypass a mismatch.

## Operations

```bash
sudo systemctl status physical-ai-gr1 physical-ai-isaac
python3 workshop/scripts/deploy_lab.py status
python3 workshop/scripts/verify_package.py
python3 -m unittest discover -s workshop/tests
python3 workshop/scripts/validate_training_sweep.py --seeds 101 202 303 404 505 606 707 808 909 1010
```

Run one operator per instance. Credentials and service logs live in `~/.local/state/vss-robotics-workshop/`; results and videos live in ignored `workshop/results/`.

The UI API listens on localhost 8097; GR00T listens on localhost 15556. Service restarts terminate their worker process group. Interrupted runs remain failures with retained artifacts; they are not resumed or replaced by recorded successes.

The workshop applies 15-second send/receive timeouts to the native GR00T client and rejects new missions when the policy port is unavailable. The VSS catalogue limit is 512 sensor/file entries so the evaluation history can remain indexed. This profile uses the VIOS clip catalogue and timelines; RTVI embedding generation is not enabled. Large-catalogue semantic retrieval is separate from the episode-evidence workflow demonstrated here.

## Source references

- [Official pretrained GR1 evaluation](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/example_workflows/sequential_static_manipulation/step_5_evaluation.html)
- [Checkpoint and evaluation license](https://huggingface.co/nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task)
- [Arena release](https://github.com/isaac-sim/IsaacLab-Arena/tree/3032c7888b7b56e54d907e5c46e98c6dfdfb7dba)
