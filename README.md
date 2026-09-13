# Physical AI Masterclass: Teach a Robot to Act

A 90-minute developer lab that performs real GR00T fine-tuning and tests the new checkpoint in Isaac Sim. The robot must place a bottle in a refrigerator and close the door. Attendees test an early policy, inspect NVIDIA demonstrations, update model weights, retest the same scene and export the measured evidence.

## Start here

Use one matching Brev environment per team and one operator at a time. Open [`00_START_HERE_ROBOTICS_VSS.ipynb`](00_START_HERE_ROBOTICS_VSS.ipynb) inside this repository and run it from top to bottom.

The notebook contains the complete workflow:

1. Verify the two-GPU Brev host and install required build tools.
2. Enter NVIDIA/NGC and Hugging Face credentials through hidden prompts.
3. Deploy VSS, Cosmos and Nemotron.
4. Install the pinned Isaac Sim, Isaac Lab Arena and GR00T environments.
5. Validate NVIDIA's original task checkpoint in three real simulator episodes.
6. Download 100 demonstrations and create the early checkpoint with 200 optimizer updates.
7. Test, inspect demonstrations, train for 2,000 further updates, retest and compare.
8. Activate the five-stage browser UI on the instance's port 7777 Brev Secure Link.

Set `RUN_SETUP=True` on a new matching environment. Set it to `False` only when that environment is already prepared. Cold model downloads and compilation take additional time beyond the 90-minute learning lab.

## NVIDIA components

| Component | Purpose |
|---|---|
| GR00T N1.6 | Foundation robot policy, supervised task adaptation and action inference. |
| Isaac Sim + Isaac Lab Arena | Physics execution, reproducible scenes and task predicates. |
| Cosmos | Visual assessment of robot behavior and uncertainty. |
| VSS/VIOS | Episode registration, timelines and retained video evidence. |
| Nemotron | VSS agent reasoning dependency; paused while GPU 0 trains or evaluates GR00T. |
| Brev | Reproducible two-GPU developer environment and authenticated Secure Link. |

Only port 7777 should be shared through Brev Secure Link. Model APIs, Redis, Phoenix, VST and the mission API bind to loopback in this release. Do not expose their ports with additional Secure Links or cloud firewall rules.

## Verification

```bash
python3 workshop/scripts/check_source_secrets.py
python3 workshop/scripts/verify_package.py
python3 -m unittest discover -s workshop/tests
```

Package tests validate boundaries and source hygiene. They do not prove robot competence. A release environment still needs the notebook's official baseline, real training, checkpoint load and equal-condition before/after evaluation.

See [TRAINING_LAB.md](TRAINING_LAB.md) for teaching guidance, [ISAAC_DEPLOYMENT.md](ISAAC_DEPLOYMENT.md) for pinned dependencies and [SECURITY.md](SECURITY.md) before publishing or deploying.

## Scope

The repository contains no model weights, demonstrations, generated videos, credentials or participant reports. Those remain outside Git. The workshop sends no physical robot commands and does not establish production readiness or production licensing. Review the GR00T model, dataset, Isaac/Arena assets and container terms separately.

### Setup reruns and UI access

On a fresh VM, run Setup A–I once before starting the learning exercise. Setup G checks both the port-7777 UI and its proxied mission API. Create a Brev Secure Link targeting port 7777 on this instance. The gateway listens on the VM network so Brev can connect; all backend ports remain on loopback. A 503 connection-refused error from Brev means the gateway address/port or service must be checked.

For an already prepared VM, set `RUN_SETUP=False`. To rerun deployment, wait for any active operation to finish. Setup releases an idle GPU reservation while preserving its recorded experiment. Saved private credentials and complete cached inputs are reused. Notebook API errors explain conflicts; do not repeat completed training in the same experiment.
