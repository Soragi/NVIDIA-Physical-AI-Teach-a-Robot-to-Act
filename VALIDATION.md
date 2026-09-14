# Validation boundary

The source workshop was validated on a two-GPU RTX PRO 6000 Blackwell Server environment. A fixed ten-seed comparison recorded 0/10 task completions with the early checkpoint and 8/10 with the trained checkpoint under matching initial states and control budgets. Seeds 606 and 1010 remained failures. A separate notebook run performed 2,000 optimizer updates in 518.49 seconds and changed its seed-404 evaluation from failure to success.

These results establish that the workflow performed real weight updates and loaded the resulting checkpoint. They do not guarantee improvement for every attendee, new scene, robot or training run. The 100 demonstration episodes are training inputs rather than a held-out evaluation set.

For a release candidate, run the complete Start Here notebook in a newly prepared environment and retain its local report. Verify:

- an official unmodified GR00T task episode succeeds before activating services;
- the early and trained evaluations use the same recorded initial state and 500-step budget;
- training records changed parameter hashes and a new checkpoint path;
- the retest loads that new checkpoint explicitly;
- all failures and service errors remain visible;
- only port 7777 is exposed through the authenticated Brev Secure Link.

Generated reports remain ignored because they contain machine paths, run identifiers and operational logs. Publish only an intentionally sanitized aggregate such as this document.

## Deployment repair validation

A prepared two-GPU Brev VM was repaired and tested with the current notebook. The gateway was reachable on both loopback and the VM network address; its HTML, CSS, JavaScript and proxied training API returned HTTP 200. A read-only Chrome check connected successfully at desktop and mobile sizes, with no JavaScript errors or horizontal mobile overflow. Backend services remained on loopback.

The recorded seed-3 baseline failed. Continuing its early checkpoint for 2,000 optimizer updates took 511.68 seconds and verified changed weights. The live retest succeeded in 339 control steps, using the same recorded initial state and equal 500-step allowance. Cosmos and VSS evidence were present for both evaluations, without an evidence error. GPU resources were released and Nemotron restored afterward. All 33 source regression tests passed on the VM.

That earlier repair validated the deployed workflow and cached setup checks. The subsequent cold installation validation is recorded below.

## Fresh Brev notebook validation — 2026-09-14

The complete Start Here notebook ran on a newly provisioned matching Brev VM with no Isaac environment, task assets, demonstrations, or early checkpoint installed. NVIDIA and Hugging Face credentials were supplied privately before execution. The initial VSS download encountered truncated NVCR authentication responses; the checked-in retry logic recovered both NIM downloads. A subsequent setup rerun also motivated reusing images already present locally, avoiding unnecessary registry requests.

All 37 notebook cells were traversed, including all 18 executable cells, with zero notebook cell errors. Setup A–I, the learning exercise, report export, and final attendee UI activation completed. The official task baseline succeeded in all three episodes, and all 126 asset files matched the checked-in manifest.

The notebook generated the early checkpoint with 200 optimizer updates, then performed 2,000 additional updates in 518.92 seconds. The recorded early evaluation failed and the trained retest succeeded, with verified changed weights, matching initial states, and equal control budgets. Both evaluations included Cosmos and VSS evidence without an evidence error. This is one observed comparison, not a guarantee for every seed.

All 34 regression tests passed on the VM. The final UI returned HTTP 200, connected in Chrome, supported all five navigation steps, and passed desktop/mobile checks with no JavaScript errors or horizontal overflow. Model and VSS services were healthy, GPU resources were released, and Finish opened a fresh attendee experiment. Executed notebooks, credentials, and detailed reports remain private and outside Git.

This validation applies to the documented two-GPU RTX PRO 6000 Blackwell Brev configuration. First downloads, compilation, credentials/model access, and external service availability remain prerequisites; transient retries do not guarantee success during a sustained external outage.
