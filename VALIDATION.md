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

This validates the repaired deployed workflow and cached setup checks. It is not a new blank-VM installation test; a separate cold provisioning run remains necessary before claiming complete fresh-image reproducibility.
