# Security and publication

## Secrets

Never commit NVIDIA/NGC keys, Hugging Face tokens, Secure Link URLs, cookies, private keys or generated environment files. Start Here collects credentials with hidden prompts and stores them under `~/.local/state/vss-robotics-workshop/` with owner-only permissions. This directory is outside the repository but is still part of a full virtual-machine snapshot.

Before every public release, run:

```bash
python3 workshop/scripts/check_source_secrets.py
python3 workshop/scripts/verify_package.py
git grep -n -I -E 'hf_[A-Za-z0-9]{20,}|nvapi-[A-Za-z0-9_-]{20,}|github_pat_[A-Za-z0-9_]{20,}|BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY'
```

The final command should return no matches. Scan the complete Git history as well as the working tree. This repository begins with a sanitized single commit so the previous private instance identifier is not inherited.

## Network boundary

Expose only port 7777 using the authenticated Brev Secure Link. Backend services are configured for loopback access. Do not create Secure Links or inbound cloud rules for ports 6006, 6379, 8000, 8097, 15556, 15557, 30000–30082, 30888 or 30900.

The notebook assumes one operator per isolated instance. Its anti-CSRF token prevents stale browser actions but is not multi-user authentication. Do not reuse this workshop API as a shared production controller.

## Generated data

Credentials and logs live in `~/.local/state/vss-robotics-workshop/`. Model weights, datasets, checkpoints and reports live in ignored `third_party/` and `workshop/results/` directories. Remove those locations and browser history before turning a prepared machine into a distributable VM image, then configure fresh team credentials on each clone.

## Reporting

If you discover a credential in a public commit, revoke it first. Removing the latest file is insufficient: rewrite or replace the affected Git history before republishing.
