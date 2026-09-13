"""Check Git-visible workshop files for accidentally embedded provider tokens."""
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
names=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT).decode().split('\0')
patterns=[
    rb'hf_[A-Za-z0-9]{20,}',
    rb'nvapi-[A-Za-z0-9_-]{30,}',
    rb'github_pat_[A-Za-z0-9_]{20,}',
    rb'gh[pousr]_[A-Za-z0-9]{20,}',
    rb'AKIA[0-9A-Z]{16}',
    rb'https?://[^\s/:]+:[^\s/@]+@',
    rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
]
matches=[]
for name in set(names)-{''}:
    path=ROOT/name
    if path.is_file() and any(re.search(pattern,path.read_bytes()) for pattern in patterns): matches.append(name)
if matches: raise SystemExit('Potential credentials found in: '+', '.join(sorted(matches)))
print('No provider-token or private-key patterns found in Git-visible source.')
