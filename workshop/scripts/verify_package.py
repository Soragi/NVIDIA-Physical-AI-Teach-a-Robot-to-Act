"""Offline checks for the distributable package; no inference claims."""
import ast, json, sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
for path in root.glob('*.ipynb'):
 if path.name.startswith('._'): continue  # macOS archive metadata is not a notebook.
 nb=json.loads(path.read_text())
 for cell in nb['cells']:
  if cell['cell_type']=='code':
   ast.parse(''.join(cell['source']))
   if '--allow-notebook-outputs' not in sys.argv:
    assert cell['execution_count'] is None and not cell['outputs'], 'Distribute a clean notebook: '+path.name
for p in (root/'workshop').rglob('*.py'):
 if not p.name.startswith('._'): ast.parse(p.read_text())
for name in ('LICENSE','LICENSE.DATA','LICENSE-3rd-party.txt'):
 assert (root/name).is_file()
print('Notebook syntax/clean outputs, Python syntax, and legal-file presence passed.')
