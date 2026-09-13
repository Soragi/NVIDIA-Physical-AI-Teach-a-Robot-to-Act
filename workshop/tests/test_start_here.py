"""Deployment notebook must retain setup and leave the working experiment intact."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[2]


class StartHereTests(unittest.TestCase):
    def test_training_actions_are_present_and_clean(self):
        start = json.loads((ROOT/'00_START_HERE_ROBOTICS_VSS.ipynb').read_text())
        code='\n'.join(''.join(c['source']) for c in start['cells'] if c['cell_type']=='code')
        for action in ('lab.create(', 'lab.before()', 'lab.train(', 'lab.after()'):
            self.assertIn(action,code)
        for cell in start['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                self.assertEqual(cell['outputs'], [])

    def test_setup_order_and_no_destructive_commands(self):
        notebook = json.loads((ROOT/'00_START_HERE_ROBOTICS_VSS.ipynb').read_text())
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        commands = ["script('deploy_lab.py', 'deploy')", "script('install_isaac.py')",
                    "script('validate_isaac_baseline.py')", "script('activate_isaac.py')",
                    "script('prepare_training_lab.py')", 'lab.before()', 'lab.train(',
                    'lab.after()', "script('activate_training_lab.py')"]
        positions = [code.index(command) for command in commands]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn('rmtree', code)
        self.assertNotIn('docker system prune', code)

    def test_activation_preserves_full_notebook(self):
        from workshop.scripts import activate_training_lab as activation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ui = root/'workshop/runtime/services/ui/workshop'
            ui.mkdir(parents=True)
            (ui/'index.html').write_text('old')
            (ui/'learn.html').write_text('learning')
            (root/'00_START_HERE_ROBOTICS_VSS.ipynb').write_text('full deployment notebook')
            (root/'workshop/training_profile.json').write_text('{"validation_status":"validated"}')
            report = root/'workshop/results/training/test/report.json'
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({'phase':'complete', 'comparison':{
                'weights_changed':True, 'same_initial_state':True, 'equal_control_budget':True}}))
            response = MagicMock()
            response.__enter__.return_value.read.return_value = b'{"busy":false,"csrf":"test"}'
            with patch.object(activation, 'ROOT', root), patch.object(activation.urllib.request, 'urlopen', return_value=response):
                activation.main()
            self.assertEqual((root/'00_START_HERE_ROBOTICS_VSS.ipynb').read_text(), 'full deployment notebook')
            self.assertEqual((ui/'index.html').read_text(), 'learning')


if __name__ == '__main__':
    unittest.main()
