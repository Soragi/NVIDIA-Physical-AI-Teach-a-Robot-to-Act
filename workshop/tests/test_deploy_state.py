"""Regression: a second deploy must reuse the final private-env assignment."""
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

class SavedCredential(unittest.TestCase):
    def test_last_assignment_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            state=Path(directory)/'vss-robotics-workshop'
            state.mkdir()
            (state/'base.env').write_text('NGC_CLI_API_KEY=\nNGC_CLI_API_KEY=dummy-test-value\n')
            script=Path(__file__).resolve().parents[1]/'scripts/deploy_lab.py'
            with patch.dict(os.environ,{'XDG_STATE_HOME':directory},clear=True), \
                 patch.object(sys,'argv',[str(script),'deploy']), \
                 patch('subprocess.run') as run:
                run.return_value.returncode=0
                with self.assertRaises(SystemExit) as exit_result:
                    runpy.run_path(str(script),run_name='__main__')
                self.assertEqual(exit_result.exception.code,0)
                self.assertEqual(run.call_args.kwargs['env']['NGC_API_KEY'],'dummy-test-value')

if __name__=='__main__':unittest.main()
