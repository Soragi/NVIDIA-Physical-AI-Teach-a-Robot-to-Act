"""Notebook users should see actionable local API conflicts."""
import io
import unittest
import urllib.error
from unittest.mock import patch
from workshop.lab.training import TrainingMission

class NotebookErrors(unittest.TestCase):
    def test_conflict_explains_required_next_step(self):
        client=TrainingMission()
        error=urllib.error.HTTPError(client.url,409,'Conflict',{},
            io.BytesIO(b'{"error":"Run the early checkpoint first."}'))
        with patch.object(client,'status',return_value={'csrf':'test'}), patch('urllib.request.urlopen',side_effect=error):
            with self.assertRaisesRegex(RuntimeError,'Run the early checkpoint first'):
                client.train()

    def test_server_error_does_not_echo_response_body(self):
        client=TrainingMission()
        error=urllib.error.HTTPError(client.url,500,'Error',{},io.BytesIO(b'private upstream detail'))
        with patch.object(client,'status',return_value={'csrf':'test'}), patch('urllib.request.urlopen',side_effect=error):
            with self.assertRaisesRegex(RuntimeError,'Mission API HTTP 500') as caught:
                client.train()
            self.assertNotIn('private',str(caught.exception))
