from pathlib import Path
import json
import tempfile
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]


class PublicReleaseTests(unittest.TestCase):
    def test_only_the_supported_attendee_entry_point_is_shipped(self):
        self.assertTrue((ROOT / "00_START_HERE_ROBOTICS_VSS.ipynb").is_file())
        self.assertFalse((ROOT / "01_LEARN_TO_ACT.ipynb").exists())
        self.assertFalse((ROOT / "02_RECOVERY_MISSION_CONTROL.ipynb").exists())

    def test_source_tree_contains_no_instance_specific_identity(self):
        forbidden = (
            b"video-search-and-summarization-blueprint-" + b"b769be",
            b"/Users/" + b"isoltysik",
            b"/home/ubuntu/" + b"vss-robotics-workshop",
        )
        # Scan only the published source, never downloaded models, videos or caches.
        tracked = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
        for name in filter(None,tracked):
            path = ROOT/name
            if not path.is_file():
                continue
            value = path.read_bytes()
            for marker in forbidden:
                self.assertNotIn(marker, value, str(path.relative_to(ROOT)))

    def test_known_management_ports_default_to_loopback(self):
        env = (ROOT / "workshop/runtime/developer-profiles/dev-profile-base/.env").read_text()
        self.assertIn("HOST_IP=127.0.0.1", env)
        self.assertIn("HAPROXY_BIND_ADDR=0.0.0.0", env)  # Brev-facing gateway only
        redis = (ROOT / "workshop/runtime/services/infra/redis/configs/redis.conf").read_text()
        self.assertIn("bind 127.0.0.1 ::1", redis)
        self.assertIn("protected-mode yes", redis)
        nginx = (ROOT / "workshop/runtime/services/ui/workshop/nginx.conf").read_text()
        self.assertIn("listen 127.0.0.1:3100", nginx)

    def test_isaac_processes_do_not_inherit_jupyter_inline_backend(self):
        for name in ("validate_isaac_baseline.py", "isaac_worker.py", "evaluate_training.py"):
            source = (ROOT / "workshop/scripts" / name).read_text()
            self.assertIn("MPLBACKEND='Agg'", source, name)

    def test_fresh_deployment_retries_transient_registry_token_failures(self):
        source = (ROOT / "workshop/scripts/deploy_vss_base.sh").read_text()
        self.assertIn("compose config --images | sort -u", source)
        self.assertIn('for attempt in 1 2 3 4 5 6', source)
        self.assertIn('docker pull "$image"', source)

    def test_baseline_evidence_requires_three_boolean_outcomes(self):
        from workshop.scripts.validate_isaac_baseline import complete_evidence
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            report=root/'run/episode_results_rank0.jsonl'
            report.parent.mkdir()
            report.write_text('\n'.join(json.dumps({'success':value}) for value in (True,False)))
            self.assertEqual(complete_evidence(root),[])
            report.write_text('\n'.join(json.dumps({'success':value}) for value in (True,False,True)))
            self.assertEqual(len(complete_evidence(root)),1)


if __name__ == "__main__":
    unittest.main()
