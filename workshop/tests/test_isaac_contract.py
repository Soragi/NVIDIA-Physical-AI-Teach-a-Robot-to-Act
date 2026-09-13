import unittest
from workshop.lab.isaac_contract import scenario, validate_decision, approve, visual_assessment


class IsaacContractTests(unittest.TestCase):
    def test_visual_judgment_is_separate_from_simulator_truth(self):
        self.assertTrue(visual_assessment('complete',False)[1])
        self.assertFalse(visual_assessment('incomplete',False)[1])
        self.assertIsNone(visual_assessment('uncertain',False)[1])
        self.assertIsNone(visual_assessment('The robot might have completed it.',True)[1])
        self.assertTrue(visual_assessment('```json\n{"verdict":"incomplete"}\n```',True)[1])
    def test_seed_reproduces_scenario(self):
        self.assertEqual(scenario(42),scenario(42))
        self.assertNotEqual(scenario(42),scenario(43))

    def test_invalid_seeds(self):
        for value in (True,-1,2**31,'42',1.5):
            with self.assertRaises(ValueError):
                scenario(value)

    def test_equal_budget_and_pregrasp_disturbance(self):
        for seed in range(100):
            spec = scenario(seed)
            self.assertEqual(spec['step_budget'],500)
            self.assertLess(spec['disturbance_step'],50)
            self.assertLessEqual(abs(spec['disturbance_m'][0]),.04)

    def test_arbitrary_motion_rejected(self):
        with self.assertRaises(ValueError):
            validate_decision({'action':'move_joint','reason':'test','uncertainty':'none'}, {})

    def test_unknown_fields_rejected(self):
        with self.assertRaises(ValueError):
            validate_decision({'action':'continue','reason':'test','uncertainty':'none','joint':1}, {})

    def test_approval_binds_every_revision(self):
        binding={'run_id':'abc','scene_revision':1,'evidence_sha256':'sha','decision_id':'decision'}
        decision=validate_decision({'action':'reobserve_short','reason':'target changed','uncertainty':'limited view'},binding)
        self.assertEqual(approve(decision,{**binding,'approve':True}),'reobserve_short')
        for key in binding:
            with self.assertRaises(ValueError):
                approve(decision,{**binding,key:'stale','approve':True})
        with self.assertRaises(ValueError):
            approve(decision,{**binding,'approve':False})
