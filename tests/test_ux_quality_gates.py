import unittest

from scripts.ux_quality_gates import (
    check_one_command_fix_messaging,
    check_no_manual_json_edits_for_top_integrations,
    check_scriptable_onboarding_flow,
)


class TestUxQualityGates(unittest.TestCase):
    def test_scriptable_onboarding_flow(self):
        ok, detail = check_scriptable_onboarding_flow()
        self.assertTrue(ok, detail)

    def test_one_command_fix_messaging(self):
        ok, detail = check_one_command_fix_messaging()
        self.assertTrue(ok, detail)

    def test_no_manual_json_edits_for_top_integrations(self):
        ok, detail = check_no_manual_json_edits_for_top_integrations()
        self.assertTrue(ok, detail)


if __name__ == "__main__":
    unittest.main()
