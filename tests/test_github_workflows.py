from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GitHubWorkflowTests(unittest.TestCase):
    def test_swiggy_token_health_workflow_template_runs_public_monitor(self):
        workflow = (
            ROOT / "ops" / "swiggy-token-health.workflow.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("cron:", workflow)
        self.assertIn("SWIGGY_HEALTH_URL:", workflow)
        self.assertIn(
            "https://pure-adventure-production-3bb1.up.railway.app/health",
            workflow,
        )
        self.assertIn("scripts/check_swiggy_token_health.py", workflow)
        self.assertNotIn("SWIGGY_FOOD_TOKEN", workflow)
        self.assertNotIn("ANTHROPIC_API_KEY", workflow)


if __name__ == "__main__":
    unittest.main()
