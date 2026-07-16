from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GitHubWorkflowTests(unittest.TestCase):
    def test_swiggy_token_health_workflow_is_active_and_scheduled(self):
        """Must live under .github/workflows/ — a copy in ops/ never runs."""
        workflow = (
            ROOT / ".github" / "workflows" / "swiggy-token-health.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("cron:", workflow)
        self.assertIn("workflow_dispatch:", workflow)
        # URL comes from a repo variable — a hardcoded deployment URL goes
        # stale silently (the original hardcoded Railway app 404s today).
        self.assertIn("${{ vars.SWIGGY_HEALTH_URL }}", workflow)
        self.assertIn("scripts/check_swiggy_token_health.py", workflow)
        # The monitor only reads the public /health endpoint — no secrets.
        self.assertNotIn("SWIGGY_FOOD_TOKEN", workflow)
        self.assertNotIn("SWIGGY_IM_TOKEN", workflow)
        self.assertNotIn("ANTHROPIC_API_KEY", workflow)

    def test_no_stale_workflow_copy_outside_github_dir(self):
        self.assertFalse(
            (ROOT / "ops" / "swiggy-token-health.workflow.yml").exists(),
            "duplicate workflow template in ops/ drifts from the active one",
        )


if __name__ == "__main__":
    unittest.main()
