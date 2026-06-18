import io
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts import check_swiggy_token_health


ROOT = Path(__file__).resolve().parents[1]


class TokenHealthMonitorTests(unittest.TestCase):
    def test_returns_success_when_all_tokens_are_above_threshold(self):
        payload = {
            "swiggy_tokens": {
                "food": {"logged_in": True, "expired": False, "expires_in_s": 90_000},
                "im": {"logged_in": True, "expired": False, "expires_in_s": 90_000},
                "dineout": {"logged_in": True, "expired": False, "expires_in_s": 90_000},
            }
        }

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = check_swiggy_token_health.evaluate_health(payload, warn_seconds=86_400)

        self.assertEqual(exit_code, 0)
        self.assertIn("food: ok", stdout.getvalue())
        self.assertNotIn("TOKEN", stdout.getvalue())

    def test_warns_when_any_token_is_below_threshold(self):
        payload = {
            "swiggy_tokens": {
                "food": {"logged_in": True, "expired": False, "expires_in_s": 3_600},
                "im": {"logged_in": True, "expired": False, "expires_in_s": 90_000},
                "dineout": {"logged_in": True, "expired": False, "expires_in_s": 90_000},
            }
        }

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = check_swiggy_token_health.evaluate_health(payload, warn_seconds=86_400)

        self.assertEqual(exit_code, 1)
        self.assertIn("food: expiring_soon", stdout.getvalue())

    def test_fails_when_any_token_is_expired_or_missing(self):
        payload = {
            "swiggy_tokens": {
                "food": {"logged_in": True, "expired": True, "expires_in_s": 0},
                "im": {"logged_in": False, "expired": True, "expires_in_s": None},
                "dineout": {"logged_in": True, "expired": False, "expires_in_s": 90_000},
            }
        }

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = check_swiggy_token_health.evaluate_health(payload, warn_seconds=86_400)

        self.assertEqual(exit_code, 2)
        self.assertIn("food: failed", stdout.getvalue())
        self.assertIn("im: failed", stdout.getvalue())

    def test_script_path_invocation_can_import_repo_modules(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/check_swiggy_token_health.py",
                "--warn-hours",
                "24",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotIn("ModuleNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main()
