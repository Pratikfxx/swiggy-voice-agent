from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
SECRET_NAMES = (
    "ANTHROPIC_API_KEY",
    "ELEVENLABS_API_KEY",
    "SWIGGY_FOOD_TOKEN",
    "SWIGGY_IM_TOKEN",
    "SWIGGY_DINEOUT_TOKEN",
    "TWILIO_AUTH_TOKEN",
)


class DeployConfigTests(unittest.TestCase):
    def test_railway_uses_custom_dockerfile_builder(self):
        config = (ROOT / "railway.toml").read_text(encoding="utf-8")

        self.assertIn('builder = "DOCKERFILE"', config)
        self.assertIn('dockerfilePath = "Dockerfile"', config)

    def test_dockerfile_does_not_declare_runtime_secrets(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        for secret_name in SECRET_NAMES:
            pattern = rf"^\s*(ARG|ENV)\s+{re.escape(secret_name)}\b"
            self.assertIsNone(
                re.search(pattern, dockerfile, flags=re.MULTILINE),
                f"Dockerfile must not declare {secret_name} as ARG or ENV",
            )

    def test_docker_context_excludes_local_secrets_and_agent_state(self):
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        ignored = {
            line.strip()
            for line in dockerignore.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

        for expected in (".env", ".env.*", ".swiggy_tokens.json", ".codex/"):
            self.assertIn(expected, ignored)


if __name__ == "__main__":
    unittest.main()
