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

    def test_dockerfile_deploy_uses_docker_cmd_for_port_expansion(self):
        config = (ROOT / "railway.toml").read_text(encoding="utf-8")

        self.assertNotIn("startCommand", config)
        self.assertIn("${PORT:-8000}", (ROOT / "Dockerfile").read_text(encoding="utf-8"))

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


def test_demo_mode_has_exactly_one_default():
    """agent.py and swiggy_tools.py used to disagree — live agent, mock data."""
    import pathlib, re

    sources = [p for p in pathlib.Path(".").glob("*.py")]
    offenders = []
    for path in sources:
        for match in re.finditer(r'getenv\(\s*["\']DEMO_MODE["\']\s*,\s*["\']([^"\']+)["\']', path.read_text()):
            offenders.append(f"{path.name}:{match.group(1)}")
    assert len(offenders) <= 1, f"DEMO_MODE default defined in several places: {offenders}"


def test_swiggy_server_urls_live_in_one_place():
    """swiggy_auth.py used to repeat the URLs that swiggy_scope.py declares."""
    import pathlib

    hits = []
    for path in pathlib.Path(".").glob("*.py"):
        if path.name == "swiggy_scope.py":
            continue
        text = path.read_text()
        if "https://mcp.swiggy.com/im" in text or "https://mcp.swiggy.com/food" in text:
            hits.append(path.name)
    assert hits == [], f"Swiggy MCP server URLs duplicated in: {hits}"


def test_agent_and_tools_agree_on_demo_mode(monkeypatch):
    import importlib

    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    import agent, swiggy_tools

    importlib.reload(swiggy_tools)
    importlib.reload(agent)
    assert agent.DEMO_MODE == swiggy_tools.DEMO_MODE
