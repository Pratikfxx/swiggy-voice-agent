from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swiggy_scope import ACTIVE_TOKEN_KEYS


EXPECTED_KEYS = ACTIVE_TOKEN_KEYS


def _format_duration(seconds: int | None) -> str:
    if seconds is None:
        return "unknown"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours}h{minutes:02d}m"


def evaluate_health(
    payload: dict[str, Any],
    warn_seconds: int,
    expected_keys: tuple[str, ...] | list[str] = EXPECTED_KEYS,
) -> int:
    tokens = payload.get("swiggy_tokens")
    if not isinstance(tokens, dict):
        print("swiggy_tokens: failed missing")
        return 2

    exit_code = 0
    for key in expected_keys:
        info = tokens.get(key)
        if not isinstance(info, dict):
            print(f"{key}: failed missing")
            exit_code = max(exit_code, 2)
            continue

        logged_in = bool(info.get("logged_in"))
        expired = bool(info.get("expired"))
        expires_in_s = info.get("expires_in_s")
        if not isinstance(expires_in_s, int):
            expires_in_s = None

        if not logged_in or expired:
            print(f"{key}: failed expires_in={_format_duration(expires_in_s)}")
            exit_code = max(exit_code, 2)
        elif expires_in_s is None:
            print(f"{key}: warning expires_in=unknown")
            exit_code = max(exit_code, 1)
        elif expires_in_s < warn_seconds:
            print(f"{key}: expiring_soon expires_in={_format_duration(expires_in_s)}")
            exit_code = max(exit_code, 1)
        else:
            print(f"{key}: ok expires_in={_format_duration(expires_in_s)}")

    return exit_code


def _load_remote_health(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_local_health() -> dict[str, Any]:
    import swiggy_auth

    return {"swiggy_tokens": swiggy_auth.status()}


def _parse_keys(raw: str) -> tuple[str, ...]:
    keys = tuple(key.strip() for key in raw.split(",") if key.strip())
    if not keys:
        raise ValueError("At least one Swiggy token key is required")
    return keys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Swiggy token expiry safely")
    parser.add_argument(
        "--url",
        default=os.getenv("SWIGGY_HEALTH_URL", ""),
        help="Health endpoint URL. Defaults to local token status when omitted.",
    )
    parser.add_argument(
        "--warn-hours",
        type=float,
        default=float(os.getenv("SWIGGY_TOKEN_WARN_HOURS", "24")),
        help="Return warning when any token expires within this many hours.",
    )
    parser.add_argument(
        "--keys",
        default=os.getenv("SWIGGY_TOKEN_KEYS", ",".join(EXPECTED_KEYS)),
        help="Comma-separated Swiggy token keys to require. Defaults to active product scope.",
    )
    args = parser.parse_args(argv)

    warn_seconds = int(args.warn_hours * 3600)
    expected_keys = _parse_keys(args.keys)
    payload = _load_remote_health(args.url) if args.url else _load_local_health()
    return evaluate_health(payload, warn_seconds=warn_seconds, expected_keys=expected_keys)


if __name__ == "__main__":
    raise SystemExit(main())
