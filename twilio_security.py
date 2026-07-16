"""Twilio webhook signature validation.

The voice and WhatsApp webhooks drive an agent holding live Swiggy tokens for
a single account — an unauthenticated POST that imitates Twilio could place
real orders on that account. Twilio signs every webhook request with
X-Twilio-Signature (HMAC-SHA1 over the exact public URL + sorted POST params,
keyed by the account auth token); reject anything that fails that check.

Escape hatch: set TWILIO_VALIDATE_WEBHOOKS=false if a proxy mangles the
forwarded headers and legitimate traffic starts being rejected. Validation is
also skipped when TWILIO_AUTH_TOKEN is unset (local demo without Twilio).
"""

import logging
import os

from twilio.request_validator import RequestValidator

logger = logging.getLogger("uvicorn.error")

_validator: RequestValidator | None = None
_validator_token = ""


def _get_validator() -> RequestValidator | None:
    global _validator, _validator_token
    token = os.getenv("TWILIO_AUTH_TOKEN", "")
    if not token:
        return None
    if _validator is None or token != _validator_token:
        _validator = RequestValidator(token)
        _validator_token = token
    return _validator


def _public_url(request) -> str:
    """The URL Twilio actually requested — signature input must match exactly.

    Behind Railway/tunnels the ASGI request URL is the internal one; the
    forwarded headers carry the public scheme/host Twilio dialed.
    """
    headers = request.headers
    host = (headers.get("x-forwarded-host") or headers.get("host") or request.url.netloc)
    host = host.split(",")[0].strip()
    scheme = (headers.get("x-forwarded-proto") or request.url.scheme or "https")
    scheme = scheme.split(",")[0].strip()
    url = f"{scheme}://{host}{request.url.path}"
    if request.url.query:
        url += f"?{request.url.query}"
    return url


def verify_twilio_request(request, form: dict) -> bool:
    """True if the request is genuinely from Twilio (or validation is off)."""
    if os.getenv("TWILIO_VALIDATE_WEBHOOKS", "true").lower() == "false":
        return True

    validator = _get_validator()
    if validator is None:
        return True

    try:
        signature = request.headers.get("x-twilio-signature", "")
        url = _public_url(request)
        ok = validator.validate(url, dict(form), signature)
    except Exception:
        logger.exception("Twilio signature validation errored; rejecting request")
        return False

    if not ok:
        logger.warning(
            "Rejected webhook with bad Twilio signature url=%s sig_present=%s",
            url, bool(signature),
        )
    return ok
