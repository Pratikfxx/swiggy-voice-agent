"""Shared Swiggy MCP product scope.

Imports nothing from the project on purpose: everything else may import this,
so it must stay at the bottom of the dependency graph.

The product is currently live for Instamart only. Keep this small module as the
single place that declares which Swiggy MCP servers are active for runtime
readiness and token loading.
"""

import os

SWIGGY_SERVER_URLS = {
    "swiggy-food": "https://mcp.swiggy.com/food",
    "swiggy-instamart": "https://mcp.swiggy.com/im",
    "swiggy-dineout": "https://mcp.swiggy.com/dineout",
}

SERVER_AUTH_KEYS = {
    "swiggy-food": "food",
    "swiggy-instamart": "im",
    "swiggy-dineout": "dineout",
}

ACTIVE_SWIGGY_SERVERS = ("swiggy-instamart",)
ACTIVE_TOKEN_KEYS = tuple(SERVER_AUTH_KEYS[name] for name in ACTIVE_SWIGGY_SERVERS)


# Demo mode serves mock catalogue data instead of calling Swiggy. It lives here
# because two modules need it and they used to disagree: agent.py defaulted to
# live while swiggy_tools.py defaulted to mock, so an unset DEMO_MODE gave a
# live agent backed by fake products.
def demo_mode() -> bool:
    return os.getenv("DEMO_MODE", "false").lower() == "true"
