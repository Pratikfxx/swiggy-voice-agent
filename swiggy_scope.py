"""Shared Swiggy MCP product scope.

The product is currently live for Instamart only. Keep this small module as the
single place that declares which Swiggy MCP servers are active for runtime
readiness and token loading.
"""

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
