"""Print every tool a Swiggy MCP server exposes, with its full instructions.

Swiggy ships behavioural rules inside the tool descriptions themselves — the
payment contract, the branding wording, the "no tool can cancel an order" line
— so this is the authoritative source, not any documentation.

    python scripts/list_swiggy_tools.py            # instamart, names + sizes
    python scripts/list_swiggy_tools.py --full     # complete descriptions
    python scripts/list_swiggy_tools.py --server food --full
    python scripts/list_swiggy_tools.py --tool checkout
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp import ClientSession

from swiggy_auth import get_access_token
from swiggy_mcp import open_authenticated_mcp
from swiggy_scope import SERVER_AUTH_KEYS, SWIGGY_SERVER_URLS


async def _tools(server: str):
    name = f"swiggy-{server}"
    url = SWIGGY_SERVER_URLS[name]
    token = get_access_token(SERVER_AUTH_KEYS[name])
    async with open_authenticated_mcp(url, token) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return (await session.list_tools()).tools


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="instamart", choices=["food", "instamart", "dineout"])
    parser.add_argument("--tool", help="show only this tool, in full")
    parser.add_argument("--full", action="store_true", help="show full descriptions")
    args = parser.parse_args()

    try:
        tools = asyncio.run(_tools(args.server))
    except Exception as error:
        print(f"could not reach swiggy-{args.server}: {error}")
        print("If the token expired: python swiggy_auth.py login "
              f"{SERVER_AUTH_KEYS['swiggy-' + args.server]}")
        return 1

    if args.tool:
        tools = [t for t in tools if t.name == args.tool]
        if not tools:
            print(f"no tool named {args.tool!r}")
            return 1
        args.full = True

    print(f"{len(tools)} tools on swiggy-{args.server}\n")
    for tool in tools:
        description = tool.description or ""
        schema = json.dumps(getattr(tool, "inputSchema", {}) or {})
        if args.full:
            print("=" * 72)
            print(tool.name)
            print("=" * 72)
            print(description.strip() or "(no description)")
            print(f"\ninput schema: {schema}\n")
        else:
            approx = (len(tool.name) + len(description) + len(schema)) // 4
            first = description.strip().splitlines()[0] if description.strip() else ""
            print(f"  {tool.name:24s} ~{approx:5d} tok  {first[:64]}")

    if not args.full:
        print("\nRun with --full, or --tool <name>, to read the instructions "
              "Swiggy embeds in each description.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
