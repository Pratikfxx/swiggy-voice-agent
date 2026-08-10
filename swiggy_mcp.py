import inspect
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

import httpx

try:
    from mcp.client import streamable_http as _streamable_http
except ImportError as exc:
    try:
        _MCP_VERSION = version("mcp")
    except PackageNotFoundError:
        _MCP_VERSION = "unknown"
    raise ImportError(
        f"Incompatible mcp version {_MCP_VERSION!r}: "
        "mcp.client.streamable_http is unavailable"
    ) from exc


def _resolve_transport(module):
    try:
        mcp_version = version("mcp")
    except PackageNotFoundError:
        mcp_version = "unknown"

    for name in ("streamable_http_client", "streamablehttp_client"):
        transport = getattr(module, name, None)
        if transport is None:
            continue
        try:
            signature = inspect.signature(transport)
        except (TypeError, ValueError):
            continue
        for parameter in ("http_client", "headers"):
            try:
                signature.bind("https://mcp.invalid", **{parameter: object()})
            except TypeError:
                continue
            return transport, name, parameter

    raise ImportError(
        f"Incompatible mcp version {mcp_version!r}: "
        "streamable HTTP client must accept http_client= or headers="
    )


_TRANSPORT, _TRANSPORT_NAME, _TRANSPORT_PARAMETER = _resolve_transport(
    _streamable_http
)


@asynccontextmanager
async def _open_authenticated_mcp(url, token, transport, parameter):
    headers = {"Authorization": f"Bearer {token}"}
    if parameter == "http_client":
        client = httpx.AsyncClient(headers=headers)
        try:
            async with transport(url, http_client=client) as streams:
                yield streams
        finally:
            await client.aclose()
    else:
        async with transport(url, headers=headers) as streams:
            yield streams


@asynccontextmanager
async def open_authenticated_mcp(url, token):
    async with _open_authenticated_mcp(
        url, token, _TRANSPORT, _TRANSPORT_PARAMETER
    ) as streams:
        if len(streams) == 3:
            read, write, _ = streams
        else:
            read, write = streams
        yield read, write
