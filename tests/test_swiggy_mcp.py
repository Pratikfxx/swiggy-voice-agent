import asyncio
import inspect
import types
from contextlib import asynccontextmanager

import httpx

import swiggy_mcp


def test_helper_arguments_bind_against_real_installed_mcp_signature():
    transport = getattr(swiggy_mcp._streamable_http, swiggy_mcp._TRANSPORT_NAME)
    signature = inspect.signature(transport)
    if swiggy_mcp._TRANSPORT_PARAMETER == "http_client":
        client = httpx.AsyncClient(headers={"Authorization": "Bearer token"})
        try:
            signature.bind("https://mcp.invalid", http_client=client)
        finally:
            asyncio.run(client.aclose())
    else:
        signature.bind(
            "https://mcp.invalid", headers={"Authorization": "Bearer token"}
        )


def test_old_headers_api_works(monkeypatch):
    calls = []

    @asynccontextmanager
    async def streamablehttp_client(url, *, headers):
        calls.append((url, headers))
        yield ("read", "write", None)

    old_module = types.SimpleNamespace(streamablehttp_client=streamablehttp_client)
    transport, _, parameter = swiggy_mcp._resolve_transport(old_module)
    monkeypatch.setattr(swiggy_mcp, "_TRANSPORT", transport)
    monkeypatch.setattr(swiggy_mcp, "_TRANSPORT_PARAMETER", parameter)

    async def run():
        async with swiggy_mcp.open_authenticated_mcp(
            "https://mcp.invalid", "old-token"
        ) as streams:
            assert streams == ("read", "write")

    asyncio.run(run())
    assert calls == [
        (
            "https://mcp.invalid",
            {"Authorization": "Bearer old-token"},
        )
    ]


def test_new_http_client_is_closed_when_context_exits(monkeypatch):
    clients = []

    @asynccontextmanager
    async def streamable_http_client(url, *, http_client=None):
        clients.append(http_client)
        yield ("read", "write")

    new_module = types.SimpleNamespace(streamable_http_client=streamable_http_client)
    transport, _, parameter = swiggy_mcp._resolve_transport(new_module)
    monkeypatch.setattr(swiggy_mcp, "_TRANSPORT", transport)
    monkeypatch.setattr(swiggy_mcp, "_TRANSPORT_PARAMETER", parameter)

    async def run():
        async with swiggy_mcp.open_authenticated_mcp(
            "https://mcp.invalid", "new-token"
        ) as streams:
            assert streams == ("read", "write")
            assert not clients[0].is_closed

    asyncio.run(run())
    assert clients[0].is_closed
