import asyncio
import importlib
import json
import sys
from unittest.mock import AsyncMock, patch


def _fresh_agent(monkeypatch, tmp_path):
    monkeypatch.setenv("ORDER_HISTORY_DB", str(tmp_path / "orders.db"))
    for name in ("store", "order_history", "agent"):
        sys.modules.pop(name, None)
    return importlib.import_module("agent")


def _fresh_voice(monkeypatch, tmp_path):
    monkeypatch.setenv("ORDER_HISTORY_DB", str(tmp_path / "orders.db"))
    for name in ("store", "order_history", "agent", "voice_handler"):
        sys.modules.pop(name, None)
    return importlib.import_module("voice_handler")


def _fresh_whatsapp(monkeypatch, tmp_path):
    monkeypatch.setenv("ORDER_HISTORY_DB", str(tmp_path / "orders.db"))
    for name in ("store", "order_history", "agent", "whatsapp_handler"):
        sys.modules.pop(name, None)
    return importlib.import_module("whatsapp_handler")


def test_order_history_crosses_voice_calls_for_same_caller(monkeypatch, tmp_path):
    agent = _fresh_agent(monkeypatch, tmp_path)
    caller = "+919876543210"

    with patch.object(agent, "place_instamart_order_mock", return_value={"success": True}):
        agent.execute_tool(
            "place_grocery_order",
            {"items": [{"name": "Amul milk"}]},
            session_id="CA-first-call",
            user_id=caller,
        )

    result = json.loads(
        agent.execute_tool(
            "get_order_history",
            {},
            session_id="CA-second-call",
            user_id=caller,
        )
    )

    assert result["orders"][0]["items"] == [{"name": "Amul milk"}]


def test_whatsapp_order_is_visible_to_normalized_voice_identity(monkeypatch, tmp_path):
    whatsapp = _fresh_whatsapp(monkeypatch, tmp_path)
    agent = importlib.import_module("agent")
    caller = "+919876543210"

    def fake_process_message(**kwargs):
        assert kwargs["session_id"] == f"whatsapp:{caller}"
        assert kwargs["user_id"] == caller
        with patch.object(agent, "place_instamart_order_mock", return_value={"success": True}):
            agent.execute_tool(
                "place_grocery_order",
                {"items": [{"name": "Brown bread"}]},
                session_id=kwargs["session_id"],
                user_id=kwargs["user_id"],
            )
        return "saved"

    async def run():
        with (
            patch.object(whatsapp, "process_message", side_effect=fake_process_message),
            patch.object(whatsapp, "_send", new=AsyncMock()),
        ):
            await whatsapp._handle_incoming_inner(
                f"whatsapp:{caller}", "order bread", 0, "", ""
            )

    asyncio.run(run())

    result = json.loads(
        agent.execute_tool(
            "get_order_history",
            {},
            session_id="CA-voice-call",
            user_id=caller,
        )
    )
    assert result["orders"][0]["items"] == [{"name": "Brown bread"}]


def test_gather_hints_include_recent_products(monkeypatch, tmp_path):
    voice = _fresh_voice(monkeypatch, tmp_path)
    orders = [
        {"items": [{"name": "Amla juice"}, {"name": "Milk"}]},
        {"items": [{"name": "amla JUICE"}, {"name": "Brown bread"}]},
    ]

    async def run():
        with (
            patch.object(voice, "get_recent_orders", return_value=orders) as recent,
            patch.object(voice, "generate_tts_audio", return_value=None),
        ):
            return await voice.make_twiml_response(
                "What would you like?", session_id="CA-call", user_id="+919876543210"
            ), recent

    twiml, recent = asyncio.run(run())

    recent.assert_called_once_with("+919876543210", limit=5)
    assert twiml.count("Amla juice") == 2
    assert twiml.count("amla JUICE") == 0
    assert twiml.count("Brown bread") == 2


def test_gather_hints_fall_back_when_order_history_fails(monkeypatch, tmp_path):
    voice = _fresh_voice(monkeypatch, tmp_path)

    async def run():
        with (
            patch.object(voice, "get_recent_orders", side_effect=OSError("database down")),
            patch.object(voice, "generate_tts_audio", return_value=None),
        ):
            return await voice.make_twiml_response(
                "What would you like?", session_id="CA-call", user_id="+919876543210"
            )

    twiml = asyncio.run(run())

    assert voice.SPEECH_HINTS in twiml
    assert "database down" not in twiml
