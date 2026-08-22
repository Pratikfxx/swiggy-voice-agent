import importlib
import os
import sys
import unittest
import warnings
from unittest.mock import patch


def _fresh_agent():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ResourceWarning)
        sys.modules.pop("agent", None)
        return importlib.import_module("agent")


def _fresh_voice_handler():
    os.environ["TWILIO_VALIDATE_WEBHOOKS"] = "false"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ResourceWarning)
        for name in ("voice_handler", "agent"):
            sys.modules.pop(name, None)
        return importlib.import_module("voice_handler")


class InstamartOnlyTests(unittest.IsolatedAsyncioTestCase):
    def test_live_routing_is_instamart_only_for_now(self):
        agent = _fresh_agent()

        for surface in ("voice", "chat"):
            for message in ("one masala dosa", "book a table tonight", "milk and eggs"):
                self.assertEqual(
                    agent._route_servers(message, surface),
                    ["swiggy-instamart"],
                )

    def test_voice_prompt_is_instamart_only(self):
        agent = _fresh_agent()
        prompt = agent.VOICE_SYSTEM_PROMPT

        self.assertIn("Instamart-only", prompt)
        self.assertNotIn("Food:", prompt)
        self.assertNotIn("restaurant", prompt.lower())
        self.assertNotIn("dineout", prompt.lower())
        self.assertNotIn("book a table", prompt.lower())

    def test_voice_prompt_optimizes_for_low_friction_top_picks(self):
        agent = _fresh_agent()
        prompt = agent.VOICE_SYSTEM_PROMPT

        self.assertIn("Do not say \"I'll search\"", prompt)
        self.assertIn("pick clear top matches", prompt)
        self.assertIn("Add these?", prompt)

    def test_voice_prompt_does_not_checkout_on_the_first_yes(self):
        agent = _fresh_agent()
        prompt = agent.VOICE_SYSTEM_PROMPT
        self.assertIn("final cart, address, and payment summary", prompt)
        self.assertNotIn("On yes → checkout", prompt)
        self.assertIn("call get_cart, not search again", prompt)

    def test_chat_prompt_is_instamart_only(self):
        agent = _fresh_agent()
        prompt = agent.CHAT_SYSTEM_PROMPT

        self.assertIn("Instamart-only", prompt)
        self.assertNotIn("book a table", prompt.lower())
        self.assertNotIn("dineout", prompt.lower())
        self.assertNotIn("restaurant options", prompt.lower())

    def test_live_repeat_order_prompt_names_the_authoritative_tool(self):
        agent = _fresh_agent()
        for prompt in (agent.VOICE_SYSTEM_PROMPT, agent.CHAT_SYSTEM_PROMPT):
            self.assertIn("get_orders", prompt)

    async def test_voice_greeting_names_instamart(self):
        voice_handler = _fresh_voice_handler()

        class FakeRequest:
            async def form(self):
                return {"CallSid": "call-test"}

        with patch.object(voice_handler, "generate_tts_audio", return_value=None):
            response = await voice_handler.voice_answer(FakeRequest())

        twiml = response.body.decode()
        self.assertIn("Instamart", twiml)
        self.assertIn("groceries", twiml.lower())
        self.assertNotIn("What would you like to order?", twiml)


if __name__ == "__main__":
    unittest.main()


class AddressPromptTests(unittest.TestCase):
    """A caller asked to deliver to "Rithik" — a real saved address — and was
    told it was not saved. The model denied it without calling get_addresses."""

    def test_prompt_forbids_denying_an_address_without_looking(self):
        agent = _fresh_agent()
        rules = agent.ADDRESS_SELECTION_RULES
        self.assertIn("MUST call get_addresses", rules)
        self.assertIn("NEVER say an address is not saved", rules)

    def test_prompt_paginates_until_the_named_address_is_found(self):
        agent = _fresh_agent()
        self.assertIn("page=2", agent.ADDRESS_SELECTION_RULES)
        self.assertNotIn("takes no page argument", agent.ADDRESS_SELECTION_RULES)


class CartEditPromptTests(unittest.TestCase):
    """The agent said "Monster Energy Zero and Diet Coke, 205 rupees" after a
    removal request without calling any tool. The cart still held all six
    items. It had simply repeated its own earlier sentence."""

    def test_prompt_requires_a_tool_call_before_claiming_a_cart_change(self):
        agent = _fresh_agent()
        rules = agent.CART_EDIT_RULES
        self.assertIn("MUST call remove_from_cart", rules)
        self.assertIn("keep_only", rules)
        self.assertIn("in THIS turn", rules)

    def test_cart_rules_reach_the_live_system_prompt(self):
        agent = _fresh_agent()
        self.assertIn("CART EDITS", agent.LIVE_SYSTEM_SUFFIX)


class OrderPlacementPromptTests(unittest.TestCase):
    """A caller was told "Your order is already placed! ... arriving in 32
    minutes" in 3.5 seconds — no tool ran and Swiggy had zero orders. Two
    causes: checkout rejects a call with no payment method, and nothing
    forbade announcing an order that was never placed."""

    def test_payment_method_is_specified_for_voice(self):
        agent = _fresh_agent()
        rules = agent._payment_rules_for_surface("voice")
        self.assertIn("paymentMethod", rules)
        self.assertIn("Cash", rules)

    def test_announcing_an_unplaced_order_is_forbidden(self):
        agent = _fresh_agent()
        rules = agent.ORDER_PLACEMENT_RULES
        self.assertIn("NEVER say an order is placed", rules)
        self.assertIn("THIS turn", rules)

    def test_rules_reach_the_live_prompt(self):
        agent = _fresh_agent()
        self.assertIn("PLACING THE ORDER", agent.LIVE_SYSTEM_SUFFIX)

    def test_voice_prompt_closes_warmly_after_a_real_order(self):
        agent = _fresh_agent()
        self.assertIn("thank you", agent.VOICE_SYSTEM_PROMPT.lower())

    def test_prompt_requires_reading_back_the_whole_cart(self):
        agent = _fresh_agent()
        self.assertIn("BEFORE checkout", agent.ORDER_PLACEMENT_RULES)
        self.assertIn("cart_total", agent.ORDER_PLACEMENT_RULES)

    def test_cancellation_routes_to_customer_care_not_a_tool(self):
        """Swiggy's checkout contract states plainly: no tool can cancel an
        order, and the caller must be given the customer care number."""
        agent = _fresh_agent()
        for text in (agent.ORDER_PLACEMENT_RULES, agent.VOICE_SYSTEM_PROMPT):
            self.assertIn("080 67466729", text)
        self.assertIn("do NOT call any tool", agent.ORDER_PLACEMENT_RULES)

    def test_success_wording_follows_swiggy_branding(self):
        agent = _fresh_agent()
        self.assertIn("Instamart order placed successfully", agent.VOICE_SYSTEM_PROMPT)

    def test_voice_and_chat_have_different_payment_rules(self):
        agent = _fresh_agent()
        self.assertIn("cash on delivery", agent._payment_rules_for_surface("voice").lower())
        self.assertIn("get_payment_options", agent._payment_rules_for_surface("chat"))
        self.assertNotIn("Use cash on delivery", agent._payment_rules_for_surface("chat"))

    def test_live_mode_uses_swiggys_order_history_not_the_local_copy(self):
        agent = _fresh_agent()
        self.assertNotIn("get_order_history", agent.LOCAL_NAMES)
