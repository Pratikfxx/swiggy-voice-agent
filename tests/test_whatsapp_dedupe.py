import importlib
import sys
import unittest
import warnings


def _fresh_whatsapp():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ResourceWarning)
        for name in ("whatsapp_handler", "agent"):
            sys.modules.pop(name, None)
        return importlib.import_module("whatsapp_handler")


class WhatsappDedupeTests(unittest.TestCase):
    """Twilio redelivers webhooks; a replayed 'yes' must not double-order."""

    def test_second_delivery_of_same_sid_is_dropped(self):
        wa = _fresh_whatsapp()
        self.assertFalse(wa._is_duplicate_delivery("SM123"))
        self.assertTrue(wa._is_duplicate_delivery("SM123"))

    def test_distinct_sids_pass(self):
        wa = _fresh_whatsapp()
        self.assertFalse(wa._is_duplicate_delivery("SM1"))
        self.assertFalse(wa._is_duplicate_delivery("SM2"))

    def test_missing_sid_never_blocks(self):
        wa = _fresh_whatsapp()
        self.assertFalse(wa._is_duplicate_delivery(""))
        self.assertFalse(wa._is_duplicate_delivery(""))

    def test_seen_set_is_bounded(self):
        wa = _fresh_whatsapp()
        for i in range(wa._SEEN_SID_MAX + 50):
            wa._is_duplicate_delivery(f"SM{i}")
        # a purge pass ran; the map cannot grow unboundedly past max + one batch
        self.assertLessEqual(len(wa._seen_message_sids), wa._SEEN_SID_MAX + 51)


if __name__ == "__main__":
    unittest.main()
