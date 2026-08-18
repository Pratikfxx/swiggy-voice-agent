import main


def test_demo_page_describes_instamart_ordering_only():
    page = main.demo_page()

    # Assert the scope the page claims, not its exact marketing copy — the
    # wording is expected to change, the Instamart-only scope is not.
    assert "Instamart" in main.app.description
    assert "Instamart" in page

    # The agent refuses cooked meals and reservations, so the page must not
    # advertise them. This drifted once already: the page demoed a restaurant
    # biryani order the agent would have declined.
    for out_of_scope in ("Biryani Blues", "chicken biryani", "restaurant", "book a table"):
        assert out_of_scope.lower() not in page.lower(), out_of_scope
