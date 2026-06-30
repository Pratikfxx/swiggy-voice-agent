# Live Instamart Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make live-mode readiness and token loading match the current Instamart-only product scope, while preventing production from silently falling back to demo ordering.

**Architecture:** Add a tiny shared scope module that declares the active Swiggy server and token key. Keep existing large modules intact, but route token loading, health readiness, and monitor checks through that active scope.

**Tech Stack:** Python 3.12, FastAPI, Anthropic SDK MCP client, unittest.

---

### Task 1: Add active-scope tests

**Files:**
- Modify: `tests/test_swiggy_auth.py`
- Modify: `tests/test_health.py`
- Modify: `tests/test_token_health_monitor.py`
- Modify: `tests/test_agent_timeouts.py`

- [ ] **Step 1: Write failing token-scope tests**

Add tests proving `swiggy_auth.get_access_tokens(("im",))` succeeds with only the Instamart token and does not require food or dineout.

- [ ] **Step 2: Write failing health and monitor tests**

Add tests proving `/health` and `scripts/check_swiggy_token_health.py` treat Instamart as the required live token set by default, while the monitor can still be pointed at all keys explicitly.

- [ ] **Step 3: Write failing live-agent tests**

Add tests proving live mode calls the live runner with only the Instamart token when it exists, and returns an auth-not-ready response when the Instamart token is missing instead of falling back to demo.

### Task 2: Implement minimal active scope

**Files:**
- Create: `swiggy_scope.py`
- Modify: `swiggy_auth.py`
- Modify: `agent.py`
- Modify: `main.py`
- Modify: `scripts/check_swiggy_token_health.py`

- [ ] **Step 1: Add `swiggy_scope.py`**

Create constants for `ACTIVE_SWIGGY_SERVERS = ("swiggy-instamart",)`, `SERVER_AUTH_KEYS`, and `ACTIVE_TOKEN_KEYS = ("im",)`.

- [ ] **Step 2: Add scoped token loading**

Add `swiggy_auth.get_access_tokens(keys)` and keep `get_all_access_tokens()` as a compatibility wrapper for all resources.

- [ ] **Step 3: Wire active scope into live agent**

Make `_route_servers()` return the shared active server list, load only `ACTIVE_TOKEN_KEYS`, and return a clear auth-not-ready response on missing live auth when `DEMO_MODE=false`.

- [ ] **Step 4: Wire active scope into health and monitor**

Make `/health` expose `swiggy_required_tokens` and compute readiness from active tokens. Make the monitor default to active tokens and accept an explicit `--keys` override.

### Task 3: Verify

**Files:**
- Test: `tests/*.py`

- [ ] **Step 1: Run focused tests**

Run:

```bash
python3 -m unittest tests.test_swiggy_auth tests.test_health tests.test_token_health_monitor tests.test_agent_timeouts -v
```

Expected: all focused tests pass.

- [ ] **Step 2: Run full suite**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 3: Check formatting hazards**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.
