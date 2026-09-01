"""
End-to-end test for the RESPONSE-phase-unreachable warning
(omnigent-ai/omnigent#5939), through the real
``POST /v1/sessions/{id}/policies/evaluate`` route rather than calling
the detection functions directly.

A ``claude-sdk`` agent whose guardrails declare a policy has that
policy's ``response``-phase handling (if any) silently never
evaluated: this harness's own turn loop only ever asks this route for
``PHASE_LLM_REQUEST`` / ``PHASE_LLM_RESPONSE``, and the separate route
that would ask for ``PHASE_RESPONSE`` (``POST .../events`` with an
assistant message) is never reached in this topology. This confirms
the real route logs a loud warning instead of doing nothing.

The only policy type the spec parser accepts today is ``type:
function`` (``type: prompt`` was removed — see
``_parse_policy_spec``), and a function policy self-selects which
event types it handles at runtime rather than declaring them in
``on:``, so these tests exercise the general-caveat warning tier, not
the (currently unreachable via the parser) precise-names tier that
``tests/server/routes/test_response_phase_policy_warning.py`` covers
directly against a hand-built spec.

Uses the shared ``client`` fixture from ``tests/server/conftest.py``
(real stores + mock LLM), same as the sibling
``test_sessions_policy_evaluate.py``.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest

from tests.server.helpers import create_test_agent

pytestmark = pytest.mark.asyncio


def _tripwire_policy(event: dict[str, Any]) -> dict[str, Any]:
    """Self-selecting policy that would deny a ``response`` event
    containing a marker — mirrors the issue's own repro
    (``tripwire_policy.py``). Never actually reached by the LLM-phase
    round trip this test drives, which is exactly the bug: the
    function is declared and would act on ``response``, but nothing
    ever calls it with one in this topology."""
    if event.get("type") != "response":
        return {"result": "ALLOW"}
    return {"result": "DENY", "reason": "tripwire hit"}


async def _create_session(client: httpx.AsyncClient, agent_id: str) -> str:
    resp = await client.post("/v1/sessions", json={"agent_id": agent_id})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _llm_response_payload() -> dict[str, object]:
    return {
        "event": {
            "type": "PHASE_LLM_RESPONSE",
            "data": {"model": "gpt-4o", "text_preview": "Hello!", "tool_calls_count": 0},
            "context": {},
        },
    }


async def test_claude_sdk_session_with_guardrails_warns_on_llm_response_call(
    client: httpx.AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    A claude-sdk agent with any guardrail policy logs the general-caveat
    unreachable-response-phase warning the first time an LLM-phase
    evaluation call comes in for its session — the realistic case today,
    since every constructible policy is ``type: function`` and
    self-selects its phase, so the exact affected policy can't be named
    without executing it.
    """
    agent = await create_test_agent(
        client,
        executor={"type": "omnigent", "config": {"harness": "claude-sdk"}},
        guardrails={
            "policies": {
                "tripwire": {
                    "type": "function",
                    "function": f"{__name__}._tripwire_policy",
                },
            },
        },
    )
    session_id = await _create_session(client, agent["id"])

    with caplog.at_level(logging.WARNING):
        resp = await client.post(
            f"/v1/sessions/{session_id}/policies/evaluate",
            json=_llm_response_payload(),
        )
    assert resp.status_code == 200

    matching = [r for r in caplog.records if "policy_response_phase_unreachable" in r.getMessage()]
    assert matching, f"expected the warning, got: {[r.getMessage() for r in caplog.records]}"
    assert session_id in matching[0].getMessage()
    assert "claude-sdk" in matching[0].getMessage()


async def test_second_llm_call_in_same_session_does_not_warn_again(
    client: httpx.AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The warning fires once per session, not once per round trip."""
    agent = await create_test_agent(
        client,
        executor={"type": "omnigent", "config": {"harness": "claude-sdk"}},
        guardrails={
            "policies": {
                "tripwire": {
                    "type": "function",
                    "function": f"{__name__}._tripwire_policy",
                },
            },
        },
    )
    session_id = await _create_session(client, agent["id"])

    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            resp = await client.post(
                f"/v1/sessions/{session_id}/policies/evaluate",
                json=_llm_response_payload(),
            )
            assert resp.status_code == 200

    matching = [r for r in caplog.records if "policy_response_phase_unreachable" in r.getMessage()]
    assert len(matching) == 1, f"expected exactly one warning, got {len(matching)}: {matching}"


async def test_codex_session_with_same_guardrails_does_not_warn(
    client: httpx.AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    The same guardrails on a harness not known to have this gap (codex)
    must not trigger the warning — it's scoped to harnesses confirmed to
    skip the message-POST path, not a blanket "any guardrails" alarm.
    """
    agent = await create_test_agent(
        client,
        executor={"type": "omnigent", "config": {"harness": "codex"}},
        guardrails={
            "policies": {
                "tripwire": {
                    "type": "function",
                    "function": f"{__name__}._tripwire_policy",
                },
            },
        },
    )
    session_id = await _create_session(client, agent["id"])

    with caplog.at_level(logging.WARNING):
        resp = await client.post(
            f"/v1/sessions/{session_id}/policies/evaluate",
            json=_llm_response_payload(),
        )
    assert resp.status_code == 200
    assert not any("policy_response_phase_unreachable" in r.getMessage() for r in caplog.records)


async def test_claude_sdk_session_without_guardrails_does_not_warn(
    client: httpx.AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A claude-sdk agent with no guardrails at all has nothing to warn
    about."""
    agent = await create_test_agent(
        client,
        executor={"type": "omnigent", "config": {"harness": "claude-sdk"}},
    )
    session_id = await _create_session(client, agent["id"])

    with caplog.at_level(logging.WARNING):
        resp = await client.post(
            f"/v1/sessions/{session_id}/policies/evaluate",
            json=_llm_response_payload(),
        )
    assert resp.status_code == 200
    assert not any("policy_response_phase_unreachable" in r.getMessage() for r in caplog.records)
