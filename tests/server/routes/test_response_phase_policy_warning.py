"""Tests for the RESPONSE-phase-unreachable warning (omnigent-ai/omnigent#5939).

A ``claude-sdk`` session's LLM-phase round trip (``PHASE_LLM_REQUEST`` /
``PHASE_LLM_RESPONSE`` via ``POST .../policies/evaluate``) never also posts
the assistant's message back through ``POST .../events``, so a guardrail
policy that would act on a ``response`` event is declared but structurally
never evaluated for that session — see the issue for the full trace.

The spec parser only accepts ``type: function`` policies today (``type:
prompt`` was removed), and a function policy discards whatever ``on:`` the
bundle author wrote and self-selects its handled event types at runtime —
so in practice we can never statically prove a given policy DOES act on
``response``, only that a policy with an explicit, RESPONSE-free ``on:``
does NOT. These tests cover all three outcomes
(:func:`_unreachable_response_phase_policy_names`,
:func:`_self_selecting_policy_names`) and the once-per-session warning
trigger (:func:`_warn_if_response_phase_policy_unreachable`) in isolation,
without spinning up the full route/DB stack.
"""

from __future__ import annotations

import logging

import pytest

from omnigent.server.routes.sessions.routes_hooks import (
    _self_selecting_policy_names,
    _unreachable_response_phase_policy_names,
    _warn_if_response_phase_policy_unreachable,
    _warned_unreachable_response_policy_sessions,
)
from omnigent.spec import AgentSpec
from omnigent.spec.types import ExecutorSpec, GuardrailsSpec, Phase, PhaseSelector, PolicySpec


def _spec(
    *,
    harness: str = "claude-sdk",
    policies: list[PolicySpec] | None = None,
    no_guardrails: bool = False,
) -> AgentSpec:
    """Build a minimal AgentSpec for these tests.

    :param harness: Value written to ``executor.config["harness"]``.
    :param policies: Policies to attach via a fresh ``GuardrailsSpec``.
        ``None`` builds a ``GuardrailsSpec`` with no ``policies:`` key
        (the "labels only, no policies" case).
    :param no_guardrails: When true, ``guardrails`` is ``None`` entirely
        (no ``guardrails:`` block in the bundle at all).
    :returns: A populated :class:`AgentSpec`.
    """
    guardrails = None if no_guardrails else GuardrailsSpec(policies=policies)
    return AgentSpec(
        spec_version=1,
        name="test-agent",
        executor=ExecutorSpec(config={"harness": harness}),
        guardrails=guardrails,
    )


@pytest.fixture(autouse=True)
def _clear_warned_sessions():
    """Isolate the module-level dedup set between tests."""
    _warned_unreachable_response_policy_sessions.clear()
    yield
    _warned_unreachable_response_policy_sessions.clear()


# ── _unreachable_response_phase_policy_names ────────────────────


def test_finds_policy_with_explicit_response_phase():
    policy = PolicySpec(name="redact_pii", on=[PhaseSelector(phase=Phase.RESPONSE)])
    spec = _spec(policies=[policy])
    assert _unreachable_response_phase_policy_names(spec) == ["redact_pii"]


def test_finds_policy_bound_to_multiple_phases_including_response():
    policy = PolicySpec(
        name="redact_pii",
        on=[PhaseSelector(phase=Phase.REQUEST), PhaseSelector(phase=Phase.RESPONSE)],
    )
    spec = _spec(policies=[policy])
    assert _unreachable_response_phase_policy_names(spec) == ["redact_pii"]


def test_ignores_policy_scoped_to_other_phases_only():
    policy = PolicySpec(name="gate_tools", on=[PhaseSelector(phase=Phase.TOOL_CALL)])
    spec = _spec(policies=[policy])
    assert _unreachable_response_phase_policy_names(spec) == []


def test_ignores_self_selecting_policy_since_on_is_discarded_at_parse_time():
    # Real function-type policy instances always have on=None regardless of
    # what the bundle YAML wrote (see _parse_policy_base_fields) — this
    # models that, rather than a real function policy self-selecting
    # "response" at runtime, which is undetectable statically.
    policy = PolicySpec(name="tripwire", on=None)
    spec = _spec(policies=[policy])
    assert _unreachable_response_phase_policy_names(spec) == []


def test_no_guardrails_returns_empty():
    spec = _spec(no_guardrails=True)
    assert _unreachable_response_phase_policy_names(spec) == []


def test_guardrails_with_labels_but_no_policies_returns_empty():
    # policies defaults to None on GuardrailsSpec — a bundle that only
    # declares `guardrails.labels` and no `policies:` key hits this.
    spec = _spec(policies=None)
    assert _unreachable_response_phase_policy_names(spec) == []


def test_preserves_declaration_order_for_multiple_matches():
    policies = [
        PolicySpec(name="first", on=[PhaseSelector(phase=Phase.RESPONSE)]),
        PolicySpec(name="second", on=[PhaseSelector(phase=Phase.RESPONSE)]),
    ]
    spec = _spec(policies=policies)
    assert _unreachable_response_phase_policy_names(spec) == ["first", "second"]


# ── _self_selecting_policy_names ────────────────────────────────


def test_finds_self_selecting_policy():
    policy = PolicySpec(name="tripwire", on=None)
    spec = _spec(policies=[policy])
    assert _self_selecting_policy_names(spec) == ["tripwire"]


def test_ignores_policy_with_explicit_on():
    policy = PolicySpec(name="gate_tools", on=[PhaseSelector(phase=Phase.TOOL_CALL)])
    spec = _spec(policies=[policy])
    assert _self_selecting_policy_names(spec) == []


def test_self_selecting_no_guardrails_returns_empty():
    spec = _spec(no_guardrails=True)
    assert _self_selecting_policy_names(spec) == []


# ── _warn_if_response_phase_policy_unreachable ──────────────────


def test_warns_naming_policy_with_explicit_response_phase(caplog):
    spec = _spec(
        harness="claude-sdk",
        policies=[PolicySpec(name="redact_pii", on=[PhaseSelector(phase=Phase.RESPONSE)])],
    )
    with caplog.at_level(logging.WARNING):
        _warn_if_response_phase_policy_unreachable(
            session_id="conv_1", phase=Phase.LLM_RESPONSE, spec=spec
        )
    matching = [r for r in caplog.records if "redact_pii" in r.getMessage()]
    assert matching, (
        f"expected a warning naming redact_pii, got: {[r.getMessage() for r in caplog.records]}"
    )
    assert "conv_1" in matching[0].getMessage()
    assert "claude-sdk" in matching[0].getMessage()


def test_warns_as_caveat_for_self_selecting_policy(caplog):
    spec = _spec(
        harness="claude-sdk",
        policies=[PolicySpec(name="tripwire", on=None)],
    )
    with caplog.at_level(logging.WARNING):
        _warn_if_response_phase_policy_unreachable(
            session_id="conv_1", phase=Phase.LLM_RESPONSE, spec=spec
        )
    matching = [r for r in caplog.records if "tripwire" in r.getMessage()]
    got = [r.getMessage() for r in caplog.records]
    assert matching, f"expected a caveat warning naming tripwire, got: {got}"
    assert "self-select" in matching[0].getMessage()


def test_does_not_warn_when_every_policy_provably_excludes_response(caplog):
    # gate_tools has an explicit on: that names only TOOL_CALL — provably
    # not affected, and it's the only policy, so there's nothing to caveat
    # either.
    spec = _spec(
        policies=[PolicySpec(name="gate_tools", on=[PhaseSelector(phase=Phase.TOOL_CALL)])]
    )
    with caplog.at_level(logging.WARNING):
        _warn_if_response_phase_policy_unreachable(
            session_id="conv_1", phase=Phase.LLM_RESPONSE, spec=spec
        )
    assert not any("policy_response_phase_unreachable" in r.getMessage() for r in caplog.records)


def test_second_call_for_same_session_does_not_warn_again(caplog):
    spec = _spec(policies=[PolicySpec(name="tripwire", on=None)])
    with caplog.at_level(logging.WARNING):
        _warn_if_response_phase_policy_unreachable(
            session_id="conv_1", phase=Phase.LLM_REQUEST, spec=spec
        )
        _warn_if_response_phase_policy_unreachable(
            session_id="conv_1", phase=Phase.LLM_RESPONSE, spec=spec
        )
    matching = [r for r in caplog.records if "tripwire" in r.getMessage()]
    assert len(matching) == 1, f"expected exactly one warning, got {len(matching)}: {matching}"


def test_different_sessions_each_get_their_own_warning(caplog):
    spec = _spec(policies=[PolicySpec(name="tripwire", on=None)])
    with caplog.at_level(logging.WARNING):
        _warn_if_response_phase_policy_unreachable(
            session_id="conv_1", phase=Phase.LLM_RESPONSE, spec=spec
        )
        _warn_if_response_phase_policy_unreachable(
            session_id="conv_2", phase=Phase.LLM_RESPONSE, spec=spec
        )
    matching = [r for r in caplog.records if "tripwire" in r.getMessage()]
    assert len(matching) == 2


def test_does_not_warn_for_unaffected_harness(caplog):
    spec = _spec(
        harness="codex",
        policies=[PolicySpec(name="tripwire", on=None)],
    )
    with caplog.at_level(logging.WARNING):
        _warn_if_response_phase_policy_unreachable(
            session_id="conv_1", phase=Phase.LLM_RESPONSE, spec=spec
        )
    assert not any("tripwire" in r.getMessage() for r in caplog.records)


def test_does_not_warn_for_non_llm_phase(caplog):
    # A TOOL_CALL round trip isn't the signal we key off of — only the
    # LLM-phase round trip confirms we're on this harness's own path.
    spec = _spec(policies=[PolicySpec(name="tripwire", on=None)])
    with caplog.at_level(logging.WARNING):
        _warn_if_response_phase_policy_unreachable(
            session_id="conv_1", phase=Phase.TOOL_CALL, spec=spec
        )
    assert not any("tripwire" in r.getMessage() for r in caplog.records)


def test_does_not_warn_with_no_guardrails(caplog):
    spec = _spec(no_guardrails=True)
    with caplog.at_level(logging.WARNING):
        _warn_if_response_phase_policy_unreachable(
            session_id="conv_1", phase=Phase.LLM_RESPONSE, spec=spec
        )
    assert not any("policy_response_phase_unreachable" in r.getMessage() for r in caplog.records)
