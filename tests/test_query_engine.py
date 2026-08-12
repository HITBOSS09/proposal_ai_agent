"""Unit tests for online query reception."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from proposal_ai_agent.online import QueryEngine, UserQuery


def test_receive_query_returns_normalized_contract() -> None:
    received = QueryEngine().receive_query("  Find eligible proposals  ")

    assert isinstance(received, UserQuery)
    assert received.query == "Find eligible proposals"
    assert received.trace_metadata == {}


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_receive_query_rejects_empty_or_whitespace_only_query(query: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        QueryEngine().receive_query(query)


def test_receive_query_generates_uuid4_request_id_and_utc_timestamp() -> None:
    before = datetime.now(timezone.utc)
    received = QueryEngine().receive_query("status")
    after = datetime.now(timezone.utc)

    assert isinstance(received.request_id, UUID)
    assert received.request_id.version == 4
    assert received.timestamp_utc.tzinfo == timezone.utc
    assert before <= received.timestamp_utc <= after


def test_receive_query_preserves_session_and_optional_metadata() -> None:
    history = [{"role": "user", "content": "Earlier question"}]
    user_context = {"organization": "Acme", "preferences": {"language": "en"}}
    auth_context = {"roles": ["proposal_reader"]}

    received = QueryEngine().receive_query(
        "Current question",
        session_id="session-123",
        conversation_history=history,
        user_context=user_context,
        auth_context=auth_context,
    )

    assert received.session_id == "session-123"
    assert received.conversation_history == tuple(history)
    assert received.user_context == user_context
    assert received.auth_context["roles"] == ("proposal_reader",)


def test_user_query_is_immutable_and_metadata_is_defensively_frozen() -> None:
    history = [{"role": "user", "content": "Earlier question"}]
    context = {"preferences": {"language": "en"}}
    received = QueryEngine().receive_query(
        "Current question", conversation_history=history, user_context=context
    )
    history[0]["content"] = "Changed"
    context["preferences"]["language"] = "fr"

    with pytest.raises(FrozenInstanceError):
        received.query = "Changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        received.user_context["new"] = "value"  # type: ignore[index]

    assert received.conversation_history[0]["content"] == "Earlier question"
    assert received.user_context["preferences"]["language"] == "en"


def test_user_query_equality_uses_immutable_field_values() -> None:
    request_id = uuid4()
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = UserQuery(request_id=request_id, query="status", timestamp_utc=timestamp)
    second = UserQuery(request_id=request_id, query=" status ", timestamp_utc=timestamp)

    assert first == second
