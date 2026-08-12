"""Unit tests for online query qualification."""

from dataclasses import FrozenInstanceError

import pytest

from proposal_ai_agent.online import QualifiedQuery, QueryEngine
from proposal_ai_agent.online.benchmarks import BenchmarkProfile, BenchmarkRegistry


def _received(query: str):
    engine = QueryEngine()
    return engine, engine.receive_query(query)


def test_qualify_query_detects_rag_qa_intent_and_selected_benchmark() -> None:
    engine, user_query = _received("What is the proposal approval policy?")

    qualified = engine.qualify_query(user_query)

    assert isinstance(qualified, QualifiedQuery)
    assert qualified.original is user_query
    assert qualified.intent == "RAG_QA"
    assert qualified.benchmark_id == "RAG_QA"
    assert qualified.extracted_parameters["question"] == "What is the proposal approval policy?"
    assert qualified.validation_result.is_valid is True
    assert qualified.clarification_required is False


def test_registry_selects_declaratively_registered_matching_profile() -> None:
    fallback = BenchmarkProfile(
        intent_id="FALLBACK",
        required_parameters=("question",),
        optional_parameters=(),
        is_default=True,
    )
    specialized = BenchmarkProfile(
        intent_id="SPECIAL",
        required_parameters=("question",),
        optional_parameters=(),
        intent_patterns=(r"^special:",),
    )
    registry = BenchmarkRegistry((fallback, specialized))

    assert registry.select("special: status").intent_id == "SPECIAL"
    assert registry.select("ordinary status").intent_id == "FALLBACK"


def test_qualify_query_extracts_declared_optional_parameters() -> None:
    engine, user_query = _received(
        "What are the retention rules? document: handbook, department: legal, version: v2"
    )

    qualified = engine.qualify_query(user_query)

    assert qualified.extracted_parameters == {
        "question": "What are the retention rules?",
        "document": "handbook",
        "department": "legal",
        "version": "v2",
    }
    assert qualified.optional_parameters == ("document", "department", "version")
    assert qualified.missing_parameters == ()


def test_qualify_query_detects_missing_required_question_and_requires_clarification() -> None:
    engine, user_query = _received("document: handbook")

    qualified = engine.qualify_query(user_query)

    assert qualified.missing_parameters == ("question",)
    assert qualified.validation_result.is_valid is False
    assert qualified.validation_result.errors == ("missing required parameter: question",)
    assert qualified.confidence_score == 0.55
    assert qualified.clarification_required is True
    assert qualified.clarification_requests[0].parameter == "question"
    assert qualified.clarification_requests[0].reason == "missing_required"


def test_qualify_query_detects_ambiguous_and_conflicting_parameters() -> None:
    engine, ambiguous_user_query = _received(
        "Which policy applies? document: handbook or procedures"
    )
    _, conflicting_user_query = _received(
        "Which policy applies? document: handbook, document: procedures"
    )

    ambiguous = engine.qualify_query(ambiguous_user_query)
    conflicting = engine.qualify_query(conflicting_user_query)

    assert ambiguous.ambiguity_flags == ("document",)
    assert ambiguous.clarification_required is True
    assert any(request.reason == "ambiguous" for request in ambiguous.clarification_requests)
    assert conflicting.conflict_flags == ("document",)
    assert any(request.reason == "conflicting" for request in conflicting.clarification_requests)


def test_qualified_query_is_immutable_and_defensively_copies_parameters() -> None:
    engine, user_query = _received("What applies? document: handbook")
    qualified = engine.qualify_query(user_query)

    with pytest.raises(FrozenInstanceError):
        qualified.intent = "OTHER"  # type: ignore[misc]
    with pytest.raises(TypeError):
        qualified.extracted_parameters["document"] = "changed"  # type: ignore[index]

    assert qualified.extracted_parameters["document"] == "handbook"


def test_qualified_query_equality_is_value_based() -> None:
    engine, user_query = _received("What applies? document: handbook")

    first = engine.qualify_query(user_query)
    second = engine.qualify_query(user_query)

    assert first == second


def test_query_reception_regression_remains_a_qualification_boundary() -> None:
    engine = QueryEngine()

    received = engine.receive_query("  What applies?  ")

    assert received.query == "What applies?"
    assert engine.qualify_query(received).original == received
