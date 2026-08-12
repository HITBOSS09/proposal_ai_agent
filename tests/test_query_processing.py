"""Unit tests for online syntactic query processing."""

from dataclasses import FrozenInstanceError, replace
from hashlib import sha256

import pytest

from proposal_ai_agent.online import ProcessedQuery, QueryEngine


def _qualified(query: str):
    engine = QueryEngine()
    return engine, engine.qualify_query(engine.receive_query(query))


def test_process_query_normalizes_text_without_changing_qualification() -> None:
    engine, qualified = _qualified("  What\r\n\tpolicy applies?   document: handbook  ")

    processed = engine.process_query(qualified)

    assert isinstance(processed, ProcessedQuery)
    assert processed.normalized_query == "What policy applies? document: handbook"
    assert processed.qualified_query == qualified
    assert processed.qualified_query.intent == "RAG_QA"
    assert processed.qualified_query.extracted_parameters["document"] == "handbook"
    assert processed.processing_flags["line_breaks_normalized"] is True
    assert processed.processing_flags["whitespace_normalized"] is True


def test_process_query_normalizes_unicode_and_detects_language() -> None:
    engine, qualified = _qualified("Cafe\u0301 policy")

    processed = engine.process_query(qualified)

    assert processed.normalized_query == "Café policy"
    assert processed.processing_flags["unicode_normalized"] is True
    assert processed.language == "en"
    assert processed.language_confidence == 0.6


def test_process_query_detects_supported_unicode_script_language() -> None:
    engine, qualified = _qualified("नमस्ते नीति")

    processed = engine.process_query(qualified)

    assert processed.language == "hi"
    assert processed.language_confidence > 0


@pytest.mark.parametrize("query", ["contains\x00null", "x" * 10_001, "bad\ud800text"])
def test_process_query_rejects_invalid_character_length_or_encoding(query: str) -> None:
    engine, qualified = _qualified(query)

    with pytest.raises(ValueError):
        engine.process_query(qualified)


def test_process_query_hash_and_cache_key_are_deterministic() -> None:
    first_engine, first_qualified = _qualified("What   policy applies?")
    second_engine, second_qualified = _qualified("What policy applies?")

    first = first_engine.process_query(first_qualified)
    second = second_engine.process_query(second_qualified)
    expected_hash = sha256("What policy applies?".encode("utf-8")).hexdigest()

    assert first.query_hash == expected_hash
    assert first.query_hash == second.query_hash
    assert first.cache_key == second.cache_key == f"query:{expected_hash}"


def test_process_query_calculates_deterministic_statistics() -> None:
    engine, qualified = _qualified("one two three four")

    processed = engine.process_query(qualified)

    assert processed.character_count == 18
    assert processed.word_count == 4
    assert processed.estimated_token_count == 5
    assert processed.processing_timestamp_utc.tzinfo is not None
    assert processed.processing_version == "1.0"


def test_processed_query_is_immutable_and_defensively_copies_flags() -> None:
    engine, qualified = _qualified("What applies?")
    flags = {"nested": {"state": "ready"}}
    processed = replace(
        engine.process_query(qualified),
        processing_flags=flags,
    )
    flags["nested"]["state"] = "changed"

    with pytest.raises(FrozenInstanceError):
        processed.language = "fr"  # type: ignore[misc]
    with pytest.raises(TypeError):
        processed.processing_flags["new"] = True  # type: ignore[index]

    assert processed.processing_flags["nested"]["state"] == "ready"


def test_processed_query_equality_is_value_based() -> None:
    engine, qualified = _qualified("What applies?")
    first = engine.process_query(qualified)
    second = replace(first)

    assert first == second


def test_process_query_regression_preserves_phase_two_outcome() -> None:
    engine, qualified = _qualified("What applies? document: handbook")
    before = (
        qualified.intent,
        qualified.benchmark_id,
        qualified.extracted_parameters,
        qualified.validation_result,
    )

    processed = engine.process_query(qualified)

    assert (
        processed.qualified_query.intent,
        processed.qualified_query.benchmark_id,
        processed.qualified_query.extracted_parameters,
        processed.qualified_query.validation_result,
    ) == before
