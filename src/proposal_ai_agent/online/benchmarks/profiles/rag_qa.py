"""Declarative benchmark profile for retrieval-augmented question answering."""

from ..registry import BenchmarkProfile


RAG_QA_PROFILE = BenchmarkProfile(
    intent_id="RAG_QA",
    required_parameters=("question",),
    optional_parameters=("document", "department", "version"),
    defaults={},
    validation_rules={
        "question": r".*\S.*",
        "document": r".*\S.*",
        "department": r".*\S.*",
        "version": r".*\S.*",
    },
    confidence_threshold=0.75,
    clarification_policy={
        "missing_required": True,
        "ambiguity": True,
        "conflict": True,
        "low_confidence": True,
    },
    is_default=True,
)
