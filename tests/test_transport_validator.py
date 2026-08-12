"""Tests for validation-only proposal transport boundary behavior."""

from copy import deepcopy

import pytest

from proposal_ai_agent.proposal_generation.transport_validator import (
    CircularSectionHierarchy,
    DuplicateIdentifier,
    DuplicateReference,
    InvalidHeadingLevel,
    InvalidHierarchy,
    MalformedTable,
    MissingField,
    ProposalTransportValidator,
    UnknownBlockType,
    UnknownReference,
    validate_proposal_transport,
)


def _payload() -> dict[str, object]:
    return {
        "proposal_id": "proposal-001",
        "title": "Autonomous Perimeter Monitoring Proposal",
        "metadata": {"transport_version": "1.0"},
        "sections": [{
            "section_id": "solution",
            "heading": {"text": "Solution", "level": 1},
            "blocks": [
                {"type": "paragraph", "text": "Persistent monitoring.", "reference_ids": ["REF-1"]},
                {"type": "bullet_list", "items": ["EO/IR observation"]},
                {"type": "table", "headers": ["Requirement", "Response"], "rows": [["R-1", "Covered"]]},
                {"type": "visual_placeholder", "visual_id": "VIS-1", "description": "Architecture"},
                {"type": "callout", "label": "Note", "text": "Survey required."},
                {"type": "requirement_matrix", "entries": [{
                    "requirement_id": "R-1",
                    "requirement": "Provide EO/IR observation",
                    "response": "EO/IR payload included",
                    "evidence_reference_ids": ["REF-1"],
                }]},
            ],
            "children": [{
                "section_id": "payload",
                "heading": {"text": "Payload", "level": 2},
            }],
        }],
        "references": [{"reference_id": "REF-1", "title": "Payload specification", "source": "Technical library"}],
    }


def test_validator_accepts_valid_payload_without_mutating_it() -> None:
    payload = _payload()
    original = deepcopy(payload)

    assert validate_proposal_transport(payload) is None
    assert payload == original


@pytest.mark.parametrize(
    ("mutate", "error_type"),
    [
        (lambda value: value.pop("proposal_id"), MissingField),
        (lambda value: value["sections"][0]["blocks"].__setitem__(0, {"type": "unknown", "text": "x"}), UnknownBlockType),  # type: ignore[index]
        (lambda value: value["sections"][0]["blocks"].__setitem__(2, {"type": "table", "headers": ["A", "B"], "rows": [["A"]]}), MalformedTable),  # type: ignore[index]
        (lambda value: value["sections"][0]["children"][0]["heading"].__setitem__("level", 1), InvalidHierarchy),  # type: ignore[index]
        (lambda value: value["sections"][0]["heading"].__setitem__("level", 0), InvalidHeadingLevel),  # type: ignore[index]
        (lambda value: value["sections"][0]["blocks"][0].__setitem__("reference_ids", ["MISSING"]), UnknownReference),  # type: ignore[index]
    ],
)
def test_validator_rejects_explicit_transport_failures(mutate, error_type) -> None:
    payload = _payload()
    mutate(payload)

    with pytest.raises(error_type):
        ProposalTransportValidator().validate(payload)


def test_validator_rejects_duplicate_section_visual_requirement_and_reference_identifiers() -> None:
    duplicate_section = _payload()
    duplicate_section["sections"].append({"section_id": "solution", "heading": {"text": "Again", "level": 1}})  # type: ignore[index]
    with pytest.raises(DuplicateIdentifier, match="section_id"):
        validate_proposal_transport(duplicate_section)

    duplicate_visual = _payload()
    duplicate_visual["sections"][0]["blocks"].append({"type": "visual_placeholder", "visual_id": "VIS-1", "description": "Repeated"})  # type: ignore[index]
    with pytest.raises(DuplicateIdentifier, match="visual_id"):
        validate_proposal_transport(duplicate_visual)

    duplicate_requirement = _payload()
    duplicate_requirement["sections"][0]["blocks"][-1]["entries"].append({"requirement_id": "R-1", "requirement": "Repeated", "response": "Repeated"})  # type: ignore[index]
    with pytest.raises(DuplicateIdentifier, match="requirement_id"):
        validate_proposal_transport(duplicate_requirement)

    duplicate_reference = _payload()
    duplicate_reference["references"].append({"reference_id": "REF-1", "title": "Repeated", "source": "Library"})  # type: ignore[index]
    with pytest.raises(DuplicateReference):
        validate_proposal_transport(duplicate_reference)


def test_validator_rejects_circular_raw_section_hierarchy() -> None:
    payload = _payload()
    section = payload["sections"][0]  # type: ignore[index]
    section["children"] = [section]  # type: ignore[index]

    with pytest.raises(CircularSectionHierarchy):
        validate_proposal_transport(payload)
