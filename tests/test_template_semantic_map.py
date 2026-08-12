"""Contracts for the audited PRU template semantic map."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from proposal_ai_agent.proposal_generation.publishing import (
    ElementKind,
    PRU_TEMPLATE_SHA256,
    ReusableComponent,
    SemanticField,
    Story,
    StructuralLocator,
    file_sha256,
    pru_template_semantic_map,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "documents/PRU_T72_Module_Breakdown.docx"


def test_pru_map_is_version_pinned_and_contains_only_verified_semantics() -> None:
    semantic_map = pru_template_semantic_map(TEMPLATE)

    assert semantic_map.template_sha256 == PRU_TEMPLATE_SHA256 == file_sha256(TEMPLATE)
    assert {target.field for target in semantic_map.dynamic_fields} == set(SemanticField)
    assert "CLIENT_NAME" not in {field.value for field in SemanticField}
    assert "REVISION" not in {field.value for field in SemanticField}
    assert "COMPLIANCE_TABLE" not in {component.value for component in ReusableComponent}
    assert semantic_map.target(SemanticField.DOCUMENT_NUMBER).locators[0].surrounding_label == "Document No."
    assert len(semantic_map.target(SemanticField.HEADER_PROJECT_TEXT).locators) == 5
    assert semantic_map.prototype(ReusableComponent.MODULE_BANNER).locator.drawing_name == "Textbox 6"


def test_structural_locator_rejects_incomplete_or_cross_story_coordinates() -> None:
    with pytest.raises(ValidationError):
        StructuralLocator(story=Story.HEADER, element_kind=ElementKind.DRAWING_TEXT, drawing_name="x")
    with pytest.raises(ValidationError):
        StructuralLocator(
            story=Story.BODY, element_kind=ElementKind.TABLE_CELL,
            part_name="word/header1.xml", table_index=0,
        )
