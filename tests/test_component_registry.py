"""Tests for semantic role-to-component registry behavior."""

import pytest
from pydantic import ValidationError

from proposal_ai_agent.proposal_generation.component_registry import (
    AppendixComponent,
    BaseComponentDefinition,
    ComponentRegistry,
    ComponentRoleNotRegistered,
    CoverPageComponent,
    DuplicateComponentRegistration,
    HeadingComponent,
    ModuleBannerComponent,
    ReferenceComponent,
    TOCComponent,
)
from proposal_ai_agent.proposal_generation.document_plan import SectionRole


def test_standard_registry_resolves_each_registered_semantic_role() -> None:
    registry = ComponentRegistry.standard()

    assert isinstance(registry.resolve(SectionRole.COVER), CoverPageComponent)
    assert isinstance(registry.resolve(SectionRole.TABLE_OF_CONTENTS), TOCComponent)
    assert isinstance(registry.resolve(SectionRole.MODULE), ModuleBannerComponent)
    assert isinstance(registry.resolve(SectionRole.BODY), HeadingComponent)
    assert isinstance(registry.resolve(SectionRole.APPENDIX), AppendixComponent)
    assert isinstance(registry.resolve(SectionRole.ANNEX), AppendixComponent)
    assert isinstance(registry.resolve(SectionRole.REFERENCES), ReferenceComponent)


def test_registry_supports_registration_and_rejects_duplicate_roles() -> None:
    registry = ComponentRegistry()
    definition = CoverPageComponent()

    registry.register(definition)

    assert registry.resolve(SectionRole.COVER) is definition
    with pytest.raises(DuplicateComponentRegistration, match="cover"):
        registry.register(CoverPageComponent())


def test_registry_fails_explicitly_for_unregistered_roles() -> None:
    with pytest.raises(ComponentRoleNotRegistered, match="body"):
        ComponentRegistry().resolve(SectionRole.BODY)


def test_component_metadata_is_semantic_and_immutable() -> None:
    definition = HeadingComponent()

    assert definition.component_name == "heading"
    assert definition.supported_section_roles == (SectionRole.BODY,)
    assert definition.optional_block_types == (
        "paragraph", "bullet_list", "table", "requirement_matrix", "visual_placeholder", "callout",
    )
    assert definition.allows_children is True
    assert definition.supports_numbering is True
    with pytest.raises(ValidationError):
        definition.component_name = "changed"  # type: ignore[misc]


def test_custom_component_definition_can_extend_an_unregistered_role() -> None:
    custom = BaseComponentDefinition(
        component_name="custom_body",
        supported_section_roles=(SectionRole.BODY,),
        required_block_types=("paragraph",),
        optional_block_types=(),
        allows_children=False,
        supports_numbering=False,
        supports_page_break=False,
    )
    registry = ComponentRegistry((custom,))

    assert registry.resolve(SectionRole.BODY) == custom
