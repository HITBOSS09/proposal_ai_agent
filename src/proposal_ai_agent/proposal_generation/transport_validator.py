"""Validation-only boundary for provider-neutral proposal transport payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from .transport_contract import (
    BulletListResponse,
    CalloutResponse,
    KnowledgeReferenceResponse,
    ParagraphResponse,
    ProposalResponse,
    RequirementMatrixResponse,
    SectionResponse,
    TableResponse,
    VisualPlaceholderResponse,
)


class TransportValidationError(ValueError):
    """Base class for explicit transport-contract validation failures."""


class MissingField(TransportValidationError):
    """A mandatory transport value was omitted or blank."""


class UnknownBlockType(TransportValidationError):
    """A transport block declared an unsupported discriminator value."""


class InvalidEnumValue(TransportValidationError):
    """A transport enum-like field contains an unsupported value."""


class InvalidHierarchy(TransportValidationError):
    """The section hierarchy is structurally invalid."""


class InvalidHeadingLevel(TransportValidationError):
    """A heading level is not a positive document-outline level."""


class DuplicateIdentifier(TransportValidationError):
    """A non-reference semantic identifier is repeated."""


class DuplicateReference(TransportValidationError):
    """A declared knowledge-reference identifier is repeated."""


class UnknownReference(TransportValidationError):
    """A block cites a reference not declared by the proposal response."""


class MalformedTable(TransportValidationError):
    """A table does not have a valid rectangular transport shape."""


class CircularSectionHierarchy(TransportValidationError):
    """A section is its own direct or indirect descendant."""


class ProposalTransportValidator:
    """Validate transport payloads without mapping or modifying them."""

    def validate(self, payload: ProposalResponse | Mapping[str, Any]) -> None:
        """Raise a typed error when ``payload`` violates the transport contract."""
        response = self._as_response(payload)
        self._require_text(response.proposal_id, "proposal_id")
        self._require_text(response.title, "title")
        self._require_text(response.metadata.transport_version, "metadata.transport_version")

        reference_ids = self._validate_references(response.references)
        section_ids: set[str] = set()
        visual_ids: set[str] = set()
        active_sections: set[int] = set()
        for index, section in enumerate(response.sections):
            self._validate_section(
                section,
                path=f"sections[{index}]",
                parent_level=None,
                reference_ids=reference_ids,
                section_ids=section_ids,
                visual_ids=visual_ids,
                active_sections=active_sections,
            )

    @staticmethod
    def _as_response(payload: ProposalResponse | Mapping[str, Any]) -> ProposalResponse:
        if isinstance(payload, ProposalResponse):
            return payload
        if not isinstance(payload, Mapping):
            raise MissingField("transport payload must be a ProposalResponse object or JSON mapping")
        try:
            return ProposalResponse.model_validate(payload)
        except ValidationError as error:
            raise ProposalTransportValidator._translate_schema_error(error) from error

    @staticmethod
    def _translate_schema_error(error: ValidationError) -> TransportValidationError:
        detail = error.errors()[0]
        path = ".".join(str(part) for part in detail["loc"])
        error_type = detail["type"]
        if error_type == "missing":
            return MissingField(f"missing required transport field: {path}")
        if error_type in {"union_tag_invalid", "union_tag_not_found"}:
            return UnknownBlockType(f"unknown block type at {path}")
        if error_type == "literal_error":
            return InvalidEnumValue(f"invalid enum value at {path}")
        if error_type == "recursion_loop":
            return CircularSectionHierarchy(f"circular section hierarchy at {path}")
        return TransportValidationError(f"invalid transport payload at {path}: {detail['msg']}")

    def _validate_references(self, references: tuple[KnowledgeReferenceResponse, ...]) -> set[str]:
        reference_ids: set[str] = set()
        for index, reference in enumerate(references):
            path = f"references[{index}]"
            self._require_text(reference.reference_id, f"{path}.reference_id")
            self._require_text(reference.title, f"{path}.title")
            self._require_text(reference.source, f"{path}.source")
            if reference.reference_id in reference_ids:
                raise DuplicateReference(f"duplicate reference_id: {reference.reference_id}")
            reference_ids.add(reference.reference_id)
        return reference_ids

    def _validate_section(
        self,
        section: SectionResponse,
        *,
        path: str,
        parent_level: int | None,
        reference_ids: set[str],
        section_ids: set[str],
        visual_ids: set[str],
        active_sections: set[int],
    ) -> None:
        section_identity = id(section)
        if section_identity in active_sections:
            raise CircularSectionHierarchy(f"circular section hierarchy at {path}")
        active_sections.add(section_identity)
        try:
            self._require_text(section.section_id, f"{path}.section_id")
            self._require_text(section.heading.text, f"{path}.heading.text")
            if section.heading.level < 1:
                raise InvalidHeadingLevel(f"heading level must be positive at {path}.heading.level")
            if parent_level is not None and section.heading.level <= parent_level:
                raise InvalidHierarchy(f"child heading must be deeper than its parent at {path}.heading.level")
            if section.section_id in section_ids:
                raise DuplicateIdentifier(f"duplicate section_id: {section.section_id}")
            section_ids.add(section.section_id)

            for index, block in enumerate(section.blocks):
                self._validate_block(
                    block,
                    path=f"{path}.blocks[{index}]",
                    reference_ids=reference_ids,
                    visual_ids=visual_ids,
                )
            for index, child in enumerate(section.children):
                self._validate_section(
                    child,
                    path=f"{path}.children[{index}]",
                    parent_level=section.heading.level,
                    reference_ids=reference_ids,
                    section_ids=section_ids,
                    visual_ids=visual_ids,
                    active_sections=active_sections,
                )
        finally:
            active_sections.remove(section_identity)

    def _validate_block(
        self,
        block: object,
        *,
        path: str,
        reference_ids: set[str],
        visual_ids: set[str],
    ) -> None:
        if isinstance(block, ParagraphResponse):
            self._require_text(block.text, f"{path}.text")
            self._validate_reference_ids(block.reference_ids, f"{path}.reference_ids", reference_ids)
        elif isinstance(block, BulletListResponse):
            if not block.items:
                raise MissingField(f"missing required transport field: {path}.items")
            for index, item in enumerate(block.items):
                self._require_text(item, f"{path}.items[{index}]")
        elif isinstance(block, TableResponse):
            self._validate_table(block, path)
        elif isinstance(block, VisualPlaceholderResponse):
            self._require_text(block.visual_id, f"{path}.visual_id")
            self._require_text(block.description, f"{path}.description")
            if block.visual_id in visual_ids:
                raise DuplicateIdentifier(f"duplicate visual_id: {block.visual_id}")
            visual_ids.add(block.visual_id)
        elif isinstance(block, CalloutResponse):
            self._require_text(block.label, f"{path}.label")
            self._require_text(block.text, f"{path}.text")
            self._validate_reference_ids(block.reference_ids, f"{path}.reference_ids", reference_ids)
        elif isinstance(block, RequirementMatrixResponse):
            self._validate_requirement_matrix(block, path, reference_ids)
        else:
            raise UnknownBlockType(f"unknown block type at {path}")

    def _validate_table(self, table: TableResponse, path: str) -> None:
        if not table.headers:
            raise MalformedTable(f"table has no headers at {path}")
        if not table.rows:
            raise MalformedTable(f"table has no rows at {path}")
        for index, header in enumerate(table.headers):
            self._require_text(header, f"{path}.headers[{index}]")
        for index, row in enumerate(table.rows):
            if len(row) != len(table.headers):
                raise MalformedTable(f"table row {index} does not match header count at {path}")

    def _validate_requirement_matrix(
        self,
        matrix: RequirementMatrixResponse,
        path: str,
        reference_ids: set[str],
    ) -> None:
        if not matrix.entries:
            raise MissingField(f"missing required transport field: {path}.entries")
        requirement_ids: set[str] = set()
        for index, entry in enumerate(matrix.entries):
            entry_path = f"{path}.entries[{index}]"
            self._require_text(entry.requirement_id, f"{entry_path}.requirement_id")
            self._require_text(entry.requirement, f"{entry_path}.requirement")
            self._require_text(entry.response, f"{entry_path}.response")
            if entry.requirement_id in requirement_ids:
                raise DuplicateIdentifier(f"duplicate requirement_id: {entry.requirement_id}")
            requirement_ids.add(entry.requirement_id)
            self._validate_reference_ids(entry.evidence_reference_ids, f"{entry_path}.evidence_reference_ids", reference_ids)

    def _validate_reference_ids(self, cited_ids: tuple[str, ...], path: str, known_ids: set[str]) -> None:
        for reference_id in cited_ids:
            self._require_text(reference_id, path)
            if reference_id not in known_ids:
                raise UnknownReference(f"unknown reference_id: {reference_id} at {path}")

    @staticmethod
    def _require_text(value: str, path: str) -> None:
        if not value.strip():
            raise MissingField(f"missing required transport field: {path}")


def validate_proposal_transport(payload: ProposalResponse | Mapping[str, Any]) -> None:
    """Validate one transport payload without mapping or modifying it."""
    ProposalTransportValidator().validate(payload)
