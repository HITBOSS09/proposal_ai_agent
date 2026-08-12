"""Deterministic prompt composition for proposal sections."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import TypeAliasType

from .models import ProjectModel, ProposalPlan
from .transport_contract import ProposalResponse

if TYPE_CHECKING:
    from .retrieval_context import RetrievedContext


JSONValue = TypeAliasType(
    "JSONValue",
    str
    | int
    | float
    | bool
    | None
    | Mapping[str, "JSONValue"]
    | list["JSONValue"]
    | tuple["JSONValue", ...],
)


class _FrozenJSONMapping(dict[str, JSONValue]):
    """Dictionary-compatible JSON mapping that rejects mutation."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        raise TypeError("RetrievedReference metadata is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _freeze_json_value(value: JSONValue) -> JSONValue:
    """Recursively detach and freeze JSON-compatible metadata values."""
    if isinstance(value, Mapping):
        return _FrozenJSONMapping({key: _freeze_json_value(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json_value(item) for item in value)
    return value


class GenerationStrategy(str, Enum):
    """Reference combinations permitted for a section-generation request."""

    USER_ONLY = "USER_ONLY"
    USER_PLUS_STYLE = "USER_PLUS_STYLE"
    USER_PLUS_TECHNICAL = "USER_PLUS_TECHNICAL"
    FULL_HYBRID = "FULL_HYBRID"


class _PromptModel(BaseModel):
    """Frozen base model for proposal-specific prompt contracts."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)


class RetrievedReference(_PromptModel):
    """Typed reference content supplied by a future retrieval executor."""

    reference_id: str = Field(min_length=1)
    reference_type: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    source_document: str = Field(min_length=1)
    score: float = Field(ge=0.0)
    content: str = Field(min_length=1)
    metadata: Mapping[str, JSONValue] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def freeze_metadata(cls, value: Mapping[str, JSONValue]) -> Mapping[str, JSONValue]:
        return _freeze_json_value(value)  # type: ignore[return-value]


class SectionPromptPackage(_PromptModel):
    """Provider-agnostic prompt package for one complete proposal response.

    The historical class name is retained as a compatibility import only.  The
    package now requests the frozen transport contract for the entire proposal;
    it never requests legacy raw section text.
    """

    proposal_id: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)
    generation_strategy: GenerationStrategy
    expected_output_model: type[BaseModel]
    planned_section_ids: tuple[str, ...]
    style_references: tuple[RetrievedReference, ...] = ()
    technical_references: tuple[RetrievedReference, ...] = ()
    blueprint_references: tuple[RetrievedReference, ...] = ()
    metadata: Mapping[str, str | int | float | bool | None] = Field(default_factory=dict)


class PromptComposer:
    """Compose one deterministic structured-JSON proposal prompt."""

    _SYSTEM_RULES = "\n".join(
        (
            "Generate only a JSON object conforming to the supplied ProposalResponse schema.",
            "Do not generate Markdown, HTML, DOCX, styling, layout, or explanatory text outside the JSON object.",
            "Use user-supplied project facts as the highest-priority source of truth.",
            "Fact priority is: User Input > Technical References > Authoring References > Blueprint References.",
            "Authoring and blueprint references may guide structure and tone only; never copy their client names, project names, budgets, dates, quantities, or specifications.",
            "Never invent specifications. Omit unsupported content rather than guessing.",
            "Use the required section identifiers exactly once and preserve their order.",
        )
    )

    def compose(
        self,
        project: ProjectModel,
        proposal_plan: ProposalPlan | None = None,
        retrieved_contexts: Sequence[RetrievedContext] = (),
    ) -> SectionPromptPackage:
        """Build one complete proposal prompt with the transport schema contract."""
        if not isinstance(project, ProjectModel):
            raise TypeError("project must be a ProjectModel")
        if proposal_plan is None:
            from .proposal_planner import ProposalPlanner

            plan = ProposalPlanner().plan(project)
        else:
            plan = proposal_plan
        if plan is not None and not isinstance(plan, ProposalPlan):
            raise TypeError("proposal_plan must be a ProposalPlan")
        if plan is not None and plan.project != project:
            raise ValueError("proposal_plan.project must match project")
        from .retrieval_context import RetrievedContext

        contexts = tuple(retrieved_contexts)
        if any(not isinstance(context, RetrievedContext) for context in contexts):
            raise TypeError("retrieved_contexts must contain RetrievedContext values")
        sections = tuple((section.section_id, section.title) for section in plan.sections)
        style_references = tuple(reference for context in contexts for reference in context.style_references)
        technical_references = tuple(reference for context in contexts for reference in context.technical_references)
        blueprint_references = tuple(reference for context in contexts for reference in context.blueprint_references)
        blocks = [
            self._user_ground_truth_block(project),
            self._reference_block("TECHNICAL_REFERENCES", contexts, "technical_references"),
            self._reference_block("AUTHORING_REFERENCES", contexts, "style_references"),
            self._reference_block("BLUEPRINT_REFERENCES", contexts, "blueprint_references"),
            self._document_structure_block(sections),
            self._output_contract_block(),
        ]

        return SectionPromptPackage(
            proposal_id=str(project.source_request_id),
            system_prompt=self._SYSTEM_RULES,
            user_prompt="\n\n".join(blocks),
            generation_strategy=self._generation_strategy(
                style_references, technical_references, blueprint_references
            ),
            expected_output_model=ProposalResponse,
            planned_section_ids=tuple(section_id for section_id, _ in sections),
            style_references=style_references,
            technical_references=technical_references,
            blueprint_references=blueprint_references,
            metadata={
                "proposal_type": str(project.metadata.get("proposal_type", "general")),
                "section_count": len(sections),
                "reference_count": len(style_references) + len(technical_references) + len(blueprint_references),
            },
        )

    @staticmethod
    def _user_ground_truth_block(project: ProjectModel) -> str:
        requirements = "\n".join(f"- {item.description}" for item in project.requirements) or "- None supplied"
        constraints = "\n".join(f"- {item.description}" for item in project.constraints) or "- None supplied"
        project_data = json.dumps(project.project_data, sort_keys=True, default=str)
        return "\n".join(
            (
                "[USER_GROUND_TRUTH]",
                "Highest priority. Project-specific facts supplied by the user.",
                f"Client: {project.client.name}",
                f"Proposal Title: {project.proposal_title}",
                f"Project Data: {project_data}",
                "Requirements:",
                requirements,
                "Constraints:",
                constraints,
            )
        )

    @staticmethod
    def _reference_block(
        heading: str,
        contexts: tuple[RetrievedContext, ...],
        attribute: str,
    ) -> str:
        lines = [f"[{heading}]", "Lower priority than user ground truth."]
        found = False
        for context in contexts:
            for reference in getattr(context, attribute):
                found = True
                lines.extend(
                    (
                        f"Section: {context.section_id}",
                        f"Reference ID: {reference.reference_id}",
                        f"Source: {reference.source_document}",
                        f"Content: {reference.content}",
                    )
                )
        if not found:
            lines.append("None retrieved")
        return "\n".join(lines)

    @staticmethod
    def _generation_strategy(
        style_references: tuple[RetrievedReference, ...],
        technical_references: tuple[RetrievedReference, ...],
        blueprint_references: tuple[RetrievedReference, ...],
    ) -> GenerationStrategy:
        if technical_references and (style_references or blueprint_references):
            return GenerationStrategy.FULL_HYBRID
        if technical_references:
            return GenerationStrategy.USER_PLUS_TECHNICAL
        if style_references or blueprint_references:
            return GenerationStrategy.USER_PLUS_STYLE
        return GenerationStrategy.USER_ONLY

    def _document_structure_block(self, sections: Sequence[tuple[str, str]]) -> str:
        required_sections = "\n".join(
            f"{index}. section_id: {section_id}\n   title: {heading}"
            for index, (section_id, heading) in enumerate(sections, start=1)
        )
        return "\n".join(
            (
                "[PLANNED_SECTION_CONTRACT]",
                "Return exactly these top-level sections, exactly once, in exactly this order:",
                required_sections,
                "Rules:",
                "- section_id values are immutable identifiers. Copy each one exactly.",
                "- Do not rename, add, omit, duplicate, nest, or reorder sections.",
                "- Do not add child sections; the canonical plan contains only the sections listed above.",
                "- Every planned section must appear exactly once.",
                "- If factual information is limited, return a valid minimal section rather than omitting it.",
                "- Do not fabricate unsupported project facts.",
            )
        )

    @staticmethod
    def _output_contract_block() -> str:
        return "\n".join(
            (
                "[OUTPUT_CONTRACT]",
                "Return exactly one ProposalResponse JSON object.",
                json.dumps(ProposalResponse.model_json_schema(), sort_keys=True),
            )
        )
