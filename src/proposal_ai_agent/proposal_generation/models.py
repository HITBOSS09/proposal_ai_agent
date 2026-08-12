"""Template-independent contracts for enterprise proposal authoring."""

from __future__ import annotations

from typing import Any, Mapping
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class _ProposalGenerationModel(BaseModel):
    """Frozen base model for explicit, serializable proposal-domain contracts."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ClientInformation(_ProposalGenerationModel):
    """Organization and contact information supplied for a proposal recipient."""

    name: str = Field(min_length=1)
    contact_name: str | None = None
    contact_email: str | None = None
    attributes: Mapping[str, Any] = Field(default_factory=dict)


class Requirement(_ProposalGenerationModel):
    """One user-supplied requirement, without imposing an industry taxonomy."""

    description: str = Field(min_length=1)
    identifier: str | None = None
    category: str | None = None
    priority: str | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class Constraint(_ProposalGenerationModel):
    """One user-supplied delivery, commercial, or technical constraint."""

    description: str = Field(min_length=1)
    identifier: str | None = None
    category: str | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class ProposalRequest(_ProposalGenerationModel):
    """Raw questionnaire submission used to author a single proposal."""

    request_id: UUID = Field(default_factory=uuid4)
    client: ClientInformation
    proposal_title: str = Field(min_length=1)
    project_data: Mapping[str, Any] = Field(default_factory=dict)
    requirements: tuple[Requirement, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class ProjectModel(_ProposalGenerationModel):
    """Normalized project representation consumed by future proposal planners."""

    source_request_id: UUID
    client: ClientInformation
    proposal_title: str = Field(min_length=1)
    project_data: Mapping[str, Any] = Field(default_factory=dict)
    requirements: tuple[Requirement, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class SectionPlan(_ProposalGenerationModel):
    """One planned section, independent of any particular proposal template."""

    section_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    purpose: str | None = None
    input_keys: tuple[str, ...] = ()
    authoring_reference: bool = False
    technical_reference: bool = False
    template_reference: bool = False
    flags: Mapping[str, bool] = Field(default_factory=dict)
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class ProposalPlan(_ProposalGenerationModel):
    """High-level, future planner output for a proposal document."""

    project: ProjectModel
    sections: tuple[SectionPlan, ...] = ()
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class ProposalContent(_ProposalGenerationModel):
    """Dynamic authored content keyed by template-defined section identifiers."""

    project: ProjectModel
    sections: Mapping[str, Mapping[str, Any]] = Field(default_factory=dict)
    metadata: Mapping[str, Any] = Field(default_factory=dict)
