"""Deterministic validation of assembled proposal-domain documents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from .contracts import ProposalDocument, ProposalReviewReport, SectionContent
from .prompt_composer import RetrievedReference


class ProposalReviewer:
    """Report proposal-document findings without changing document content."""

    def review(self, document: ProposalDocument) -> ProposalReviewReport:
        """Return deterministic findings for one assembled proposal document."""
        if not isinstance(document, ProposalDocument):
            raise TypeError("document must be a ProposalDocument")

        findings: list[str] = []
        planned_ids = tuple(section.section_id for section in document.proposal_plan.sections)
        actual_sections = tuple(document.sections)
        actual_ids = tuple(section.section_id for section in actual_sections if isinstance(section, SectionContent))

        duplicate_plan_ids = self._duplicates(planned_ids)
        for section_id in duplicate_plan_ids:
            findings.append(f"duplicate ProposalPlan section: {section_id}")

        duplicate_section_ids = self._duplicates(actual_ids)
        for section_id in duplicate_section_ids:
            findings.append(f"duplicate section content: {section_id}")

        for section_id in planned_ids:
            if section_id not in actual_ids:
                findings.append(f"missing section: {section_id}")
        for section_id in actual_ids:
            if section_id not in planned_ids:
                findings.append(f"unplanned section: {section_id}")
        if actual_ids != planned_ids:
            findings.append("section content does not match ProposalPlan ordering")

        for index, section in enumerate(actual_sections, start=1):
            if not isinstance(section, SectionContent):
                findings.append(f"invalid section content at position {index}")
            elif not section.generated_text.strip():
                findings.append(f"empty generated content: {section.section_id}")

        findings.extend(self._reference_findings(document.references))
        findings.extend(self._metadata_findings(document.metadata, "document metadata"))
        return ProposalReviewReport(
            proposal_title=document.proposal_title,
            findings=tuple(findings),
        )

    @staticmethod
    def _duplicates(section_ids: Sequence[str]) -> tuple[str, ...]:
        seen: set[str] = set()
        duplicates: list[str] = []
        for section_id in section_ids:
            if section_id in seen and section_id not in duplicates:
                duplicates.append(section_id)
            seen.add(section_id)
        return tuple(duplicates)

    def _reference_findings(self, references: Sequence[RetrievedReference]) -> tuple[str, ...]:
        findings: list[str] = []
        for index, reference in enumerate(references, start=1):
            if not isinstance(reference, RetrievedReference):
                findings.append(f"invalid reference at position {index}")
                continue
            for field_name in ("reference_id", "reference_type", "chunk_id", "source_document", "content"):
                value = getattr(reference, field_name, None)
                if not isinstance(value, str) or not value.strip():
                    findings.append(f"invalid reference {index}: {field_name}")
            if (
                isinstance(reference.score, bool)
                or not isinstance(reference.score, (int, float))
                or not isfinite(reference.score)
                or reference.score < 0
            ):
                findings.append(f"invalid reference {index}: score")
            findings.extend(self._metadata_findings(reference.metadata, f"reference {index} metadata"))
        return tuple(findings)

    def _metadata_findings(self, metadata: Any, location: str) -> tuple[str, ...]:
        if not isinstance(metadata, Mapping):
            return (f"invalid {location}",)
        findings: list[str] = []
        for key in sorted(metadata, key=lambda item: str(item)):
            if not isinstance(key, str) or not key.strip():
                findings.append(f"invalid {location} key")
                continue
            findings.extend(self._metadata_value_findings(metadata[key], f"{location}.{key}"))
        return tuple(findings)

    def _metadata_value_findings(self, value: Any, location: str) -> tuple[str, ...]:
        if isinstance(value, Mapping):
            return self._metadata_findings(value, location)
        if isinstance(value, (list, tuple)):
            findings: list[str] = []
            for index, item in enumerate(value):
                findings.extend(self._metadata_value_findings(item, f"{location}[{index}]"))
            return tuple(findings)
        if value is None or isinstance(value, (str, int, bool)):
            return ()
        if isinstance(value, float) and isfinite(value):
            return ()
        return (f"invalid {location} value",)
