"""Deterministic assembly of generated proposal sections into a domain document."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from .contracts import ProposalDocument, SectionContent
from .models import ProposalPlan
from .prompt_composer import RetrievedReference


class ProposalAssembler:
    """Assemble complete proposal content without retrieval, generation, or rendering."""

    def assemble(
        self,
        proposal_plan: ProposalPlan,
        sections: Sequence[SectionContent],
        *,
        references: Sequence[RetrievedReference] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> ProposalDocument:
        """Return a proposal document ordered exactly as the supplied proposal plan."""
        started_at = perf_counter()
        print(
            f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds')} PROPOSAL_ASSEMBLER enter sections={len(sections)} references={len(references)}",
            flush=True,
        )
        if not isinstance(proposal_plan, ProposalPlan):
            raise TypeError("proposal_plan must be a ProposalPlan")
        ordered_sections = tuple(sections)
        ordered_references = tuple(references)
        if any(not isinstance(section, SectionContent) for section in ordered_sections):
            raise TypeError("sections must contain SectionContent values")
        if any(not isinstance(reference, RetrievedReference) for reference in ordered_references):
            raise TypeError("references must contain RetrievedReference values")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        planned_ids = tuple(section.section_id for section in proposal_plan.sections)
        if len(set(planned_ids)) != len(planned_ids):
            raise ValueError("proposal_plan contains duplicate section identifiers")

        supplied_ids = tuple(section.section_id for section in ordered_sections)
        duplicates = tuple(section_id for section_id in dict.fromkeys(supplied_ids) if supplied_ids.count(section_id) > 1)
        if duplicates:
            raise ValueError(f"duplicate SectionContent supplied for sections: {', '.join(duplicates)}")

        supplied_by_id = {section.section_id: section for section in ordered_sections}
        missing = tuple(section_id for section_id in planned_ids if section_id not in supplied_by_id)
        if missing:
            raise ValueError(f"missing SectionContent for sections: {', '.join(missing)}")
        unexpected = tuple(section_id for section_id in supplied_ids if section_id not in set(planned_ids))
        if unexpected:
            raise ValueError(f"SectionContent supplied for unplanned sections: {', '.join(unexpected)}")

        document = ProposalDocument(
            proposal_plan=proposal_plan,
            proposal_title=proposal_plan.project.proposal_title,
            sections=tuple(supplied_by_id[section_id] for section_id in planned_ids),
            references=ordered_references,
            metadata={} if metadata is None else metadata,
        )
        print(
            f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds')} PROPOSAL_ASSEMBLER exit elapsed_ms={(perf_counter() - started_at) * 1000.0:.1f}",
            flush=True,
        )
        return document
