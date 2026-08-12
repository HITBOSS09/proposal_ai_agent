"""End-to-end coordination for generating one proposal section."""

from __future__ import annotations

from collections.abc import Sequence

from .contracts import SectionContent
from .models import ProjectModel
from .prompt_composer import PromptComposer
from .proposal_planner import ProposalPlanner
from .retrieval_executor import RetrievalExecutor
from .retrieval_planner import RetrievalPlanner
from .retrieval_query import SectionRetrievalQuery
from .section_generator import SectionGenerator


class ProposalSectionPipeline:
    """Coordinate the certified planning, retrieval, composition, and generation steps."""

    def __init__(
        self,
        proposal_planner: ProposalPlanner,
        retrieval_planner: RetrievalPlanner,
        retrieval_executor: RetrievalExecutor,
        prompt_composer: PromptComposer,
        section_generator: SectionGenerator,
    ) -> None:
        if not isinstance(proposal_planner, ProposalPlanner):
            raise TypeError("proposal_planner must be a ProposalPlanner")
        if not isinstance(retrieval_planner, RetrievalPlanner):
            raise TypeError("retrieval_planner must be a RetrievalPlanner")
        if not isinstance(retrieval_executor, RetrievalExecutor):
            raise TypeError("retrieval_executor must be a RetrievalExecutor")
        if not isinstance(prompt_composer, PromptComposer):
            raise TypeError("prompt_composer must be a PromptComposer")
        if not isinstance(section_generator, SectionGenerator):
            raise TypeError("section_generator must be a SectionGenerator")
        self._proposal_planner = proposal_planner
        self._retrieval_planner = retrieval_planner
        self._retrieval_executor = retrieval_executor
        self._prompt_composer = prompt_composer
        self._section_generator = section_generator

    def generate(
        self,
        project: ProjectModel,
        section_id: str,
        retrieval_queries: Sequence[SectionRetrievalQuery],
    ) -> SectionContent:
        """Generate one section from caller-supplied immutable retrieval queries."""
        if not isinstance(project, ProjectModel):
            raise TypeError("project must be a ProjectModel")
        if not isinstance(section_id, str) or not section_id.strip():
            raise ValueError("section_id must be a non-empty string")
        queries = tuple(retrieval_queries)
        if not queries:
            raise ValueError("retrieval_queries must contain at least one query")
        if any(not isinstance(query, SectionRetrievalQuery) for query in queries):
            raise TypeError("retrieval_queries must contain SectionRetrievalQuery values")
        if any(query.section_id != section_id for query in queries):
            raise ValueError("retrieval_queries must belong to section_id")

        proposal_plan = self._proposal_planner.plan(project)
        retrieval_plan = self._retrieval_planner.plan(proposal_plan)
        contexts = self._retrieval_executor.execute(queries)
        if len(contexts) != 1 or contexts[0].section_id != section_id:
            raise RuntimeError("retrieval execution did not produce exactly one section context")
        context = contexts[0]
        prompt = self._prompt_composer.compose(
            project,
            proposal_plan,
            retrieval_plan,
            section_id,
            style_references=context.style_references,
            technical_references=context.technical_references,
            blueprint_references=context.blueprint_references,
        )
        return self._section_generator.generate(prompt)
