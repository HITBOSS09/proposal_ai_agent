"""Focused integration coverage for the proposal retrieval intelligence bridge."""

import importlib.util
from pathlib import Path

from docx import Document as DocxDocument
from docx.enum.style import WD_STYLE_TYPE

from proposal_ai_agent.proposal_generation.models import ClientInformation, ProjectModel, Requirement
from proposal_ai_agent.proposal_generation.prompt_composer import RetrievedReference
from proposal_ai_agent.proposal_generation.proposal_planner import ProposalPlanner
from proposal_ai_agent.proposal_generation.retrieval_context import RetrievedContext
from proposal_ai_agent.proposal_generation.retrieval_planner import RetrievalPlanner
from proposal_ai_agent.proposal_generation.retrieval_query import ReferenceType, SectionRetrievalQuery
from proposal_ai_agent.proposal_generation.transport_contract import HeadingResponse, ProposalResponse, SectionResponse
from proposal_ai_agent.proposal_generation.word_style_contract import (
    BodyStyle, BulletStyle, CalloutStyle, CaptionStyle, CoverTitleStyle, FooterStyle,
    HeaderStyle, Heading1Style, Heading2Style, Heading3Style, ModuleBannerStyle,
    PageNumberStyle, ProposalTitleStyle, RequirementMatrixStyle, TableCellStyle,
    TableHeaderStyle,
)
from proposal_ai_agent.embeddings.providers import MockEmbeddingProvider
import pytest


def _project() -> ProjectModel:
    return ProjectModel(
        source_request_id="7d2e5f0f-dcca-4c93-9de8-2d7e04e33142",
        client=ClientInformation(name="Ground Truth Client"),
        proposal_title="Ground Truth Project",
        project_data={"quantity": 8},
        requirements=(Requirement(description="Use eight user-specified sensors"),),
        metadata={"proposal_type": "technical"},
    )


def _demo_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_proposal_demo.py"
    spec = importlib.util.spec_from_file_location("run_proposal_demo_bridge", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _template(path: Path) -> None:
    document = DocxDocument()
    for contract in (
        ProposalTitleStyle(), CoverTitleStyle(), ModuleBannerStyle(), Heading1Style(), Heading2Style(), Heading3Style(),
        BodyStyle(), BulletStyle(), TableHeaderStyle(), TableCellStyle(), RequirementMatrixStyle(), CalloutStyle(),
        CaptionStyle(), HeaderStyle(), FooterStyle(), PageNumberStyle(),
    ):
        document.styles.add_style(contract.template_style_name, WD_STYLE_TYPE.PARAGRAPH)
    document.save(path)


def test_canonical_plan_produces_typed_retrieval_queries() -> None:
    plan = ProposalPlanner().plan(_project())
    retrieval_planner = RetrievalPlanner()

    queries = retrieval_planner.queries(retrieval_planner.plan(plan), max_results=4)

    assert queries
    assert all(isinstance(query, SectionRetrievalQuery) for query in queries)
    assert all(query.max_results == 4 for query in queries)
    assert {query.reference_type for query in queries} == {
        ReferenceType.AUTHORING,
        ReferenceType.TECHNICAL,
        ReferenceType.BLUEPRINT,
    }
    assert not any(query.section_id in {"cover", "references"} for query in queries)


def test_runtime_retrieval_builder_calls_retrieval_executor(monkeypatch) -> None:
    module = _demo_module()
    seen = {}

    class Vectors:
        size = 3

    class Params:
        vectors = Vectors()

    class Config:
        params = Params()

    class Collection:
        config = Config()

    class Client:
        def __init__(self, **kwargs):
            seen["client_configuration"] = kwargs

        def collection_exists(self, name):
            return True

        def get_collection(self, name):
            return Collection()

    class Executor:
        def __init__(self, retriever):
            seen["retriever"] = retriever

        def execute(self, queries):
            seen["queries"] = queries
            return ()

    monkeypatch.setattr(module, "QdrantClient", Client)
    monkeypatch.setattr(module, "OllamaEmbeddingProvider", lambda *args: MockEmbeddingProvider(dimensions=3))
    monkeypatch.setattr(module, "RetrievalExecutor", Executor)
    plan = ProposalPlanner().plan(_project())

    queries, contexts = module._retrieve_contexts(
        plan,
        collection_name="proposal-collection",
        qdrant_url="http://qdrant.test",
        ollama_url="http://ollama.test/api",
        embedding_model="embedding-test",
        timeout=10,
        max_results=2,
    )

    assert queries == seen["queries"]
    assert queries
    assert contexts == ()
    assert seen["retriever"].__class__ is module.QdrantProposalRetriever
    request = seen["retriever"]._request_builder.build(queries[0])
    assert request.retrieval_profile == "proposal-section"
    assert request.query_embedding.processed_query.qualified_query.intent == "PROPOSAL_RETRIEVAL"


def test_retrieved_context_enters_structured_runtime_and_all_downstream_stages(tmp_path, monkeypatch) -> None:
    module = _demo_module()
    template = tmp_path / "master.docx"
    _template(template)
    project = _project()
    plan = ProposalPlanner().plan(project)
    query = SectionRetrievalQuery(
        section_id="executive-summary",
        reference_type=ReferenceType.AUTHORING,
        query_text="Executive Summary authoring",
        max_results=2,
    )
    reference = RetrievedReference(
        reference_id="REF-A",
        reference_type="authoring",
        chunk_id="chunk-a",
        source_document="old-proposal.docx",
        score=0.9,
        content="Legacy Client project dated 2020 used twelve sensors with a confidential budget.",
    )
    context = RetrievedContext(
        section_id="executive-summary",
        queries=(query,),
        style_references=(reference,),
    )
    seen = {}

    class Provider:
        provider_name = "test"
        model_name = "test-model"
        provider_metadata = {}

        def generate(self, prompt):
            seen["prompt"] = prompt
            return ProposalResponse(
                proposal_id=prompt.proposal_id,
                title=project.proposal_title,
                sections=tuple(
                    SectionResponse(
                        section_id=section.section_id,
                        heading=HeadingResponse(text=section.title, level=1),
                    )
                    for section in plan.sections
                ),
            )

    monkeypatch.setattr(module, "_project", lambda prompt_text=None: project)
    monkeypatch.setattr(module, "_retrieve_contexts", lambda *args, **kwargs: ((query,), (context,)))
    monkeypatch.setattr(module.ProposalLLMFactory, "create", lambda *args, **kwargs: Provider())
    monkeypatch.setattr(module, "ROOT", tmp_path)

    for name, cls, method in (
        ("validator", module.ProposalTransportValidator, "validate"),
        ("mapper", module.ProposalTransportMapper, "map"),
        ("composition", module.CompositionEngine, "compose"),
        ("compiler", module.DOCXCompiler, "compile"),
    ):
        original = getattr(cls, method)

        def wrapper(self, *args, _original=original, _name=name, **kwargs):
            seen[_name] = args
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(cls, method, wrapper)

    args = module.parse_args(["--template", str(template), "--collection", "test-proposals"])
    assert module.run(args) == 0

    prompt = seen["prompt"]
    assert prompt.expected_output_model is ProposalResponse
    assert prompt.style_references == (reference,)
    assert "[USER_GROUND_TRUTH]" in prompt.user_prompt
    assert "Ground Truth Client" in prompt.user_prompt
    assert '"quantity": 8' in prompt.user_prompt
    assert "[AUTHORING_REFERENCES]" in prompt.user_prompt
    assert "Legacy Client" in prompt.user_prompt
    assert "User Input > Technical References > Authoring References > Blueprint References" in prompt.system_prompt
    assert "Markdown" in prompt.system_prompt and "Do not generate Markdown" in prompt.system_prompt
    assert {"validator", "mapper", "composition", "compiler"} <= seen.keys()
    assert seen["composition"][1].proposal_id == str(project.source_request_id)
    assert (tmp_path / "output" / "proposal.docx").is_file()
    assert (tmp_path / "output" / "proposal.json").is_file()

    active_source = (Path(__file__).resolve().parents[1] / "scripts" / "run_proposal_demo.py").read_text()
    assert "ProposalSectionPipeline" not in active_source
    assert "SectionContent" not in active_source
    assert "ProposalContent" not in active_source


def _response_for_ids(ids: tuple[str, ...]) -> ProposalResponse:
    return ProposalResponse(
        proposal_id=str(_project().source_request_id),
        title="Proposal",
        sections=tuple(
            SectionResponse(section_id=section_id, heading=HeadingResponse(text=section_id, level=1))
            for section_id in ids
        ),
    )


def test_exact_section_plan_conformance_passes() -> None:
    module = _demo_module()
    plan = ProposalPlanner().plan(_project())
    expected = tuple(section.section_id for section in plan.sections)

    assert module._validate_plan_conformance(plan, _response_for_ids(expected)) is None


@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    (
        (lambda ids: ids[:-1], "Missing IDs"),
        (lambda ids: (*ids, "unexpected"), "Unexpected IDs"),
        (lambda ids: ("renamed", *ids[1:]), "Unexpected IDs"),
        (lambda ids: (*ids, ids[-1]), "Duplicate IDs"),
        (lambda ids: (ids[1], ids[0], *ids[2:]), "Order mismatch: True"),
    ),
)
def test_section_plan_conformance_fails_closed_with_diagnostics(mutation, diagnostic) -> None:
    module = _demo_module()
    plan = ProposalPlanner().plan(_project())
    expected = tuple(section.section_id for section in plan.sections)
    generated = mutation(expected)

    with pytest.raises(module.PlanConformanceError) as error:
        module._validate_plan_conformance(plan, _response_for_ids(generated))

    message = str(error.value)
    assert f"Expected section IDs:\n{list(expected)}" in message
    assert f"Generated section IDs:\n{list(generated)}" in message
    assert diagnostic in message


def test_nested_unplanned_section_fails_closed() -> None:
    module = _demo_module()
    plan = ProposalPlanner().plan(_project())
    expected = tuple(section.section_id for section in plan.sections)
    sections = list(_response_for_ids(expected).sections)
    sections[0] = sections[0].model_copy(
        update={
            "children": (
                SectionResponse(
                    section_id="invented-child",
                    heading=HeadingResponse(text="Invented", level=2),
                ),
            )
        }
    )

    with pytest.raises(module.PlanConformanceError, match="invented-child"):
        module._validate_plan_conformance(
            plan,
            ProposalResponse(proposal_id=str(_project().source_request_id), title="Proposal", sections=tuple(sections)),
        )


def _configure_retry_runtime(module, tmp_path, monkeypatch, provider):
    template = tmp_path / "retry-master.docx"
    _template(template)
    project = _project()
    monkeypatch.setattr(module, "_project", lambda prompt_text=None: project)
    monkeypatch.setattr(module, "_retrieve_contexts", lambda *args, **kwargs: ((), ()))
    monkeypatch.setattr(module.ProposalLLMFactory, "create", lambda *args, **kwargs: provider)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return project, module.parse_args(["--template", str(template), "--collection", "test-proposals"])


def test_first_mismatch_triggers_one_retry_and_corrected_response_proceeds(tmp_path, monkeypatch) -> None:
    module = _demo_module()
    calls = []
    project = _project()
    plan = ProposalPlanner().plan(project)
    expected = tuple(section.section_id for section in plan.sections)

    class Provider:
        provider_name = "test"
        model_name = "test-model"
        provider_metadata = {}

        def generate(self, prompt):
            calls.append(prompt)
            return _response_for_ids(()) if len(calls) == 1 else _response_for_ids(expected)

    _, args = _configure_retry_runtime(module, tmp_path, monkeypatch, Provider())

    assert module.run(args) == 0
    assert len(calls) == 2
    assert "[SECTION_CONTRACT_CORRECTION]" in calls[1].user_prompt
    assert f"Expected IDs: {list(expected)}" in calls[1].user_prompt
    assert "Received IDs: []" in calls[1].user_prompt
    assert (tmp_path / "output" / "proposal.docx").is_file()


def test_second_conformance_mismatch_fails_closed_without_third_attempt(tmp_path, monkeypatch, capsys) -> None:
    module = _demo_module()
    calls = []

    class Provider:
        provider_name = "test"
        model_name = "test-model"
        provider_metadata = {}

        def generate(self, prompt):
            calls.append(prompt)
            return _response_for_ids(())

    _, args = _configure_retry_runtime(module, tmp_path, monkeypatch, Provider())

    assert module.run(args) == 1
    assert len(calls) == 2
    captured = capsys.readouterr()
    assert "Expected section IDs:" in captured.err
    assert "Generated section IDs:" in captured.err
    assert "FAILED at [7/10] Verifying corrective plan conformance" in captured.err


def test_provider_failure_is_not_treated_as_conformance_retry(tmp_path, monkeypatch) -> None:
    module = _demo_module()
    calls = 0

    class Provider:
        provider_name = "test"
        model_name = "test-model"
        provider_metadata = {}

        def generate(self, prompt):
            nonlocal calls
            calls += 1
            raise RuntimeError("provider unavailable")

    _, args = _configure_retry_runtime(module, tmp_path, monkeypatch, Provider())

    assert module.run(args) == 1
    assert calls == 1
