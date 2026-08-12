#!/usr/bin/env python3
"""Run the structured Proposal Compiler against the configured Ollama provider."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proposal_ai_agent.proposal_generation.composition import CompositionEngine
from proposal_ai_agent.proposal_generation.document_plan import ProposalPlan, SectionPlan, SectionRole
from proposal_ai_agent.proposal_generation.docx_compiler import DOCXCompiler
from proposal_ai_agent.proposal_generation.models import ClientInformation, ProjectModel, Requirement
from proposal_ai_agent.proposal_generation.prompt_composer import PromptComposer
from proposal_ai_agent.proposal_generation.proposal_planner import ProposalPlanner
from proposal_ai_agent.proposal_generation.providers.factory import ProposalLLMFactory
from proposal_ai_agent.proposal_generation.retrieval_executor import RetrievalExecutor
from proposal_ai_agent.proposal_generation.retrieval_planner import RetrievalPlanner
from proposal_ai_agent.proposal_generation.retrieval_request_builder import ProposalRetrievalRequestBuilder
from proposal_ai_agent.proposal_generation.retrievers.qdrant import QdrantProposalRetriever
from proposal_ai_agent.proposal_generation.section_generator import SectionGenerator
from proposal_ai_agent.proposal_generation.transport_mapper import ProposalTransportMapper
from proposal_ai_agent.proposal_generation.transport_validator import ProposalTransportValidator
from proposal_ai_agent.proposal_generation.word_style_contract import (
    BodyStyle,
    BulletStyle,
    CalloutStyle,
    CaptionStyle,
    CoverTitleStyle,
    FooterStyle,
    HeaderStyle,
    Heading1Style,
    Heading2Style,
    Heading3Style,
    ModuleBannerStyle,
    PageNumberStyle,
    ProposalTitleStyle,
    RequirementMatrixStyle,
    TableCellStyle,
    TableHeaderStyle,
    WordStyleContract,
)
from proposal_ai_agent.online.contracts import SearchScope
from proposal_ai_agent.online.benchmarks import BenchmarkProfile, BenchmarkRegistry
from proposal_ai_agent.online.engines import QueryEngine
from proposal_ai_agent.online.providers.embedding import OllamaEmbeddingProvider
from proposal_ai_agent.online.repositories import QdrantVectorRepository


class ProposalDemoError(RuntimeError):
    """A user-facing failure from the developer proposal demo."""


class PlanConformanceError(ProposalDemoError):
    """A structured response does not conform to the canonical proposal plan."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a structured proposal through the Proposal Compiler.")
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(os.environ["BDIL_PROPOSAL_MASTER_TEMPLATE"]) if "BDIL_PROPOSAL_MASTER_TEMPLATE" in os.environ else None,
        help="Existing master .docx template (or set BDIL_PROPOSAL_MASTER_TEMPLATE)",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("BDIL_PROPOSAL_COLLECTION", "bdil_reference"),
        help="Existing Qdrant collection containing proposal references",
    )
    parser.add_argument(
        "--retrieval-results",
        type=int,
        default=int(os.getenv("BDIL_PROPOSAL_RETRIEVAL_RESULTS", "3")),
        help="Maximum references retrieved for each section/reference query",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Text file whose complete contents are used as the temporary project requirement input",
    )
    args = parser.parse_args(argv)
    if args.template is None or not args.template.is_file():
        parser.error("--template must identify an existing master .docx template")
    if args.prompt_file is not None and not args.prompt_file.is_file():
        parser.error("--prompt-file must identify an existing text file")
    if not args.collection.strip():
        parser.error("--collection must not be empty")
    if args.retrieval_results <= 0:
        parser.error("--retrieval-results must be positive")
    return args


def _project(prompt_text: str | None = None) -> ProjectModel:
    """Build demo-only project facts without ingestion, retrieval, or planning I/O."""
    if prompt_text is not None:
        return ProjectModel(
            source_request_id=UUID("00000000-0000-0000-0000-000000000001"),
            client=ClientInformation(name="Prompt File Request"),
            proposal_title="Proposal Request",
            requirements=(Requirement(identifier="PROMPT-1", description=prompt_text),),
            metadata={"proposal_type": "technical", "input_source": "prompt_file"},
        )
    return ProjectModel(
        source_request_id=UUID("00000000-0000-0000-0000-000000000001"),
        client=ClientInformation(name="Aegis Perimeter Systems", attributes={"industry": "defence"}),
        proposal_title="Autonomous Perimeter Monitoring Platform Proposal",
        project_data={
            "platform": "Autonomous fixed-wing and multi-rotor surveillance platform",
            "ai_accuracy": "Minimum 95% object detection accuracy in validated operating conditions",
            "camera_configuration": "EO/IR gimballed camera with day and thermal imaging",
            "battery": "Minimum 90-minute endurance with field-swappable battery packs",
            "communication_range": "Encrypted command-and-control link up to 15 km line-of-sight",
            "power": "AC mains charging with vehicle and portable generator support",
            "environmental_conditions": "Operate from -10°C to 50°C, dust, light rain, and winds up to 35 km/h",
        },
        metadata={"proposal_type": "technical"},
    )


def _walk_sections(sections):
    for section in sections:
        yield section
        yield from _walk_sections(section.children)


def _validate_plan_conformance(authoring_plan, response) -> None:
    """Fail closed when generated sections differ from the canonical plan."""
    expected_ids = tuple(section.section_id for section in authoring_plan.sections)
    generated_ids = tuple(section.section_id for section in response.sections)
    flattened_ids = tuple(section.section_id for section in _walk_sections(response.sections))
    if generated_ids == expected_ids and flattened_ids == generated_ids:
        return

    expected_set = set(expected_ids)
    generated_set = set(flattened_ids)
    missing = tuple(section_id for section_id in expected_ids if section_id not in generated_set)
    unexpected = tuple(section_id for section_id in flattened_ids if section_id not in expected_set)
    duplicates = tuple(
        section_id for section_id in dict.fromkeys(flattened_ids) if flattened_ids.count(section_id) > 1
    )
    order_mismatch = (
        not missing
        and not unexpected
        and not duplicates
        and generated_ids != expected_ids
    )
    details = (
        "transport response violates the canonical proposal plan\n"
        f"Expected section IDs:\n{list(expected_ids)}\n"
        f"Generated section IDs:\n{list(flattened_ids)}\n"
        f"Missing IDs: {list(missing)}\n"
        f"Unexpected IDs: {list(unexpected)}\n"
        f"Duplicate IDs: {list(duplicates)}\n"
        f"Order mismatch: {order_mismatch}"
    )
    raise PlanConformanceError(details)


def _corrective_prompt(prompt, authoring_plan, response):
    """Add one concise plan-conformance correction without changing generation contracts."""
    expected_ids = [section.section_id for section in authoring_plan.sections]
    generated_ids = [section.section_id for section in _walk_sections(response.sections)]
    correction = "\n".join(
        (
            "[SECTION_CONTRACT_CORRECTION]",
            "Your previous response violated the planned section contract.",
            f"Expected IDs: {expected_ids}",
            f"Received IDs: {generated_ids}",
            "Regenerate the complete ProposalResponse using exactly the expected IDs once each and in the exact order.",
            "Do not rename, add, omit, duplicate, nest, or reorder sections.",
            "Return a valid minimal section when facts are limited.",
            "Do not change user facts or add unsupported content.",
        )
    )
    return prompt.model_copy(update={"user_prompt": f"{correction}\n\n{prompt.user_prompt}"})


def _document_plan(authoring_plan, document) -> ProposalPlan:
    """Adapt the canonical authoring plan to the existing composition plan."""
    section_plans: list[SectionPlan] = []
    for section in authoring_plan.sections:
        try:
            role = SectionRole(str(section.metadata.get("document_role", SectionRole.BODY.value)))
        except ValueError as error:
            raise ProposalDemoError(f"canonical plan contains unsupported document role for {section.section_id}") from error
        section_plans.append(
            SectionPlan(
                section_id=section.section_id,
                role=role,
                include_in_toc=role not in {SectionRole.COVER, SectionRole.REFERENCES},
                numbering_enabled=role not in {SectionRole.COVER, SectionRole.REFERENCES},
            )
        )
    return ProposalPlan(proposal_id=document.proposal_id, sections=tuple(section_plans))


def _collection_dimension(collection: object) -> int:
    vectors = collection.config.params.vectors
    size = getattr(vectors, "size", None)
    if isinstance(size, int) and size > 0:
        return size
    raise ProposalDemoError("proposal collection must use one unnamed dense vector")


def _retrieve_contexts(
    authoring_plan,
    *,
    collection_name: str,
    qdrant_url: str,
    ollama_url: str,
    embedding_model: str,
    timeout: int,
    max_results: int,
):
    """Execute the existing proposal retrieval contracts for a canonical plan."""
    client = QdrantClient(url=qdrant_url, timeout=timeout)
    if not client.collection_exists(collection_name):
        raise ProposalDemoError(f"Qdrant collection '{collection_name}' does not exist")
    dimension = _collection_dimension(client.get_collection(collection_name))
    query_engine = QueryEngine(
        registry=BenchmarkRegistry(
            (
                BenchmarkProfile(
                    intent_id="PROPOSAL_RETRIEVAL",
                    required_parameters=("question",),
                    optional_parameters=(),
                    is_default=True,
                ),
            )
        ),
        embedding_provider=OllamaEmbeddingProvider(embedding_model, ollama_url, timeout),
        embedding_dimension=dimension,
        embedding_model_id=embedding_model,
    )
    retrieval_planner = RetrievalPlanner()
    retrieval_plan = retrieval_planner.plan(authoring_plan)
    queries = retrieval_planner.queries(retrieval_plan, max_results=max_results)
    request_builder = ProposalRetrievalRequestBuilder(
        query_engine,
        search_scope=SearchScope(collection=collection_name),
    )
    retriever = QdrantProposalRetriever(
        request_builder,
        QdrantVectorRepository(client, collection_name, timeout),
    )
    return queries, RetrievalExecutor(retriever).execute(queries)


def _style_contracts() -> dict[str, WordStyleContract]:
    return {
        "proposal_title": ProposalTitleStyle(),
        "cover_title": CoverTitleStyle(),
        "module_banner": ModuleBannerStyle(),
        "heading_1": Heading1Style(),
        "heading_2": Heading2Style(),
        "heading_3": Heading3Style(),
        "body": BodyStyle(),
        "bullet": BulletStyle(),
        "table_header": TableHeaderStyle(),
        "table_cell": TableCellStyle(),
        "requirement_matrix": RequirementMatrixStyle(),
        "callout": CalloutStyle(),
        "caption": CaptionStyle(),
        "header": HeaderStyle(),
        "footer": FooterStyle(),
        "page_number": PageNumberStyle(),
    }


def run(args: argparse.Namespace) -> int:
    load_dotenv(ROOT / ".env")
    ollama_url = os.getenv("BDIL_OLLAMA_URL", "http://localhost:11434/api")
    llm_model = os.getenv("BDIL_PROPOSAL_OLLAMA_MODEL", os.getenv("BDIL_OLLAMA_MODEL", "qwen2.5:3b"))
    proposal_ollama_timeout = int(os.getenv("BDIL_PROPOSAL_OLLAMA_TIMEOUT", "300"))
    provider_timeout = int(os.getenv("BDIL_PROVIDER_TIMEOUT", "60"))
    qdrant_url = os.getenv("BDIL_QDRANT_URL", "http://localhost:6333")
    embedding_model = os.getenv("BDIL_EMBEDDING_MODEL", "bge-m3")
    stage = "initialization"
    try:
        stage = "[1/10] Building ProjectModel"
        print(f"{stage}...")
        prompt_text = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file is not None else None
        project = _project(prompt_text)

        stage = "[2/10] Planning proposal"
        print(f"{stage}...")
        authoring_plan = ProposalPlanner().plan(project)
        print(f"Proposal plan: sections={len(authoring_plan.sections)}")

        stage = "[3/10] Retrieving proposal references"
        print(f"{stage}...")
        queries, contexts = _retrieve_contexts(
            authoring_plan,
            collection_name=args.collection,
            qdrant_url=qdrant_url,
            ollama_url=ollama_url,
            embedding_model=embedding_model,
            timeout=provider_timeout,
            max_results=args.retrieval_results,
        )
        query_counts = {section.section_id: 0 for section in authoring_plan.sections}
        reference_counts = {section.section_id: 0 for section in authoring_plan.sections}
        for query in queries:
            query_counts[query.section_id] += 1
        for context in contexts:
            reference_counts[context.section_id] = (
                len(context.style_references)
                + len(context.technical_references)
                + len(context.blueprint_references)
            )
        for section in authoring_plan.sections:
            print(
                f"Retrieval: section={section.section_id} "
                f"queries={query_counts[section.section_id]} "
                f"references={reference_counts[section.section_id]}"
            )

        stage = "[4/10] Composing structured proposal prompt"
        print(f"{stage}...")
        prompt = PromptComposer().compose(project, authoring_plan, contexts)

        stage = "[5/10] Generating ProposalResponse through Ollama"
        print(f"{stage}...")
        provider = ProposalLLMFactory().create(
            "ollama", base_url=ollama_url, model=llm_model, timeout=proposal_ollama_timeout
        )
        response = SectionGenerator(provider).generate(prompt)
        print("Generation: ProposalResponse OK")

        stage = "[6/10] Validating transport response"
        print(f"{stage}...")
        ProposalTransportValidator().validate(response)
        print("Transport validation: OK")

        stage = "[7/10] Verifying plan conformance and mapping Proposal IR"
        print(f"{stage}...")
        try:
            _validate_plan_conformance(authoring_plan, response)
        except PlanConformanceError as first_error:
            print(str(first_error), file=sys.stderr)
            print("Plan conformance: retrying generation once")
            stage = "[5/10] Corrective ProposalResponse generation"
            response = SectionGenerator(provider).generate(
                _corrective_prompt(prompt, authoring_plan, response)
            )
            print("Generation retry: ProposalResponse OK")
            stage = "[6/10] Validating corrective transport response"
            ProposalTransportValidator().validate(response)
            print("Corrective transport validation: OK")
            stage = "[7/10] Verifying corrective plan conformance"
            _validate_plan_conformance(authoring_plan, response)
        print("Plan conformance: OK")
        document = ProposalTransportMapper().map(response)
        plan = _document_plan(authoring_plan, document)
        print("Transport mapping: OK")
        print("Proposal IR: OK")

        stage = "[8/10] Composing semantic document"
        print(f"{stage}...")
        composition = CompositionEngine().compose(plan, document)
        print("Composition: OK")

        stage = "[9/10] Compiling DOCX from master template"
        print(f"{stage}...")
        output_dir = ROOT / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        document_path = DOCXCompiler(_style_contracts()).compile(
            composition, args.template, output_dir / "proposal.docx"
        )

        print(f"DOCX: {document_path.relative_to(ROOT)}")

        stage = "[10/10] Writing ProposalResponse JSON"
        print(f"{stage}...")
        json_path = output_dir / "proposal.json"
        json_path.write_text(
            json.dumps(response.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[10/10] Proposal written to {document_path.relative_to(ROOT)}")
        print(f"[10/10] Proposal JSON written to {json_path.relative_to(ROOT)}")
        return 0
    except Exception as error:
        print(f"FAILED at {stage}: {type(error).__name__}: {error}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
