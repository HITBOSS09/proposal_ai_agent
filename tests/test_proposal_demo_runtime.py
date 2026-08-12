"""Runtime wiring coverage for the structured proposal compiler demo."""

import importlib.util
from pathlib import Path

from docx import Document as DocxDocument
from docx.enum.style import WD_STYLE_TYPE

from proposal_ai_agent.proposal_generation.transport_contract import HeadingResponse, ProposalResponse, SectionResponse
from proposal_ai_agent.proposal_generation.word_style_contract import (
    BodyStyle, BulletStyle, CalloutStyle, CaptionStyle, CoverTitleStyle, FooterStyle,
    HeaderStyle, Heading1Style, Heading2Style, Heading3Style, ModuleBannerStyle,
    PageNumberStyle, ProposalTitleStyle, RequirementMatrixStyle, TableCellStyle,
    TableHeaderStyle,
)


def _demo_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_proposal_demo.py"
    spec = importlib.util.spec_from_file_location("run_proposal_demo", path)
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


class _Provider:
    provider_name = "test"
    model_name = "test-model"
    provider_metadata = {}

    def generate(self, prompt) -> ProposalResponse:
        headings = (
            ("cover", "Proposal"),
            ("executive-summary", "Executive Summary"),
            ("solution-overview", "Solution Overview"),
            ("scope-and-deliverables", "Scope and Deliverables"),
            ("delivery-plan", "Delivery Plan"),
            ("commercials", "Commercials"),
            ("references", "References"),
        )
        return ProposalResponse(
            proposal_id=prompt.proposal_id,
            title="Proposal",
            sections=tuple(
                SectionResponse(section_id=section_id, heading=HeadingResponse(text=heading, level=1))
                for section_id, heading in headings
            ),
        )


def test_demo_uses_structured_transport_to_docx_compiler_only(tmp_path, monkeypatch) -> None:
    module = _demo_module()
    template = tmp_path / "master_template.docx"
    _template(template)
    monkeypatch.setattr(module.ProposalLLMFactory, "create", lambda *args, **kwargs: _Provider())
    monkeypatch.setattr(module, "_retrieve_contexts", lambda *args, **kwargs: ((), ()))
    monkeypatch.setattr(module, "ROOT", tmp_path)

    args = module.parse_args(["--template", str(template)])

    assert module.run(args) == 0
    assert (tmp_path / "output" / "proposal.docx").is_file()
    assert (tmp_path / "output" / "proposal.json").is_file()


def test_proposal_runtime_collection_precedence(tmp_path, monkeypatch) -> None:
    module = _demo_module()
    template = tmp_path / "master_template.docx"
    _template(template)

    monkeypatch.delenv("BDIL_PROPOSAL_COLLECTION", raising=False)
    assert module.parse_args(["--template", str(template)]).collection == "bdil_reference"

    monkeypatch.setenv("BDIL_PROPOSAL_COLLECTION", "operator-reference")
    assert module.parse_args(["--template", str(template)]).collection == "operator-reference"
    assert module.parse_args(
        ["--template", str(template), "--collection", "explicit-reference"]
    ).collection == "explicit-reference"


def test_active_runtime_boundary_has_no_legacy_pipeline_imports() -> None:
    root = Path(__file__).resolve().parents[1]
    active_files = (
        root / "scripts" / "run_proposal_demo.py",
        root / "src" / "proposal_ai_agent" / "proposal_generation" / "__init__.py",
        root / "src" / "proposal_ai_agent" / "proposal_generation" / "prompt_composer.py",
        root / "src" / "proposal_ai_agent" / "proposal_generation" / "section_generator.py",
        root / "src" / "proposal_ai_agent" / "proposal_generation" / "providers" / "provider.py",
        root / "src" / "proposal_ai_agent" / "proposal_generation" / "providers" / "ollama.py",
    )
    forbidden = (
        "proposal_generation.contracts",
        "SectionContent",
        "ProposalContent",
        "ProposalAssembler",
        "ProposalReviewer",
        "DOCXRenderer",
    )

    for path in active_files:
        source = path.read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden), path
