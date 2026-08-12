"""Template-preserving DOCX publishing infrastructure."""

from .certified_binding import (
    BindingAuthorizationError,
    CertifiedClonedRow,
    CertifiedContracts,
    CertifiedTemplateBinding,
    StructuralOperation,
)

from .openxml_editor import (
    LocatorMismatch,
    OpenXmlEditor,
    TemplateIntegrityError,
    UnsafeOpenXmlOperation,
    assert_footer_contract,
    copy_template,
    file_sha256,
)
from .template_map import (
    ComponentPrototype,
    DocumentRegion,
    ElementKind,
    PRU_TEMPLATE_SHA256,
    ReusableComponent,
    SemanticField,
    SemanticTarget,
    Story,
    StructuralLocator,
    TemplateSemanticMap,
    pru_template_semantic_map,
)
from .template_publisher import MarkdownContentError, TemplatePublisher, UnsupportedTemplateComponent

__all__ = [
    "BindingAuthorizationError", "CertifiedClonedRow", "CertifiedContracts", "CertifiedTemplateBinding",
    "ComponentPrototype", "DocumentRegion", "ElementKind", "LocatorMismatch",
    "OpenXmlEditor", "PRU_TEMPLATE_SHA256", "ReusableComponent", "SemanticField",
    "SemanticTarget", "Story", "StructuralLocator", "TemplateIntegrityError",
    "TemplateSemanticMap", "UnsafeOpenXmlOperation", "assert_footer_contract",
    "copy_template", "file_sha256",
    "pru_template_semantic_map", "MarkdownContentError", "TemplatePublisher",
    "StructuralOperation", "UnsupportedTemplateComponent",
]
