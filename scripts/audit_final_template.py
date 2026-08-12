#!/usr/bin/env python3
"""Read-only OPC inventory for the sole final BDIL publishing template."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any
from zipfile import BadZipFile, ZipFile

from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE_DIRECTORY = ROOT / "documents" / "Template Document"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}


class TemplateCandidateError(RuntimeError):
    """The publishing-template directory does not contain exactly one DOCX."""


def discover_final_template(directory: str | Path = DEFAULT_TEMPLATE_DIRECTORY) -> Path:
    """Resolve exactly one direct-child DOCX, failing closed otherwise."""

    root = Path(directory).expanduser().resolve()
    candidates = sorted(
        path.resolve()
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() == ".docx"
    ) if root.is_dir() else []
    if len(candidates) != 1:
        raise TemplateCandidateError(
            f"expected exactly one final DOCX in {root}; found {len(candidates)}"
        )
    return candidates[0]


def _xml(package: ZipFile, name: str):
    return etree.fromstring(package.read(name))


def inspect_template(path: str | Path) -> dict[str, Any]:
    """Return a deterministic package inventory without rewriting the DOCX."""

    template = Path(path).resolve()
    try:
        with ZipFile(template, "r") as package:
            bad_member = package.testzip()
            if bad_member is not None:
                raise BadZipFile(f"corrupt package member: {bad_member}")
            names = tuple(sorted(package.namelist()))
            required = {
                "[Content_Types].xml",
                "_rels/.rels",
                "word/document.xml",
                "word/_rels/document.xml.rels",
                "word/styles.xml",
                "word/numbering.xml",
                "word/theme/theme1.xml",
            }
            missing = sorted(required.difference(names))
            if missing:
                raise BadZipFile("missing required OPC parts: " + ", ".join(missing))

            document = _xml(package, "word/document.xml")
            body = document.find("w:body", NS)
            assert body is not None
            sections = document.xpath(
                "./w:body/w:p/w:pPr/w:sectPr | ./w:body/w:sectPr",
                namespaces=NS,
            )
            section_types = [
                section.xpath("string(w:type/@w:val)", namespaces=NS)
                or "implicit-nextPage"
                for section in sections
            ]
            headers = tuple(
                name for name in names if re.fullmatch(r"word/header\d+\.xml", name)
            )
            footers = tuple(
                name for name in names if re.fullmatch(r"word/footer\d+\.xml", name)
            )
            media = tuple(name for name in names if name.startswith("word/media/"))

            drawings = picts = text_boxes = page_fields = numpages_fields = 0
            for name in names:
                if not (name.startswith("word/") and name.endswith(".xml")):
                    continue
                try:
                    root = _xml(package, name)
                except etree.XMLSyntaxError:
                    continue
                drawings += len(root.xpath(".//w:drawing", namespaces=NS))
                picts += len(root.xpath(".//w:pict", namespaces=NS))
                text_boxes += len(root.xpath(".//w:txbxContent", namespaces=NS))
                instructions = [
                    "".join(node.itertext()).strip()
                    for node in root.xpath(".//w:instrText", namespaces=NS)
                ]
                page_fields += sum(value == "PAGE" for value in instructions)
                numpages_fields += sum(value == "NUMPAGES" for value in instructions)

            styles = _xml(package, "word/styles.xml")
            numbering = _xml(package, "word/numbering.xml")
            tables = body.findall("w:tbl", NS)
            table_rows = sum(len(table.findall("w:tr", NS)) for table in tables)
            table_cells = sum(
                len(row.findall("w:tc", NS))
                for table in tables
                for row in table.findall("w:tr", NS)
            )
            merged_cells = sum(
                len(table.xpath(".//w:gridSpan | .//w:vMerge", namespaces=NS))
                for table in tables
            )
            relationships = sum(
                len(_xml(package, name)) for name in names if name.endswith(".rels")
            )

            return {
                "path": str(template),
                "sha256": sha256(template.read_bytes()).hexdigest(),
                "package_valid": True,
                "package_parts": len(names),
                "body_children": len(body),
                "paragraphs": len(body.findall("w:p", NS)),
                "sections": len(sections),
                "section_types": section_types,
                "headers": len(headers),
                "footers": len(footers),
                "page_fields": page_fields,
                "numpages_fields": numpages_fields,
                "tables": len(tables),
                "table_rows": table_rows,
                "table_cells": table_cells,
                "merged_cell_markers": merged_cells,
                "media": len(media),
                "drawingml": drawings,
                "vml_pict": picts,
                "text_boxes": text_boxes,
                "styles": len(styles.findall("w:style", NS)),
                "abstract_numbering": len(numbering.findall("w:abstractNum", NS)),
                "numbering_instances": len(numbering.findall("w:num", NS)),
                "relationships": relationships,
                "has_theme": "word/theme/theme1.xml" in names,
                "has_content_types": "[Content_Types].xml" in names,
            }
    except (BadZipFile, KeyError, etree.XMLSyntaxError) as error:
        raise BadZipFile(f"invalid DOCX package: {template}: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-directory", type=Path, default=DEFAULT_TEMPLATE_DIRECTORY)
    args = parser.parse_args()
    template = discover_final_template(args.template_directory)
    print(json.dumps(inspect_template(template), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
