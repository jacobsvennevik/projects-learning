#!/usr/bin/env python3
"""
split_pdf_sections.py — Split a PDF into coarse sections (chapters, parts, …).

Example for your book:
    python split_pdf_sections.py \
        textbook.pdf \
        "Foreword:10" "Preface:13" \
        "Ch01_DesigningForTheLearner:20" \
        "Ch02_IntegratedFramework:34" \
        "Ch03_CourseStructure:58" \
        "Ch04_ContentDesign:80" \
        "Ch05_LearningActivities:104" \
        "Ch06_SocialInteractions:122" \
        "Ch07_AssessmentFeedback:142" \
        "Ch08_PuttingItTogether:158"

…will generate
    01_Foreword.pdf
    02_Preface.pdf
    03_Ch01_DesigningForTheLearner.pdf
    …
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

try:
    from pypdf import PdfReader, PdfWriter            # pip install pypdf
except ImportError:
    sys.exit("Error: Install the dependency first — `pip install pypdf`")

PAGE_SPEC_RX = re.compile(r"^(.+?):(\d+)$")          # "Label:PageNumber"

def parse_specs(args: list[str]) -> list[tuple[str, int]]:
    """Return list of (label, 0-based start_page_index)."""
    specs: list[tuple[str, int]] = []
    for raw in args:
        m = PAGE_SPEC_RX.match(raw)
        if not m:
            sys.exit(f'Bad spec "{raw}". Expected form LABEL:PAGE')
        label, page = m.group(1), int(m.group(2))
        if page < 1:
            sys.exit("Page numbers must start at 1 (what you see in the viewer).")
        specs.append((label, page - 1))              # convert to 0-based
    # ensure ascending order
    if [p for _, p in specs] != sorted(p for _, p in specs):
        sys.exit("Page numbers must be in strictly ascending order.")
    return specs

def split_by_specs(
    pdf_path: Path,
    specs: list[tuple[str, int]],
    output_dir: Optional[Path] = None,
    empty_policy: str = "allow",  # "allow" | "skip" | "blank"
) -> None:
    # Be tolerant when reading slightly non-conformant PDFs
    reader = PdfReader(str(pdf_path), strict=False)
    total_pages = len(reader.pages)
    destination_base = output_dir if output_dir is not None else pdf_path.parent
    destination_base.mkdir(parents=True, exist_ok=True)

    # Add implicit "end" marker to walk in pairs
    specs_with_end = specs + [("END", total_pages)]
    for idx, ((label, start), (_, nxt_start)) in enumerate(
        zip(specs_with_end, specs_with_end[1:]), start=1
    ):
        writer = PdfWriter()
        # Add pages for this range using robust append API
        if start < nxt_start:
            # append expects a tuple (start, stop[, step]) or list of indices
            writer.append(reader, pages=(start, nxt_start))
        else:
            # Handle empty range based on policy
            if empty_policy == "skip":
                print(
                    f"⚠  Skipping empty section {idx:02d}_{label} (pages {start+1}–{nxt_start})"
                )
                continue
            if empty_policy == "blank":
                # Create a valid one-page PDF with a blank page (US Letter size)
                writer.add_blank_page(width=612, height=792)

        # Copy metadata safely: coerce to plain strings and drop odd objects
        safe_meta = {}
        try:
            for k, v in (reader.metadata or {}).items():
                if isinstance(v, (str, int, float)):
                    safe_meta[k] = str(v)
        except Exception:
            safe_meta = {}
        if safe_meta:
            writer.add_metadata(safe_meta)
        # Avoid copying page_layout/page_mode as they can hold odd objects

        safe_label = re.sub(r"[^\w\-]+", "_", label).strip("_")
        output_name = f"{idx:02d}_{safe_label}.pdf"
        out_path = destination_base / output_name
        try:
            with out_path.open("wb") as f_out:
                writer.write(f_out)
            print(f"✔  {out_path}  (pages {start+1}–{nxt_start})")
        except Exception as e:
            print(f"✖  Error writing {out_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(
            "USAGE:\n"
            "  python split_pdf_sections.py INPUT.pdf [--out OUTPUT_DIR] [--skip-empty | --blank-empty] "
            '"Label1:StartPage1" "Label2:StartPage2" …\n\n'
            "Example:\n"
            '  python split_pdf_sections.py report.pdf --out out/ --skip-empty "Intro:1" "Methods:9" "Results:23"'
        )
    pdf_file = Path(sys.argv[1]).expanduser().resolve()
    if not pdf_file.is_file():
        sys.exit(f'Input file "{pdf_file}" not found.')

    # Optional output directory flag
    raw_args = sys.argv[2:]
    out_dir: Optional[Path] = None
    if len(raw_args) >= 2 and raw_args[0] == "--out":
        out_dir = Path(raw_args[1]).expanduser().resolve()
        raw_args = raw_args[2:]
    empty_policy = "allow"
    if raw_args and raw_args[0] in ("--skip-empty", "--blank-empty"):
        empty_policy = "skip" if raw_args[0] == "--skip-empty" else "blank"
        raw_args = raw_args[1:]

    specifications = parse_specs(raw_args)
    split_by_specs(pdf_file, specifications, out_dir, empty_policy)