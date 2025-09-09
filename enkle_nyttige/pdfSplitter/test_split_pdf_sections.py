#!/usr/bin/env python3
"""
Tests for split_pdf_sections.py
"""
import io
from pathlib import Path
import tempfile
import pytest
from pypdf import PdfWriter, PdfReader

from split_pdf_sections import split_by_specs, parse_specs

def make_test_pdf(path: Path, num_pages: int):
    """
    Create a synthetic PDF with num_pages blank pages.
    """
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=72, height=72)
    with open(path, "wb") as f:
        writer.write(f)

def test_split_pdf_sections(tmp_path: Path):
    """Test basic functionality of splitting a PDF into sections."""
    # Create a dummy PDF with 10 pages
    pdf_path = tmp_path / "dummy.pdf"
    make_test_pdf(pdf_path, num_pages=10)

    # Split into 3 sections: pages 1–3, 4–6, 7–10
    specs = [
        ("PartA", 0),  # page 1 in human terms
        ("PartB", 3),  # page 4
        ("PartC", 6),  # page 7
    ]

    split_by_specs(pdf_path, specs)

    # Verify output files exist and page counts match
    expected = {
        "01_PartA.pdf": 3,
        "02_PartB.pdf": 3,
        "03_PartC.pdf": 4,
    }

    for filename, expected_pages in expected.items():
        file_path = tmp_path / filename
        assert file_path.exists(), f"{file_path} not created"
        reader = PdfReader(str(file_path))
        assert len(reader.pages) == expected_pages, \
            f"{filename} has {len(reader.pages)} pages, expected {expected_pages}"

def test_empty_sections(tmp_path: Path):
    """Test handling of empty sections (zero-length ranges)."""
    pdf_path = tmp_path / "dummy.pdf"
    make_test_pdf(pdf_path, num_pages=5)

    # Introduce an empty range (e.g., start pages 1, 3, 3)
    specs = [
        ("PartA", 0),
        ("PartB", 2),
        ("PartC", 2),  # zero-length section
    ]
    # The function should still produce the right files,
    # but the last one will be empty.
    split_by_specs(pdf_path, specs)

    # Check that all files exist
    assert (tmp_path / "01_PartA.pdf").exists()
    assert (tmp_path / "02_PartB.pdf").exists()
    assert (tmp_path / "03_PartC.pdf").exists()
    
    # Check page counts
    reader_a = PdfReader(str(tmp_path / "01_PartA.pdf"))
    reader_b = PdfReader(str(tmp_path / "02_PartB.pdf"))
    reader_c = PdfReader(str(tmp_path / "03_PartC.pdf"))
    
    # PartA should have pages 0-2 (2 pages)
    assert len(reader_a.pages) == 2
    # PartB should be empty (start=2, end=2)
    assert len(reader_b.pages) == 0
    # PartC should have pages 2-5 (3 pages)
    assert len(reader_c.pages) == 3

def test_parse_specs_valid():
    """Test parsing of valid specifications."""
    args = ["Intro:1", "Methods:9", "Results:23"]
    specs = parse_specs(args)
    expected = [("Intro", 0), ("Methods", 8), ("Results", 22)]
    assert specs == expected

def test_parse_specs_invalid_format():
    """Test parsing of invalid specification format."""
    args = ["InvalidSpec"]
    with pytest.raises(SystemExit):
        parse_specs(args)

def test_parse_specs_invalid_page_number():
    """Test parsing of invalid page numbers."""
    args = ["Intro:0"]  # Page numbers must start at 1
    with pytest.raises(SystemExit):
        parse_specs(args)

def test_parse_specs_non_ascending():
    """Test parsing of non-ascending page numbers."""
    args = ["Intro:10", "Methods:5"]  # 5 comes after 10
    with pytest.raises(SystemExit):
        parse_specs(args)

def test_safe_label_generation(tmp_path: Path):
    """Test that labels are properly sanitized for filenames."""
    pdf_path = tmp_path / "dummy.pdf"
    make_test_pdf(pdf_path, num_pages=5)

    specs = [
        ("Chapter 1: Introduction", 0),
        ("Chapter 2: Methods & Results", 2),
    ]

    split_by_specs(pdf_path, specs)

    # Check that files with sanitized names were created
    assert (tmp_path / "01_Chapter_1_Introduction.pdf").exists()
    assert (tmp_path / "02_Chapter_2_Methods_Results.pdf").exists()

def test_metadata_preservation(tmp_path: Path):
    """Test that metadata is preserved in output files."""
    pdf_path = tmp_path / "dummy.pdf"
    
    # Create a PDF with metadata
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_metadata({
        "/Title": "Test Document",
        "/Author": "Test Author",
        "/Subject": "Test Subject"
    })
    with open(pdf_path, "wb") as f:
        writer.write(f)

    specs = [("TestSection", 0)]
    split_by_specs(pdf_path, specs)

    # Check that metadata was preserved
    output_path = tmp_path / "01_TestSection.pdf"
    reader = PdfReader(str(output_path))
    assert reader.metadata["/Title"] == "Test Document"
    assert reader.metadata["/Author"] == "Test Author"
    assert reader.metadata["/Subject"] == "Test Subject"

if __name__ == "__main__":
    pytest.main([__file__]) 