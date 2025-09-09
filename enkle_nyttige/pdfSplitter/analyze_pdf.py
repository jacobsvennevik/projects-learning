#!/usr/bin/env python3
"""
Script to analyze PDF structure and help determine splitting points.
"""

import glob
from pypdf import PdfReader

def analyze_pdf(pdf_path):
    """Analyze the PDF structure to help determine splitting points."""
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    print(f"PDF: {pdf_path}")
    print(f"Total pages: {total_pages}")
    print("\n" + "="*50)
    
    # Look for chapter headings or major sections
    print("Analyzing first 20 pages for structure...")
    for i in range(min(20, total_pages)):
        text = reader.pages[i].extract_text()
        if text.strip():
            # Look for chapter numbers or major headings
            lines = text.split('\n')
            for line in lines[:5]:  # Check first 5 lines
                line = line.strip()
                if any(keyword in line.upper() for keyword in ['CHAPTER', 'PART', 'SECTION', 'INTRODUCTION', 'CONCLUSION']):
                    print(f"Page {i+1}: {line[:100]}...")
                    break
    
    print("\n" + "="*50)
    print("Suggested splitting strategy:")
    print("Based on typical academic book structure, here are some common splitting points:")
    
    # Common academic book structure
    suggestions = [
        ("Front Matter", 1),
        ("Table of Contents", 5),
        ("Chapter 1", 15),
        ("Chapter 2", 35),
        ("Chapter 3", 55),
        ("Chapter 4", 75),
        ("Chapter 5", 95),
        ("Chapter 6", 115),
        ("Chapter 7", 135),
        ("Chapter 8", 155),
        ("Chapter 9", 175),
        ("Chapter 10", 195),
        ("Back Matter", 205)
    ]
    
    for title, page in suggestions:
        if page <= total_pages:
            print(f'  "{title}:{page}"')
    
    return total_pages

if __name__ == "__main__":
    pdf_files = glob.glob("pdf/*.pdf")
    if pdf_files:
        analyze_pdf(pdf_files[0])
    else:
        print("No PDF files found in pdf/ directory") 