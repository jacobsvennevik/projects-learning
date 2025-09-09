#!/usr/bin/env python3
"""
Script to split the "Make It Stick" PDF into its chapters.
Based on the table of contents:
- Preface (page ix)
- Chapter 1: Learning Is Misunderstood (page 1)
- Chapter 2: To Learn, Retrieve (page 23)
- Chapter 3: Mix Up Your Practice (page 46)
- Chapter 4: Embrace Difficulties (page 67)
- Chapter 5: Avoid Illusions of Knowing (page 102)
- Chapter 6: Get Beyond Learning Styles (page 131)
- Chapter 7: Increase Your Abilities (page 162)
- Chapter 8: Make It Stick (page 200)
"""

import glob
from pathlib import Path
import re
from split_pdf_sections import split_by_specs, parse_specs

def main():
    # Find the "Make It Stick" PDF file
    pdf_files = glob.glob("pdf/*Make It Stick*.pdf")
    if not pdf_files:
        print("No 'Make It Stick' PDF file found in pdf/ directory")
        print("Looking for files containing 'Make It Stick' in the name...")
        return
    
    pdf_path = Path(pdf_files[0])
    print(f"Found PDF: {pdf_path}")
    
    # Create output directory for split PDFs (inside pdf/Make It Stick)
    output_dir = pdf_path.parent / "Make It Stick"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating output directory: {output_dir}")
    
    # Define the chapter structure for "Make It Stick"
    # Based on actual PDF analysis
    specs = [
        "Preface:10",  # Preface starts at page 10
        "Chapter_1_Learning_Is_Misunderstood:16",
        "Chapter_2_To_Learn_Retrieve:38",
        "Chapter_3_Mix_Up_Your_Practice:61",
        "Chapter_4_Embrace_Difficulties:84",
        "Chapter_5_Avoid_Illusions_of_Knowing:117",
        "Chapter_6_Get_Beyond_Learning_Styles:146",
        "Chapter_7_Increase_Your_Abilities:177",
        "Chapter_8_Make_It_Stick:210"  # Estimated end of book
    ]
    
    print(f"\nSplitting 'Make It Stick' PDF into {len(specs)} sections...")
    print("Specifications:")
    for spec in specs:
        print(f"  {spec}")
    
    # Parse and split directly into the output directory
    parsed_specs = parse_specs(specs)
    # Skip empty ranges to avoid zero-page PDFs
    split_by_specs(pdf_path, parsed_specs, output_dir, empty_policy="skip")
    
    print(f"\nWrote split PDFs to {output_dir}/")
    
    print(f"\n✅ PDF splitting completed! All split PDFs are in: {output_dir}/")
    print("Note: The Preface section will be empty if page 'ix' is not found.")

if __name__ == "__main__":
    main()
