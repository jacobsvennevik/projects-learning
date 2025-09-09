#!/usr/bin/env python3
"""
Script to split the specific PDF file with the correct chapter structure.
"""

import glob
from pathlib import Path
from split_pdf_sections import split_by_specs, parse_specs

def main():
    # Find the PDF file
    pdf_files = glob.glob("pdf/*.pdf")
    if not pdf_files:
        print("No PDF files found in pdf/ directory")
        return
    
    pdf_path = Path(pdf_files[0])
    print(f"Found PDF: {pdf_path}")
    
    # Define the chapter structure based on our analysis
    specs = [
        "Front_Matter:1",
        "Chapter_1_Designing_for_the_Learner:20", 
        "Chapter_2_Integrated_Framework:34",
        "Chapter_3_Course_Structure:58", 
        "Chapter_4_Instructional_Content:80",
        "Chapter_5_Learning_Activities:104",
        "Chapter_6_Social_Interactions:122", 
        "Chapter_7_Assessments_Feedback:140",
        "Chapter_8_Putting_It_Together:156",
        "Back_Matter:165"
    ]
    
    print(f"\nSplitting PDF into {len(specs)} sections...")
    print("Specifications:")
    for spec in specs:
        print(f"  {spec}")
    
    # Parse and split
    parsed_specs = parse_specs(specs)
    split_by_specs(pdf_path, parsed_specs)
    
    print(f"\n✅ PDF splitting completed! Check the current directory for the output files.")

if __name__ == "__main__":
    main() 