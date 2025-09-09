#!/usr/bin/env python3
"""
Example usage of split_pdf_sections.py

This script demonstrates how to use the PDF splitting functionality
programmatically rather than from the command line.
"""

from pathlib import Path
from split_pdf_sections import split_by_specs, parse_specs

def example_basic_usage():
    """Example of basic usage with a sample PDF."""
    print("=== Basic Usage Example ===")
    
    # Example specifications for a textbook
    specs_text = [
        "Foreword:10",
        "Preface:13", 
        "Ch01_DesigningForTheLearner:20",
        "Ch02_IntegratedFramework:34",
        "Ch03_CourseStructure:58"
    ]
    
    print(f"Specifications: {specs_text}")
    
    # Parse the specifications
    specs = parse_specs(specs_text)
    print(f"Parsed specs: {specs}")
    
    # Note: This would require an actual PDF file
    # pdf_path = Path("textbook.pdf")
    # split_by_specs(pdf_path, specs)
    print("(PDF splitting would happen here if a file was provided)")

def example_programmatic_usage():
    """Example of creating specifications programmatically."""
    print("\n=== Programmatic Usage Example ===")
    
    # Create specifications from a list of chapters
    chapters = [
        ("Introduction", 1),
        ("Literature Review", 15),
        ("Methodology", 45),
        ("Results", 80),
        ("Discussion", 120),
        ("Conclusion", 150)
    ]
    
    # Convert to the expected format
    specs_text = [f"{title}:{page}" for title, page in chapters]
    print(f"Generated specs: {specs_text}")
    
    # Parse and use
    specs = parse_specs(specs_text)
    print(f"Parsed specs: {specs}")

def example_error_handling():
    """Example of how the script handles errors."""
    print("\n=== Error Handling Examples ===")
    
    # These would raise SystemExit with error messages
    invalid_examples = [
        ["InvalidSpec"],           # Bad format
        ["Intro:0"],              # Page number < 1
        ["Intro:10", "Methods:5"] # Non-ascending
    ]
    
    for i, example in enumerate(invalid_examples, 1):
        print(f"Example {i}: {example}")
        try:
            parse_specs(example)
        except SystemExit as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    example_basic_usage()
    example_programmatic_usage()
    example_error_handling()
    
    print("\n=== Usage Instructions ===")
    print("To use this script with a real PDF:")
    print("1. Place your PDF in the same directory")
    print("2. Update the pdf_path in example_basic_usage()")
    print("3. Run: python example_usage.py") 