#!/usr/bin/env python3
"""
Simple script to check the page structure of the Make It Stick PDF.
"""

import glob
from pypdf import PdfReader

def main():
    # Find the Make It Stick PDF using glob
    pdf_files = glob.glob("pdf/*Make It Stick*.pdf")
    if not pdf_files:
        print("No Make It Stick PDF found")
        return
    
    pdf_path = pdf_files[0]
    print(f"Found PDF: {pdf_path}")
    
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        print(f"Total pages: {total_pages}")
        print("\nFirst 15 pages content:")
        print("=" * 50)
        
        for i in range(min(15, total_pages)):
            text = reader.pages[i].extract_text()
            if text.strip():
                lines = text.split('\n')
                # Get first few meaningful lines
                meaningful_lines = [line.strip() for line in lines if line.strip()][:3]
                print(f"Page {i+1}:")
                for line in meaningful_lines:
                    print(f"  {line}")
                print()
        
        print("=" * 50)
        print("Looking for chapter markers...")
        
        # Look for chapter headings
        for i in range(total_pages):
            text = reader.pages[i].extract_text()
            if text.strip():
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    if any(keyword in line.upper() for keyword in ['CHAPTER', 'LEARNING IS MISUNDERSTOOD', 'TO LEARN, RETRIEVE', 'MIX UP YOUR PRACTICE']):
                        print(f"Page {i+1}: {line}")
                        break
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
