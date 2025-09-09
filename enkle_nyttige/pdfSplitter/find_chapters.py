#!/usr/bin/env python3
"""
Script to find all chapter start pages in the Make It Stick PDF.
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
    
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"Total pages: {total_pages}")
    
    # Chapter titles to search for
    chapters = [
        "Learning Is Misunderstood",
        "To Learn, Retrieve", 
        "Mix Up Your Practice",
        "Embrace Difficulties",
        "Avoid Illusions of Knowing",
        "Get Beyond Learning Styles",
        "Increase Your Abilities",
        "Make It Stick"
    ]
    
    print("\nSearching for chapter start pages...")
    print("=" * 50)
    
    chapter_pages = {}
    
    for i in range(total_pages):
        text = reader.pages[i].extract_text()
        if text.strip():
            for chapter in chapters:
                if chapter in text and chapter not in chapter_pages:
                    # Look for the chapter title as a main heading
                    lines = text.split('\n')
                    for line in lines:
                        if chapter in line and len(line.strip()) < 100:  # Likely a heading
                            # Skip if this looks like a table of contents
                            if not any(toc_indicator in text.lower() for toc_indicator in ['contents', 'table of contents']):
                                chapter_pages[chapter] = i + 1
                                print(f"Chapter: {chapter} - Page {i+1}")
                                break
    
    print("\n" + "=" * 50)
    print("Chapter start pages found:")
    for chapter, page in sorted(chapter_pages.items()):
        print(f"  {chapter}: {page}")
    
    # Also check for Preface
    for i in range(20):
        text = reader.pages[i].extract_text()
        if "Preface" in text and "ix" in text:
            print(f"  Preface: {i+1}")
            break

if __name__ == "__main__":
    main()
