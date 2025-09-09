# Document Translator

A Python tool for translating PowerPoint (.pptx), Word (.docx), and PDF documents from Portuguese to English using OpenAI's GPT-4.

## Features

- Supports multiple document formats (.pptx, .docx, .pdf)
- Preserves original document formatting
- Uses OpenAI's GPT-4 for high-quality translations
- Handles text chunking and rate limits automatically
- Progress tracking for large documents

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

```python
from document_translator import DocumentTranslator

# Initialize the translator
translator = DocumentTranslator()

# Translate a PowerPoint file
translator.translate_file("input.pptx", "output.pptx", source_lang="pt", target_lang="en")

# Translate a Word document
translator.translate_file("input.docx", "output.docx", source_lang="pt", target_lang="en")

# Translate a PDF file
translator.translate_file("input.pdf", "output.pdf", source_lang="pt", target_lang="en")
```

## Project Structure

- `document_translator/`: Main package directory
  - `translator.py`: Core translation logic
  - `parsers/`: Document format-specific parsers
    - `pptx_parser.py`: PowerPoint file parser
    - `docx_parser.py`: Word document parser
    - `pdf_parser.py`: PDF file parser
  - `utils.py`: Utility functions
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (create this file) 