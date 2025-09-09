import os
from typing import Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

from .parsers.pptx_parser import PPTXParser
from .parsers.docx_parser import DOCXParser
from .parsers.pdf_parser import PDFParser
from .utils import chunk_text

class DocumentTranslator:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.parsers = {
            ".pptx": PPTXParser(),
            ".docx": DOCXParser(),
            ".pdf": PDFParser()
        }
        
    def translate_file(self, input_path: str, output_path: str, 
                      source_lang: str = "pt", target_lang: str = "en") -> None:
        """
        Translate a document from one language to another.
        
        Args:
            input_path: Path to the input document
            output_path: Path where the translated document should be saved
            source_lang: Source language code (default: "pt" for Portuguese)
            target_lang: Target language code (default: "en" for English)
        """
        # Get file extension and appropriate parser
        ext = os.path.splitext(input_path)[1].lower()
        if ext not in self.parsers:
            raise ValueError(f"Unsupported file format: {ext}")
            
        parser = self.parsers[ext]
        
        # Extract text and formatting information
        print(f"Extracting text from {input_path}...")
        doc_content = parser.extract_content(input_path)
        
        # Translate text chunks
        print("Translating content...")
        translated_content = self._translate_content(
            doc_content["text"], source_lang, target_lang
        )
        
        # Create new document with translated content
        print(f"Saving translated document to {output_path}...")
        parser.create_document(output_path, translated_content, doc_content["formatting"])
        
    def _translate_content(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text using OpenAI's API with chunking for large texts.
        """
        chunks = chunk_text(text)
        translated_chunks = []
        
        for chunk in tqdm(chunks, desc="Translating chunks"):
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are a professional translator. Translate from {source_lang} to {target_lang}. Preserve any special characters, numbers, and formatting."},
                    {"role": "user", "content": chunk}
                ],
                temperature=0.3
            )
            translated_chunks.append(response.choices[0].message.content)
            
        return " ".join(translated_chunks) 