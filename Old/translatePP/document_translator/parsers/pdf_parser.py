from typing import Dict, Any, List
import fitz  # PyMuPDF
import json

class PDFParser:
    def __init__(self):
        # Default fonts to use when original fonts are not available
        self.default_fonts = {
            "serif": "Times-Roman",
            "sans-serif": "Helvetica",
            "monospace": "Courier"
        }
    
    def extract_content(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text and formatting information from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing text and formatting information
        """
        doc = fitz.open(file_path)
        text_content = []
        formatting_info = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_content = {
                "page_number": page_num + 1,
                "blocks": []
            }
            
            # Extract text blocks with their formatting
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_content = {
                                "text": span["text"],
                                "formatting": {
                                    "font_name": span["font"],
                                    "font_size": span["size"],
                                    "is_bold": "bold" in span["font"].lower(),
                                    "is_italic": "italic" in span["font"].lower(),
                                    "bbox": span["bbox"],  # Bounding box coordinates
                                    "font_type": self._determine_font_type(span["font"])
                                }
                            }
                            text_content.append(span["text"])
                            page_content["blocks"].append(block_content)
            
            formatting_info.append(page_content)
        
        doc.close()
        
        return {
            "text": "\n".join(text_content),
            "formatting": formatting_info
        }
    
    def _determine_font_type(self, font_name: str) -> str:
        """Determine the font type (serif, sans-serif, or monospace) from the font name."""
        font_name = font_name.lower()
        if any(name in font_name for name in ["times", "serif"]):
            return "serif"
        elif any(name in font_name for name in ["courier", "mono"]):
            return "monospace"
        else:
            return "sans-serif"
    
    def create_document(self, file_path: str, translated_text: str, 
                       formatting_info: Dict[str, Any]) -> None:
        """
        Create a new PDF file with translated content and preserved formatting.
        
        Args:
            file_path: Path where the new PDF file should be saved
            translated_text: The translated text content
            formatting_info: Dictionary containing formatting information
        """
        # Create a new PDF document
        doc = fitz.open()
        
        # Split translated text into chunks (one per block)
        translated_chunks = translated_text.split("\n")
        chunk_index = 0
        
        for page_info in formatting_info:
            # Create a new page with the same dimensions as the original
            page = doc.new_page()
            
            for block_info in page_info["blocks"]:
                if chunk_index < len(translated_chunks):
                    # Get the original formatting
                    formatting = block_info["formatting"]
                    bbox = formatting["bbox"]
                    
                    # Use default font based on font type
                    font_type = formatting.get("font_type", "sans-serif")
                    font_name = self.default_fonts[font_type]
                    
                    # Add bold/italic variants if needed
                    if formatting["is_bold"] and formatting["is_italic"]:
                        font_name += "-BoldOblique"
                    elif formatting["is_bold"]:
                        font_name += "-Bold"
                    elif formatting["is_italic"]:
                        font_name += "-Oblique"
                    
                    try:
                        # Insert the translated text with preserved formatting
                        page.insert_text(
                            point=(bbox[0], bbox[1]),  # Top-left corner
                            text=translated_chunks[chunk_index],
                            fontname=font_name,
                            fontsize=formatting["font_size"],
                            color=(0, 0, 0)  # Black text
                        )
                    except Exception as e:
                        # Fallback to basic font if the specified font fails
                        print(f"Warning: Using fallback font for chunk {chunk_index}: {str(e)}")
                        page.insert_text(
                            point=(bbox[0], bbox[1]),
                            text=translated_chunks[chunk_index],
                            fontname="Helvetica",
                            fontsize=formatting["font_size"],
                            color=(0, 0, 0)
                        )
                    
                    chunk_index += 1
        
        # Save the document
        doc.save(file_path)
        doc.close() 