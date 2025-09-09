from typing import List
import re

def chunk_text(text: str, max_chunk_size: int = 4000) -> List[str]:
    """
    Split text into chunks that are suitable for translation while preserving
    sentence boundaries and formatting.
    
    Args:
        text: The text to chunk
        max_chunk_size: Maximum size of each chunk in characters
        
    Returns:
        List of text chunks
    """
    # Split text into sentences (preserving common sentence endings)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0
            
        current_chunk.append(sentence)
        current_size += sentence_size
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def preserve_formatting(text: str, formatting_info: dict) -> str:
    """
    Apply formatting information back to translated text.
    
    Args:
        text: The translated text
        formatting_info: Dictionary containing formatting information
        
    Returns:
        Formatted text
    """
    # Implementation will depend on the specific formatting requirements
    # This is a placeholder for future implementation
    return text 