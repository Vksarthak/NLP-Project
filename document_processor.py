import io
import re
from typing import List, Dict, Any
import pdfplumber

class DocumentProcessor:
    """Processor for extracting and cleaning text from documents."""

    def __init__(self):
        """Initialize the DocumentProcessor."""
        pass

    def _clean_text(self, text: str) -> str:
        """
        Normalize whitespace and remove excessive newlines.

        Args:
            text: The raw text.

        Returns:
            The cleaned text.
        """
        # Replace multiple spaces/tabs with a single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Replace 3 or more newlines with 2 newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a file (PDF or TXT).

        Args:
            file_path: Path to the file.

        Returns:
            Extracted text.
        """
        if file_path.lower().endswith('.pdf'):
            try:
                text_parts = []
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text_parts.append(extracted)
                return self._clean_text("\n\n".join(text_parts))
            except Exception as e:
                raise ValueError(f"Error extracting PDF: {e}")
        elif file_path.lower().endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return self._clean_text(f.read())
            except UnicodeDecodeError:
                # Fallback encoding for TXT
                with open(file_path, 'r', encoding='latin-1') as f:
                    return self._clean_text(f.read())
            except Exception as e:
                raise ValueError(f"Error extracting TXT: {e}")
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def extract_text_from_bytes(self, file_bytes: bytes, file_type: str) -> str:
        """
        Extract text from file bytes (e.g., from Streamlit uploads).

        Args:
            file_bytes: The file content as bytes.
            file_type: The MIME type or extension of the file.

        Returns:
            Extracted text.
        """
        if 'pdf' in file_type.lower():
            try:
                text_parts = []
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text_parts.append(extracted)
                return self._clean_text("\n\n".join(text_parts))
            except Exception as e:
                raise ValueError(f"Error extracting PDF from bytes: {e}")
        elif 'text' in file_type.lower() or 'txt' in file_type.lower():
            try:
                return self._clean_text(file_bytes.decode('utf-8'))
            except UnicodeDecodeError:
                return self._clean_text(file_bytes.decode('latin-1'))
            except Exception as e:
                raise ValueError(f"Error extracting TXT from bytes: {e}")
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def chunk_text(self, text: str, chunk_size: int = 3000, overlap: int = 200) -> List[str]:
        """
        Chunk text using character-based splitting at sentence boundaries.

        Args:
            text: The text to chunk.
            chunk_size: Maximum size of a chunk in characters.
            overlap: Minimum overlap in characters between chunks.

        Returns:
            List of text chunks.
        """
        # Roughly split text into sentences based on punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size, finalize the current chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Establish overlap using the end of the previous chunk
                overlap_length = 0
                overlap_chunk = []
                for s in reversed(current_chunk):
                    if overlap_length + len(s) > overlap and overlap_length > 0:
                        break
                    overlap_chunk.insert(0, s)
                    overlap_length += len(s) + 1 # +1 for space
                
                current_chunk = overlap_chunk
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1 # +1 for space
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def get_metadata(self, text: str) -> Dict[str, Any]:
        """
        Get metadata about the text.

        Args:
            text: The text to analyze.

        Returns:
            Dictionary containing word_count, char_count, sentence_count, paragraph_count.
        """
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        
        # Approximate sentence count based on common end punctuation
        sentence_count = len(re.split(r'[.!?]+', text)) - 1
        if sentence_count < 1 and char_count > 0:
            sentence_count = 1
            
        # Approximate paragraph count based on double newlines
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        
        return {
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count
        }
