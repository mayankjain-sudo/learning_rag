"""
Loading and extracting text from PDF files.
"""

from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader


class PDFLoader:
    """Load and extract text from PDF files."""
    
    def __init__(self):
        """Initialize PDF loader."""
        pass
    
    def load_pdf(self, pdf_path: Path) -> List[Dict]:
        """
        Load a PDF file and extract text from all pages.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing page text and basic metadata
        """
        pages_data = []
        
        try:
            reader = PdfReader(str(pdf_path))
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                
                if text.strip():  # Only add non-empty pages
                    pages_data.append({
                        'text': text,
                        'page': page_num,
                        'total_pages': len(reader.pages)
                    })
                    
        except Exception as e:
            print(f"Error loading {pdf_path}: {e}")
        #print("Pages_data:", pages_data)        
        return pages_data
    
    def load_multiple_pdfs(self, pdf_paths: List[Path]) -> Dict[Path, List[Dict]]:
        """
        Load multiple PDF files.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            Dictionary mapping PDF paths to their page data
        """
        results = {}
        
        for pdf_path in pdf_paths:
            print(f"Loading: {pdf_path.name}")
            pages_data = self.load_pdf(pdf_path)
            
            if pages_data:
                results[pdf_path] = pages_data
                print(f"  Loaded {len(pages_data)} pages")
            else:
                print(f"  No text extracted")
                
        return results
    
# Example usage:
##loader = PDFLoader()
##pdf_data = loader.load_pdf(Path("data/handbook.pdf"))
##pdf_data_multiple = loader.load_multiple_pdfs([Path("data/handbook.pdf"), Path("data/sample.pdf")])   