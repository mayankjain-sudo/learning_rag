"""
Metadata Extractor

It will handle the extraction and management of PDF metadata and custom metadata.
"""

import json
from pathlib import Path
from typing import Dict, Optional, List
from pypdf import PdfReader


class MetadataExtractor:
    """Extract and manage PDF and custom metadata."""
    
    # Supported custom metadata fields
    CUSTOM_FIELDS = {
        'category': str,        # Document category (e.g., 'research', 'documentation', 'reference')
        'keywords': list,       # List of keywords/tags
        'department': str,      # Department or team
        'classification': str,  # Security/access classification
        'language': str,        # Document language code (e.g., 'en', 'es')
        'version': str,         # Document version
        'project': str,         # Associated project name
        'year': str,            # Publication/creation year
        'type': str,            # Document type (e.g., 'paper', 'manual', 'report')
        'status': str,          # Document status (e.g., 'draft', 'final', 'archived')
        'tags': list,           # Additional tags
        'priority': str,        # Priority level (e.g., 'high', 'medium', 'low')
        'confidential': bool,   # Confidentiality flag
    }
    
    def __init__(self, metadata_file: Optional[str] = None):
        """
        Initialize metadata extractor.
        
        Args:
            metadata_file: Optional path to JSON file with custom metadata
        """
        self.metadata_file = Path(metadata_file) if metadata_file else None
        self.custom_metadata = self._load_custom_metadata()
    
    def _load_custom_metadata(self) -> Dict:
        """
        Load custom metadata from JSON file.
        
        Returns:
            Dictionary with custom metadata for each PDF file
        """
        if not self.metadata_file or not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                print(f"Loaded custom metadata from {self.metadata_file}")
                # Validate metadata
                self._validate_metadata(metadata)
                return metadata
        except Exception as e:
            print(f"Warning: Could not load metadata file: {e}")
            return {}
    
    def _validate_metadata(self, metadata: Dict) -> None:
        """
        Validate custom metadata structure and types.
        
        Args:
            metadata: Metadata dictionary to validate
        """
        for filename, fields in metadata.items():
            if not isinstance(fields, dict):
                print(f"Warning: Metadata for '{filename}' should be a dictionary")
                continue
            
            for field_name, field_value in fields.items():
                # Check if field is known
                if field_name in self.CUSTOM_FIELDS:
                    expected_type = self.CUSTOM_FIELDS[field_name]
                    if not isinstance(field_value, expected_type):
                        print(f"Warning: Field '{field_name}' in '{filename}' should be {expected_type.__name__}, got {type(field_value).__name__}")
                # Allow unknown fields (for flexibility)
    
    def get_supported_fields(self) -> Dict[str, str]:
        """
        Get list of supported custom metadata fields.
        
        Returns:
            Dictionary mapping field names to their types
        """
        return {
            field: type_.__name__ 
            for field, type_ in self.CUSTOM_FIELDS.items()
        }
    
    def extract_pdf_metadata(self, pdf_path: Path) -> Dict:
        """
        Extract metadata from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with PDF metadata
        """
        pdf_metadata = {}
        
        try:
            reader = PdfReader(str(pdf_path))
            
            if reader.metadata:
                pdf_metadata = {
                    'title': reader.metadata.get('/Title', ''),
                    'author': reader.metadata.get('/Author', ''),
                    'subject': reader.metadata.get('/Subject', ''),
                    'creator': reader.metadata.get('/Creator', ''),
                }
        except Exception as e:
            print(f"Warning: Could not extract metadata from {pdf_path}: {e}")
            
        return pdf_metadata
    
    def get_custom_metadata(self, pdf_filename: str) -> Dict:
        """
        Get custom metadata for a specific PDF file.
        
        Args:
            pdf_filename: Name of the PDF file
            
        Returns:
            Dictionary with custom metadata for this file
        """
        return self.custom_metadata.get(pdf_filename, {})
    
    def merge_metadata(self, pdf_path: Path, base_metadata: Dict) -> Dict:
        """
        Merge all metadata sources for a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            base_metadata: Base metadata dictionary to merge into
            
        Returns:
            Dictionary with merged metadata
        """
        merged = base_metadata.copy()
        
        # Add source information
        merged['source'] = str(pdf_path)
        merged['filename'] = pdf_path.name
        
        # Extract and merge PDF metadata
        pdf_metadata = self.extract_pdf_metadata(pdf_path)
        merged.update(pdf_metadata)
        
        # Merge custom metadata
        custom_metadata = self.get_custom_metadata(pdf_path.name)
        merged.update(custom_metadata)
        
        return merged