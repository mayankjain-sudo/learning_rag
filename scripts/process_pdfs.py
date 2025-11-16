"""
PDF Processing and Vector Database Storage

Orchestrates the PDF processing pipeline using modular components.
Supports storing custom metadata directly in the vector database.
"""

import sys
from pathlib import Path
from typing import Optional, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.pdf_loader import PDFLoader
from src.core.metadata_extractor import MetadataExtractor
from src.core.vector_db import VectorDatabase
from src.core.config import load_config, get_embedding_function

class PDFProcessor:
    """Process PDFs and store in vector database with metadata."""
    
    def __init__(
        self,
        data_dir: str = "data",
        db_dir: str = "chroma_db",
        metadata_file: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Dict]] = None
        ):
        """
        Initialize PDF processor with configuration and custom metadata.
        
        Args:
            rgs:
            data_dir: Directory containing PDF files
            db_dir: Directory for vector database storage
            metadata_file: Optional JSON file with custom metadata (legacy support)
            custom_metadata: Optional dict with custom metadata per PDF
        """
        
        self.data_dir = Path(data_dir)
        self.db_dir = Path(db_dir)
        
        #Load the configuration
        self.config = load_config()
        print(f"Using provider: {self.config.provider}")
        
        #Get the embedding function based on config
        embeddings = get_embedding_function(self.config)
        
        #Initialize components
        self.pdf_loader = PDFLoader()
        
        if custom_metadata:
            self.metadata_extractor = MetadataExtractor()
            self.metadata_extractor.custom_metadata = custom_metadata
            print("Using provided custom metadata dictionary.")
        else:
            print("Use JSON file id provided for custom metadata.")
            self.metadata_extractor = MetadataExtractor(metadata_file)
            
        self.vector_db = VectorDatabase(
            db_dir=str(self.db_dir),
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            embeddings=embeddings
        )
        
    def process_pdfs(self, force_reprocess: bool = False) -> None:
        """
        Process PDFs in data directory and store in vector database.
        
        This method:
        1. Finds all PDF files
        2. Identifies new/unprocessed PDFs
        3. Loads and extracts text (only from new PDFs)
        4. Extracts and merges metadata
        5. Creates text chunks
        6. Stores in vector database
        
        Args:
            force_reprocess: If True, reprocess all PDFs (default: False)
        """
        
       # Create data directory if it doesn't exist
        self.data_dir.mkdir(exist_ok=True)
        
        #Find all PDF files in data directory
        pdf_files = list(self.data_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.data_dir}")
            return
        print(f"Found {len(pdf_files)} PDF files in {self.data_dir}")
        
        #Load existing vector database to check already processed files (unless force_reprocess)
        
        if not force_reprocess:
            processed_pdfs = self.vector_db.get_processed_pdfs()
            
            if processed_pdfs:
                print(f"Already processed {len(processed_pdfs)} PDFs.")
                print(f"  Files: {', '.join(sorted(processed_pdfs.keys()))}\n")
                
                new_pdf_files = []
                updated_pdf_files = []
                
                for pdf in pdf_files:
                    if pdf.name not in processed_pdfs:
                        new_pdf_files.append(pdf)
                    else:
                        current_checksum = self.vector_db.calculate_pdf_checksum(pdf)
                        stored_checksum = processed_pdfs[pdf.name]
                        if stored_checksum is None:
                            print(f"  No checksum stored for {pdf.name}, reprocessing.")
                            updated_pdf_files.append(pdf)
                        elif current_checksum != stored_checksum:
                            print(f"  Checksum changed for {pdf.name}, reprocessing.")
                            updated_pdf_files.append(pdf)
                        else:
                            print(f"  Checksum unchanged for {pdf.name}, skipping.")
                
                pdfs_to_process = new_pdf_files + updated_pdf_files
                
                if not pdfs_to_process:
                    print("✓ No new or updated PDFs to process. All PDFs are up to date!")
                    print("\nTo reprocess all PDFs anyway, use: python main.py process --force")
                    return
                
                if new_pdf_files:
                    print(f"New PDFs: {len(new_pdf_files)}")
                    for pdf in new_pdf_files:
                        print(f"  + {pdf.name}")
                
                if updated_pdf_files:
                    print(f"\nUpdated PDFs: {len(updated_pdf_files)}")
                    for pdf in updated_pdf_files:
                        print(f"  ↻ {pdf.name}")
                
                print()
                pdf_files = pdfs_to_process
            else:
                print("No previous processing found. Processing all PDFs...\n")
        else:
            print("Force reprocessing enabled. Processing all PDFs...\n")
        
        # Load PDFs
        pdf_data = self.pdf_loader.load_multiple_pdfs(pdf_files)
        
        if not pdf_data:
            print("No text extracted from PDFs")
            return
        
        # Process each PDF and collect chunks
        all_texts = []
        all_metadatas = []
        
        print("\nProcessing chunks and metadata...")
        for pdf_path, pages_data in pdf_data.items():
            # Merge metadata for this PDF
            pdf_metadata = self.metadata_extractor.merge_metadata(
                pdf_path,
                {}  # base metadata
            )
            
            # Create chunks with metadata
            texts, metadatas = self.vector_db.create_chunks(
                pages_data,
                pdf_metadata
            )
            
            print(f"  {pdf_path.name}: {len(texts)} chunks")
            
            all_texts.extend(texts)
            all_metadatas.extend(metadatas)
        
        # Store in vector database
        if all_texts:
            vectordb = self.vector_db.store_chunks(all_texts, all_metadatas)
            
            # Update tracking of processed PDFs with checksums
            processed_info = {
                pdf_path.name: self.vector_db.calculate_pdf_checksum(pdf_path)
                for pdf_path in pdf_data.keys()
            }
            self.vector_db.update_processed_pdfs(processed_info)
            
            # Run test query
            ##self.vector_db.test_query(vectordb, all_texts)
        else:
            print("No chunks created")


def main():
    
    print("=== PDF Processor - Incremental Loading ===\n")
    
    # Check for flags and metadata file
    metadata_file = None
    force_reprocess = False
    
    for arg in sys.argv[1:]:
        if arg in ['--force', '-f']:
            force_reprocess = True
            print("Force reprocessing enabled\n")
        elif not arg.startswith('-'):
            metadata_file = arg
            print(f"Using metadata file: {metadata_file}\n")
    
    # Create and run processor
    processor = PDFProcessor(
        data_dir="data",
        db_dir="chroma_db",
        metadata_file=metadata_file
    )
    
    processor.process_pdfs(force_reprocess=force_reprocess)
    
    print("\n=== Processing Complete ===")


if __name__ == "__main__":
    main()
                