"""
Vector Database Module

It handles the creation, storage, and querying of vector database using ChromaDB.
Supports both Ollama and Azure OpenAI embeddings.
"""

import pickle
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import time

class VectorDatabase:
    """
    Manages ChromaDB for storing and retrieving document embeddings.
    """

    def __init__(
        self,
        db_dir: str = "chroma_db",
        embedding_model: str = "nomic-embed-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embeddings = None,
    ):
        """
        Initializes the VectorDatabase with a configuration and embedding function.

        Args:
            db_dir: Directory to store the ChromaDB database.
            embedding_model: The embedding model to use.
            chunk_size: Size of text chunks for splitting.
            chunk_overlap: Overlap size between text chunks.
            embedding: Optional embedding function.
        """
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(exist_ok=True)
        
        #Initialize Text
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        #Use provided embedding or default to Ollama
        
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            from langchain_ollama import OllamaEmbeddings
            self.embeddings = OllamaEmbeddings(model=embedding_model)
            
    def create_chunks(
        self,
        pages_data: List[Dict],
        pdf_metadata: Dict
    ) -> Tuple[List[str], List[Dict]]:
        """
        Splits pages data into text chunks and prepares metadata.

        Args:
            pages_data: List of page data dictionaries.
            pdf_metadata: Metadata for the PDF document.
        Returns:
            Tuple of list of text chunks and their corresponding metadata.
        """
        all_texts = []
        all_metadatas = []
        for page_data in pages_data:
            # extract text from page data
            text = page_data['text']
            # split text into chunks
            chunks = self.text_splitter.split_text(text)
            # prepare metadata for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                all_texts.append(chunk)
                
                # merge page metadata with pdf metadata
                chunk_metadata = pdf_metadata.copy()
                chunk_metadata['page'] = page_data['page']
                chunk_metadata['total_pages'] = page_data['total_pages']
                chunk_metadata['chunk_index'] = chunk_idx
                all_metadatas.append(chunk_metadata)
                
        return all_texts, all_metadatas
    
    def store_chunks(
        self,
        texts: List[str],
        metadatas: List[Dict],
        batch_size: int = 10
    ) -> Chroma:
        """
        Stores text chunks and their metadata into ChromaDB in batches.

        Args:
            texts: List of text chunks.
            metadatas: List of metadata dictionaries corresponding to the text chunks.
            batch_size: Number of chunks to process in each batch.
        Returns:
            Chroma database vector store instance
        """
        print(f"Storing {len(texts)} chunks into ChromaDB...with batch size {batch_size}")
        
        
        #Initialize ChromaDB or load existing
        if (self.db_dir / "chroma-sqlite").exists():
            print("Loading existing ChromaDB...")
            vectordb = Chroma(
                persist_directory=str(self.db_dir),
                embedding_function=self.embeddings
            )
        else:
            print("Creating new ChromaDB...")
            vectordb = None
            
            # Process in batches
        total_batches = (len(texts) - 1) // batch_size + 1
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            batch_num = i // batch_size + 1
            print(f"  Processing batch {batch_num}/{total_batches} ({len(batch_texts)} chunks)...", end=" ")
            
            # Retry logic for transient errors
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if vectordb is None:
                        # Create database with first batch
                        vectordb = Chroma.from_texts(
                            texts=batch_texts,
                            metadatas=batch_metadatas,
                            embedding=self.embeddings,
                            persist_directory=str(self.db_dir)
                        )
                    else:
                        # Add to existing database
                        vectordb.add_texts(
                            texts=batch_texts,
                            metadatas=batch_metadatas
                        )
                    print("✓")
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Retry {attempt + 1}/{max_retries}...")
                        time.sleep(2)  # Wait before retry
                    else:
                        print(f"Failed after {max_retries} attempts")
                        raise
            
            # Small delay between batches to avoid overwhelming Ollama
            if i + batch_size < len(texts):
                time.sleep(0.5)
        
        print(f"\nSuccessfully stored all chunks in {self.db_dir}")
        print(f"Total chunks: {len(texts)}")
        
        # Save chunk data for inspection
        self._save_chunk_data(texts, metadatas)
        
        if vectordb is None:
            raise RuntimeError("Failed to create vector database")
        
        return vectordb
    
    
    # save the chunk data
    def _save_chunk_data(self, texts: List[str], metadatas: List[Dict]) -> None:
        """
        Saves chunk texts and metadata to JSON and pickle files for inspection.

        Args:
            texts: List of text chunks.
            metadatas: List of metadata dictionaries.
        """
        chunk_data = {
            'texts': texts, 
            'metadata': metadatas
            }
        
        # Save as JSON
        json_path = self.db_dir / "chunk_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=4)
        print(f"Saved chunk data to {json_path}")
        
        # Save as pickle
        pickle_path = self.db_dir / "chunk_data.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(chunk_data, f)
        print(f"Saved chunk data to {pickle_path}")
        
    # Calculate the checksum fof each pdf file
    def calculate_pdf_checksum(self, pdf_path: Path) -> str:
        """
        Calculates the SHA256 checksum of a PDF file.

        Args:
            pdf_path: Path to the PDF file.
        Returns:
            Hexadecimal string of the checksum.
        """
        sha256_hash = hashlib.sha256()
        
        with open(pdf_path, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def get_processed_pdfs(self) -> Dict[str, str]:
        """
        Get list of PDFs that have already been processed with their checksums.
        
        Returns:
            Dict mapping PDF filenames to their checksums
        """
        tracking_file = self.db_dir / 'processed_pdfs.json'
        
        if tracking_file.exists():
            try:
                with open(tracking_file, 'r') as f:
                    data = json.load(f)
                    # Handle old format (list) and new format (dict)
                    pdfs = data.get('processed_pdfs', [])
                    if isinstance(pdfs, list):
                        # Old format: convert to dict without checksums
                        return {pdf: "" for pdf in pdfs}
                    else:
                        # New format: dict with checksums
                        return pdfs
            except Exception as e:
                print(f"Warning: Could not load processed PDFs tracking: {e}")
                return {}
        return {}
    
    def update_processed_pdfs(self, pdf_info: Dict[str, str]) -> None:
        """
        Update the list of processed PDFs with their checksums.
        
        Args:
            pdf_info: Dict mapping PDF filenames to their checksums
        """
        tracking_file = self.db_dir / 'processed_pdfs.json'
        
        # Load existing data
        existing_pdfs = self.get_processed_pdfs()
        
        # Add/update new PDFs
        existing_pdfs.update(pdf_info)
        
        # Save updated list
        data = {
            'processed_pdfs': existing_pdfs,
            'last_updated_on': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(tracking_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Updated processed PDFs tracking: {len(existing_pdfs)} total")
        
    def load_database(self) -> Chroma:
        """
        Load existing vector database.
        
        Returns:
            ChromaDB vector store instance
        """
        if not self.db_dir.exists():
            raise FileNotFoundError(f"Database directory '{self.db_dir}' not found")
        
        vectordb = Chroma(
            persist_directory=str(self.db_dir),
            embedding_function=self.embeddings
        )
        
        return vectordb
    
    
# Example usage:
##vector_db = VectorDatabase(db_dir="chroma_db", embedding_model="nomic-embed-text")
##vectordb = vector_db.load_database()
##print(f"Loaded vector database with {vectordb._collection.count()} vectors")