"""
Vector Database Module

It handles the creation, storage, and querying of vector database using ChromaDB.
Supports both Ollama and Azure OpenAI embeddings.
"""

import pickle
import json
import hashlib
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import time

class CustomSemanticChunker:
    """
    Custom semantic chunker that splits text based on semantic similarity.
    Uses embeddings to detect natural breakpoints in content.
    """
    
    def __init__(self, embeddings, breakpoint_threshold_type: str = "percentile", 
                 breakpoint_threshold_amount: float = 95, min_chunk_size: int = 100):
        """
        Initialize the semantic chunker.
        
        Args:
            embeddings: Embedding function to use
            breakpoint_threshold_type: Type of threshold ("percentile" or "standard_deviation")
            breakpoint_threshold_amount: Threshold value (percentile or std dev multiplier)
            min_chunk_size: Minimum characters per chunk
        """
        self.embeddings = embeddings
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.min_chunk_size = min_chunk_size
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Split on sentence boundaries (., !, ?) followed by whitespace
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        # Filter out empty strings and very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences
    
    def _calculate_cosine_distances(self, embeddings_list: List[List[float]]) -> List[float]:
        """
        Calculate cosine distances between consecutive embeddings.
        
        Args:
            embeddings_list: List of embedding vectors
            
        Returns:
            List of distances between consecutive embeddings
        """
        distances = []
        for i in range(len(embeddings_list) - 1):
            # Convert to numpy arrays
            emb1 = np.array(embeddings_list[i])
            emb2 = np.array(embeddings_list[i + 1])
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            # Convert to distance (1 - similarity)
            distance = 1 - similarity
            distances.append(distance)
        
        return distances
    
    def _calculate_threshold(self, distances: List[float]) -> float:
        """
        Calculate the threshold for determining breakpoints.
        
        Args:
            distances: List of cosine distances
            
        Returns:
            Threshold value
        """
        if self.breakpoint_threshold_type == "percentile":
            return float(np.percentile(distances, self.breakpoint_threshold_amount))
        elif self.breakpoint_threshold_type == "standard_deviation":
            mean = np.mean(distances)
            std = np.std(distances)
            return float(mean + (self.breakpoint_threshold_amount * std))
        else:
            # Default to 95th percentile
            return float(np.percentile(distances, 95))
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into semantic chunks.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        # Handle empty or very short text
        if not text or len(text) < self.min_chunk_size:
            return [text] if text else []
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [text]
        
        # Get embeddings for each sentence
        try:
            sentence_embeddings = self.embeddings.embed_documents(sentences)
        except Exception as e:
            print(f"Warning: Could not generate embeddings for semantic chunking: {e}")
            # Fallback to returning the whole text
            return [text]
        
        # Calculate distances between consecutive sentences
        distances = self._calculate_cosine_distances(sentence_embeddings)
        
        if not distances:
            return [text]
        
        # Calculate threshold for breakpoints
        threshold = self._calculate_threshold(distances)
        
        # Identify breakpoints where distance exceeds threshold
        breakpoints = [i + 1 for i, dist in enumerate(distances) if dist > threshold]
        
        # Add start and end points
        breakpoints = [0] + breakpoints + [len(sentences)]
        
        # Create chunks from sentences between breakpoints
        chunks = []
        for i in range(len(breakpoints) - 1):
            start_idx = breakpoints[i]
            end_idx = breakpoints[i + 1]
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = ' '.join(chunk_sentences)
            
            # Ensure minimum chunk size
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
            elif chunks:  # Append to previous chunk if too small
                chunks[-1] += ' ' + chunk_text
            else:  # First chunk, keep even if small
                chunks.append(chunk_text)
        
        return chunks if chunks else [text]


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
        chunking_strategy: str = "semantic", # recursive or semantic
    ):
        """
        Initializes the VectorDatabase with a configuration and embedding function.
        Uses semantic chunking to create meaningful chunks based on content similarity.

        Args:
            db_dir: Directory to store the ChromaDB database.
            embedding_model: The embedding model to use.
            embedding: Optional embedding function.
            
        Note:
            Requires Ollama or embedding service to be running for semantic chunking.
        """
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(exist_ok=True)
        self.chunking_strategy = chunking_strategy
        
        #Use provided embedding or default to Ollama
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            from langchain_ollama import OllamaEmbeddings
            self.embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Initialize Text Splitter
        # Semantic chunking creates chunks based on meaning/semantic similarity
        # Splits occur at natural semantic boundaries rather than fixed character counts

        if self.chunking_strategy == "semantic":
            self.text_splitter = CustomSemanticChunker(
                self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95,
                min_chunk_size=200
            )
            print("Using custom semantic chunking (meaning-based splits)")
        else:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            print("Using recursive character text splitting")
        
    
    @staticmethod
    def _sanitize_metadata(metadata: Dict) -> Dict:
        """
        Sanitize metadata to ensure all values are compatible with ChromaDB.
        ChromaDB only accepts: str, int, float, bool, or None.
        Converts lists and other complex types to strings.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Sanitized metadata dictionary
        """
        sanitized = {}
        for key, value in metadata.items():
            if value is None or isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                # Convert list to comma-separated string
                sanitized[key] = ", ".join(str(v) for v in value)
            elif isinstance(value, dict):
                # Convert dict to JSON string
                sanitized[key] = json.dumps(value)
            else:
                # Convert any other type to string
                sanitized[key] = str(value)
        return sanitized
            
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
                # Ensure reliable source identifiers for downstream UI/rendering
                # Prefer explicit fields; fallback gracefully if missing
                source_filename = (
                    pdf_metadata.get('filename')
                    or pdf_metadata.get('source')
                    or pdf_metadata.get('file_path')
                    or pdf_metadata.get('name')
                    or ""
                )
                document_id = (
                    pdf_metadata.get('document_id')
                    or pdf_metadata.get('checksum')
                    or pdf_metadata.get('sha256')
                    or ""
                )
                title = pdf_metadata.get('title') or pdf_metadata.get('pdf_title') or ""
                if source_filename:
                    chunk_metadata['source_filename'] = source_filename
                if document_id:
                    chunk_metadata['document_id'] = document_id
                if title:
                    chunk_metadata['title'] = title
                chunk_metadata['page'] = page_data['page']
                chunk_metadata['total_pages'] = page_data['total_pages']
                chunk_metadata['chunk_index'] = chunk_idx
                chunk_metadata['last_updated_on'] = time.strftime("%Y-%m-%d %H:%M:%S")
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
        
        # Sanitize metadata: ChromaDB only accepts str, int, float, bool, or None
        # Convert lists and other complex types to strings
        metadatas = [self._sanitize_metadata(m) for m in metadatas]
        
        
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
        Appends new chunks to existing data instead of overwriting.

        Args:
            texts: List of text chunks.
            metadatas: List of metadata dictionaries.
        """
        json_path = self.db_dir / "chunk_data.json"
        pickle_path = self.db_dir / "chunk_data.pkl"
        
        # Load existing data if it exists
        existing_texts = []
        existing_metadatas = []
        
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    existing_texts = existing_data.get('texts', [])
                    existing_metadatas = existing_data.get('metadata', [])
            except Exception as e:
                print(f"Warning: Could not load existing chunk data: {e}")
        
        # Append new chunks to existing
        all_texts = existing_texts + texts
        all_metadatas = existing_metadatas + metadatas
        
        chunk_data = {
            'texts': all_texts, 
            'metadata': all_metadatas,
            'last_updated': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_chunks': len(all_texts)
        }
        
        # Save as JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=4)
        print(f"Saved chunk data to {json_path} (total: {len(all_texts)} chunks)")
        
        # Save as pickle
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
                    elif isinstance(pdfs, dict):
                        # Newer format may store dict per file with checksum and timestamp
                        result: Dict[str, str] = {}
                        for fname, val in pdfs.items():
                            if isinstance(val, dict):
                                result[fname] = val.get('checksum', "")
                            else:
                                # Backward-compat: value is checksum string
                                result[fname] = str(val)
                        return result
                    else:
                        return {}
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
        
        # Load existing full structure if present
        existing_full: Dict[str, Any] = {}
        if tracking_file.exists():
            try:
                with open(tracking_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data.get('processed_pdfs'), dict):
                        existing_full = data.get('processed_pdfs', {})
                    elif isinstance(data.get('processed_pdfs'), list):
                        # Old list format -> convert to dict entries with empty checksum
                        existing_full = {name: {"checksum": "", "last_updated_on": ""} for name in data.get('processed_pdfs', [])}
            except Exception as e:
                print(f"Warning: Could not load existing processed PDFs for update: {e}")
                existing_full = {}

        # Normalize existing entries to dict form
        normalized: Dict[str, Dict[str, Any]] = {}
        for fname, val in existing_full.items():
            if isinstance(val, dict):
                normalized[fname] = {
                    "checksum": val.get("checksum", ""),
                    "last_updated_on": val.get("last_updated_on", "")
                }
            else:
                normalized[fname] = {"checksum": str(val), "last_updated_on": ""}

        # Merge incoming info with per-file timestamp
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        for fname, checksum in pdf_info.items():
            normalized[fname] = {
                "checksum": checksum,
                "last_updated_on": ts
            }

        # Save updated structure; keep a file-level timestamp for convenience
        out = {
            'processed_pdfs': normalized,
            'last_updated_on': ts
        }

        with open(tracking_file, 'w') as f:
            json.dump(out, f, indent=2)

        print(f"Updated processed PDFs tracking: {len(normalized)} total")
        
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