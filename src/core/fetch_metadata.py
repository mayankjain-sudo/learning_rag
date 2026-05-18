"""
Fetch Metadata Script

This script iterates through existing chunks, generates summaries using an LLM,
and updates the metadata in both the local JSON storage and ChromaDB.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import tqdm

from config import get_config, get_llm, get_embedding_function
from vector_db import VectorDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



def generate_summary(llm, text: str) -> str:
    """Generate a concise summary of the text chunk."""
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in 1-2 sentences, capturing the key information:\n\n{text}"
    )
    chain = prompt | llm | StrOutputParser()
    
    try:
        return chain.invoke({"text": text})
    except Exception as e:
        print(f"Error generating summary: {e}")
        return ""

def main():
    config = get_config()
    db_dir = Path(config.db_dir)
    
    # Initialize LLM and VectorDB
    print("Initializing LLM and Database...")
    llm = get_llm(config)
    
    # We need the vector db instance to update metadata
    vdb_wrapper = VectorDatabase(db_dir=str(db_dir), embeddings=get_embedding_function(config))
    vectordb = vdb_wrapper.load_database()
    
    print("Fetching all documents from ChromaDB...")
    try:
        # Fetch all data from Chroma
        results = vectordb.get()
        ids = results['ids']
        documents = results['documents']
        metadatas = results['metadatas']
        
        if not ids:
            print("No chunks found in ChromaDB.")
            return
            
        print(f"Found {len(ids)} chunks in ChromaDB.")
        
        updated_count = 0
        
        print("Starting summarization...")
        for i, chunk_id in enumerate(tqdm.tqdm(ids)):
            text = documents[i]
            metadata = metadatas[i] if metadatas[i] else {}
            
            # Check if summary already exists
            if 'summary' in metadata and metadata['summary']:
                continue
                
            # Generate summary
            summary = generate_summary(llm, text)
            if not summary:
                continue
                
            # Update metadata
            metadata['summary'] = summary
            
            try:
                # Sanitize metadata before update
                sanitized_metadata = VectorDatabase._sanitize_metadata(metadata)
                # Use internal collection to update metadata only, avoiding re-embedding
                vectordb._collection.update(ids=[chunk_id], metadatas=[sanitized_metadata])
                updated_count += 1
            except Exception as e:
                print(f"Failed to update ChromaDB for chunk {chunk_id}: {e}")
                
        print(f"Finished. Updated {updated_count} chunks.")
        
    except Exception as e:
        print(f"Error processing chunks: {e}")

if __name__ == "__main__":
    main()
