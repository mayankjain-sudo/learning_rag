# Implementation Plan - Chunk Summarization

The goal is to create `fetch_metadata.py` which generates summaries for existing chunks and updates their metadata in both the local JSON storage and the ChromaDB embeddings.

## User Review Required
> [!IMPORTANT]
> This script assumes that the text content of chunks is unique enough to map back to ChromaDB IDs. If there are duplicate chunks with identical text, the mapping might be ambiguous.
> I will also modify `vector_db.py` to save IDs in the future, but for existing data, we will attempt to recover IDs by querying Chroma.

## Proposed Changes

### Core Components

#### [NEW] [fetch_metadata.py](file:///Users/mayankjain/learning/ai/antigravity_workspace/learning_rag/src/core/fetch_metadata.py)
- **Purpose**: Iterate over chunks, generate summaries using an LLM, and update metadata.
- **Key Functions**:
    - `load_data()`: Load `chunk_data.json`.
    - `get_chroma_ids()`: Fetch all IDs and documents from Chroma to create a Text -> ID mapping.
    - `generate_summary(text)`: Call LLM (Ollama) to summarize text.
    - `process_chunks()`: Main loop to process, summarize, and update.
    - `update_chroma()`: Update ChromaDB with new metadata.
    - `save_data()`: Save updated `chunk_data.json`.

#### [MODIFY] [vector_db.py](file:///Users/mayankjain/learning/ai/antigravity_workspace/learning_rag/src/core/vector_db.py)
- **Change**: Capture IDs returned by `vectordb.add_texts` or `from_texts`.
- **Reason**: To make future updates easier and more reliable.
- **Modification**: Update `store_chunks` to collect IDs and pass them to `_save_chunk_data`. Update `_save_chunk_data` to store IDs.

## Verification Plan

### Automated Tests
- Run `fetch_metadata.py` and check if `chunk_data.json` is updated with `summary` fields.
- Query ChromaDB to verify that metadata now contains `summary`.

### Manual Verification
- Inspect `chunk_data.json` manually.
- Use a small script to query Chroma and print metadata for a sample chunk.
