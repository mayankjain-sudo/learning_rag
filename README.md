# Learning RAG

A small RAG learning project that processes PDFs, chunks text, and stores embeddings in ChromaDB. Supports Ollama and Azure embeddings and allows attaching custom metadata per document.

## Usage

Place PDFs in the `data/` directory. Then run:

```bash
python3 scripts/process_pdfs.py metadata.json
```

To force a full rebuild (re-extract and re-embed):

```bash
python3 scripts/process_pdfs.py --force metadata.json
```

This writes the database into `chroma_db/` and a debug file `chroma_db/chunk_data.json` containing all chunk texts and merged metadata.

## Custom Metadata (metadata.json)

You can provide metadata in any of the following formats, and it will be merged into each chunk's metadata:

- Global-only (applies to all PDFs):
```json
{
	"project": "RAG Learning",
	"description": "Repository for learning Retrieval-Augmented Generation (RAG) techniques using AI models."
}
```

- Explicit global + per-file overrides:
```json
{
	"global": {
		"project": "RAG Learning",
		"department": "AI"
	},
	"files": {
		"handbook.pdf": { "category": "docs", "keywords": ["guide", "intro"] },
		"other.pdf": { "classification": "internal" }
	}
}
```

- Per-file only (original format):
```json
{
	"handbook.pdf": { "category": "docs", "keywords": ["guide", "intro"] }
}
```

Notes:
- Final metadata merge order: PDF metadata → global metadata → per-file metadata (per-file overrides global).
- Recognized fields (others are allowed and passed through): `category`, `keywords`, `department`, `classification`, `language`, `version`, `project`, `year`, `type`, `status`, `tags`, `priority`, `confidential`.

