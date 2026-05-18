import json

def print_chunk_data_by_index(filename="chroma_db/chunk_data.jsonl"):
    """
    Reads a JSONL file, and prints all entries where 'metadata.chunk_index' is 0.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    if isinstance(item, dict) and 'metadata' in item and \
                       isinstance(item['metadata'], dict) and \
                       item['metadata'].get('chunk_index') == 0:
                        print(f"--- Chunk Index 0 from {item['metadata'].get('source', 'unknown')} ---")
                        print(item.get('text', 'No text found'))
                        print("-" * 50)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON on line {line_num}")

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    print_chunk_data_by_index()
