"""
RAG Streamlit Web Interface

A user-friendly web interface for the RAG system using Streamlit.
"""

import sys
from pathlib import Path
import pickle

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.rag.ragengine import RAGEngine


# Page configuration
st.set_page_config(
    page_title="RAG Chat",
    page_icon="📚",
    layout="wide"
)


@st.cache_resource
def load_rag_system(llm_model: str):
    """Load and cache the RAG system with fine-tuned defaults."""
    try:
        rag = RAGEngine(
            db_dir="chroma_db",
            embedding_model="nomic-embed-text",
            llm_model=llm_model
            # Using fine-tuned defaults: temperature=0.3, top_k=3, memory_window=2
        )
        return rag, None
    except Exception as e:
        return None, str(e)


def load_database_stats():
    """Load database statistics from RAG system."""
    try:
        # Use the RAG system to get stats directly from ChromaDB
        from src.rag.ragengine import RAGEngine
        
        rag = RAGEngine(db_dir="chroma_db")
        stats = rag.get_database_stats()
        
        return {
            'total_chunks': stats['total_chunks'],
            'total_documents': stats['total_pdfs'],
            'sources': stats['pdf_files']
        }
    except FileNotFoundError:
        # Database doesn't exist yet
        return None
    except Exception as e:
        st.error(f"Error loading stats: {e}")
        return None


def main():
    """Main Streamlit app."""
    
    # Header
    st.title("📚 RAG Chat Interface")
    st.markdown("Ask questions about your documents using AI-powered search and generation")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Load default model from config
        try:
            from src.core.config import load_config
            config = load_config()
            default_model = config.ollama.default_llm if config.ollama else "llama3.2"
        except:
            default_model = "llama3.2"
        
        # Model selection
        available_models = ["llama3.2", "llama3.1", "mistral", "deepseek-r1:7b"]
        
        # Set default index
        try:
            default_index = available_models.index(default_model)
        except ValueError:
            default_index = 0
        
        llm_model = st.selectbox(
            "LLM Model",
            available_models,
            index=default_index,
            help="Select the Ollama model to use (from config.toml)"
        )
        
        # Top-k selection - dynamic max based on database
        stats = load_database_stats()
        max_chunks = stats['total_chunks'] if stats else 10
        default_top_k = min(3, max_chunks)  # Use fine-tuned default of 3
        
        top_k = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=max_chunks,
            value=default_top_k,
            help=f"Retrieve up to {max_chunks} chunks from your documents (Fine-tuned default: 3)" 
        )
        
        # Quick access buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⚡ Fast (3)", help="Fine-tuned for speed"):
                top_k = 3
        with col2:
            if st.button("📚 All chunks", help=f"Use all {max_chunks} chunks"):
                top_k = max_chunks
        
        # Show sources toggle
        show_sources = st.checkbox("Show source documents", value=True)
        
        st.divider()
        
        # Database stats
        st.header("📊 Database Stats")
        
        # Add refresh button
        col1, col2 = st.columns([3, 1])
        with col2:
            refresh = st.button("🔄", help="Refresh stats")
        
        # Reload stats if not already loaded (for top_k calculation)
        if 'stats' not in locals():
            stats = load_database_stats()
        
        if stats:
            st.metric("Total Chunks", stats['total_chunks'])
            st.metric("Total Documents", stats['total_documents'])
            
            with st.expander("📄 Source Documents"):
                for source in stats['sources']:
                    st.text(f"• {source}")
        else:
            st.warning("No database found. Run: python main.py process")
        
        st.divider()
        
        # Instructions
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            1. Make sure Ollama is running
            2. Process PDFs: `python main.py process`
            3. Click 🔄 to refresh database stats
            4. Select your preferred LLM model
            5. Ask questions about your documents
            6. Toggle sources to see where answers come from
            """)
    
    # Main chat area
    if stats is None:
        st.error("❌ Database not found. Please run `python main.py process` first.")
        st.stop()
    
    # Load RAG system
    rag, error = load_rag_system(llm_model)
    
    if error:
        st.error(f"❌ Error initializing RAG system: {error}")
        st.info("Make sure Ollama is running and the model is pulled.")
        st.stop()
    
    if rag is None:
        st.error("❌ Failed to initialize RAG system")
        st.stop()
    
    # Update top_k
    rag.top_k = top_k
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "sources" in message:
                with st.expander(f"📚 Sources ({len(message['sources'])} chunks)"):
                    for i, source in enumerate(message["sources"], 1):
                        metadata = source.get('metadata', {})
                        filename = metadata.get('filename', 'Unknown')
                        page = metadata.get('page', 'Unknown')
                        content = source.get('content', '')[:200] + "..." if len(source.get('content', '')) > 200 else source.get('content', '')
                            
                        st.markdown(f"**[{i}]** {filename} (Page {page})")
                        st.text(content)
                        st.divider()
    
    # Chat input
    if question := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag.chat(question, verbose=show_sources)
                st.markdown(response['answer'])
                
                # Show sources if enabled
                if show_sources and 'sources' in response:
                    with st.expander(f"📚 Sources ({response['retrieved_chunks']} chunks)"):
                        for i, source in enumerate(response['sources'], 1):
                            metadata = source.get('metadata', {})
                            filename = metadata.get('filename', 'Unknown')
                            page = metadata.get('page', 'Unknown')
                            content = source.get('content', '')[:200] + "..." if len(source.get('content', '')) > 200 else source.get('content', '')
                            
                            st.markdown(f"**[{i}]** {filename} (Page {page})")
                            st.text(content)
                            st.divider()
        
        # Add assistant response to chat
        message_data = {
            "role": "assistant",
            "content": response['answer']
        }
        if show_sources and 'sources' in response:
            message_data["sources"] = response['sources']
        
        st.session_state.messages.append(message_data)
    
    # Clear chat button
    if st.session_state.messages:
        if st.sidebar.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()