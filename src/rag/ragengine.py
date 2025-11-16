""""
RAG Chat Interface

Implements a Retrieval-Augmented Generation system that:
1. Retrieves relevant chunks from the vector database
2. Uses configured LLM (Ollama or Azure OpenAI) to generate answers
"""

from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import hashlib
import time
import json
from datetime import datetime
import uuid


class RAGEngine:
    """RAG Engine for retrieval-augmented generation."""
    
    def __init__(
        self,
        db_dir: str = "chroma_db",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2",
        temperature: float = 0.3,
        top_k: int = 3,
        use_config: bool = True,
        enable_cache: bool = True,
        cache_ttl: int = 3600,
        sessions_dir: str = "sessions",
        memory_window: int = 2,
        ):
        """
        Initialize RAG engine.
        
        Args:
            db_dir: Directory containing the vector database
            embedding_model: Embedding model name (legacy, use config)
            llm_model: LLM model name (legacy, use config)
            temperature: LLM temperature (0.0 to 1.0) - Fine-tuned default: 0.3
            top_k: Number of relevant chunks to retrieve - Fine-tuned default: 3
            use_config: Whether to use config.toml for provider settings
            enable_cache: Enable query result caching
            cache_ttl: Cache time-to-live in seconds
            sessions_dir: Directory for storing chat sessions
            memory_window: Number of previous messages to include - Fine-tuned default: 2
        """
        
        self.db_dir = Path(db_dir)
        self.top_k = top_k
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self.memory_window = memory_window
        
        # Session management
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)
        self.current_session_id: Optional[str] = None
        self.chat_history: List[Dict[str, str]] = []
        
        # Query cache: {query_hash: (result, timestamp)}
        self._query_cache: Dict[str, Tuple[Dict, float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        if use_config:
            # Use configuration system
            from src.core.config import load_config, get_embedding_function, get_llm
            
            self.config = load_config()
            self.embeddings = get_embedding_function(self.config)
            self.llm = get_llm(self.config, model_name=llm_model if llm_model != "llama3.2" else None)
        else:
            # Legacy direct initialization
            from langchain_ollama import OllamaEmbeddings, ChatOllama
            
            self.embeddings = OllamaEmbeddings(model=embedding_model)
            self.llm = ChatOllama(
                model=llm_model,
                temperature=temperature
            )
        
        # Load vector database
        self.vectordb = self._load_database()
        
        # Create prompt template
        self.prompt_template = self._create_prompt_template()
        
    def _load_database(self) -> Chroma:
        """
        Load the vector database.
        
        Returns:
            ChromaDB vector store instance
        """
        if not self.db_dir.exists():
            raise FileNotFoundError(
                f"Database directory '{self.db_dir}' not found. "
                "Please run pdf_processor.py first to create the database."
            )
        
        vectordb = Chroma(
            persist_directory=str(self.db_dir),
            embedding_function=self.embeddings
        )
        
        return vectordb
        
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            # Get all documents
            collection = self.vectordb._collection
            all_data = collection.get()
            
            total_chunks = len(all_data['ids']) if all_data['ids'] else 0
            
            # Extract unique filenames
            unique_files = set()
            metadatas = all_data.get('metadatas')
            if metadatas:
                for metadata in metadatas:
                    if metadata and 'filename' in metadata:
                        unique_files.add(metadata['filename'])
            
            return {
                'total_pdfs': len(unique_files),
                'total_chunks': total_chunks,
                'pdf_files': sorted(list(unique_files))
            }
        except Exception as e:
            return {
                'total_pdfs': 0,
                'total_chunks': 0,
                'pdf_files': [],
                'error': str(e)
            }
            
    def get_document_info(self, filename: str) -> Dict[str, Any] | None:
        """
        Get information about a specific document.
        
        Args:
            filename: Name of the PDF file
            
        Returns:
            Dictionary with document information or None if not found
        """
        try:
            collection = self.vectordb._collection
            # Get one chunk from this document to extract metadata
            data = collection.get(
                limit=1,
                where={"filename": filename}
            )
            
            metadatas = data.get('metadatas') if data else None
            if not metadatas or not metadatas[0]:
                return None
            
            metadata = metadatas[0]
            
            return {
                'filename': metadata.get('filename', filename),
                'title': metadata.get('title', 'Unknown'),
                'author': metadata.get('author', 'Unknown'),
                'total_pages': metadata.get('total_pages', 'Unknown'),
                'subject': metadata.get('subject', 'Unknown'),
                'creator': metadata.get('creator', 'Unknown')
            }
        except Exception as e:
            return None
        
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """
        Create the prompt template for RAG with conversation memory.
        
        Returns:
            ChatPromptTemplate instance
        """
        template = """You are a helpful assistant that answers questions based on the provided context from documents and previous conversation.

Previous Conversation:
{conversation_history}

Context from documents:
{context}

Current Question: {question}

Instructions:
- Answer the current question based on the provided document context and conversation history
- Use conversation history to understand follow-up questions and maintain context
- If the question refers to something mentioned earlier ("it", "that", "the document"), use the conversation history
- Cite which document or page the information comes from when possible
- Be concise but thorough
- If you're unsure, say "I don't have enough information to answer that accurately"

Answer:"""
        
        return ChatPromptTemplate.from_template(template)
    
    
    def retrieve_context(self, question: str) -> List[Document]:
        """
        Retrieve relevant document chunks for a question.
        
        Args:
            question: User's question
            
        Returns:
            List of relevant document chunks
        """
        results = self.vectordb.similarity_search(question, k=self.top_k)
        return results
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents, start=1):
            metadata = doc.metadata
            source = metadata.get('filename', metadata.get('source', 'Unknown'))
            page = metadata.get('page', 'Unknown')
            
            context_part = f"[Document {i}] From: {source}, Page: {page}\n{doc.page_content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
   
    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using LLM based on question, context, and conversation history.
        
        Args:
            question: User's question
            context: Retrieved context
            
        Returns:
            Generated answer
        """
        # Format conversation history
        conversation_history = self._format_conversation_history()
        
        # Create the prompt
        prompt = self.prompt_template.format(
            conversation_history=conversation_history,
            context=context,
            question=question
        )
        
        # Generate response
        response = self.llm.invoke(prompt)
        
        # Handle both string and structured responses
        if isinstance(response.content, str):
            return response.content
        else:
            return str(response.content)
        
    def _format_conversation_history(self) -> str:
        """
        Format recent conversation history for the prompt.
        
        Returns:
            Formatted conversation history string
        """
        if not self.chat_history:
            return "No previous conversation."
        
        # Get the last N messages based on memory_window
        recent_messages = self.chat_history[-self.memory_window * 2:] if self.memory_window > 0 else []
        
        if not recent_messages:
            return "No previous conversation."
        
        formatted = []
        for msg in recent_messages:
            role = "User" if msg['role'] == 'user' else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted)
    
    def _get_query_hash(self, question: str) -> str:
        """Generate a hash for the query to use as cache key."""
        return hashlib.md5(question.lower().strip().encode()).hexdigest()
    
    def _get_cached_response(self, question: str) -> Optional[Dict]:
        """Get cached response if available and not expired."""
        if not self.enable_cache:
            return None
        
        query_hash = self._get_query_hash(question)
        
        if query_hash in self._query_cache:
            result, timestamp = self._query_cache[query_hash]
            
            # Check if cache entry is still valid
            if time.time() - timestamp < self.cache_ttl:
                self._cache_hits += 1
                result_copy = result.copy()
                result_copy['from_cache'] = True
                result_copy['cached_at'] = timestamp
                return result_copy
            else:
                # Remove expired entry
                del self._query_cache[query_hash]
        
        self._cache_misses += 1
        return None
    
    def _cache_response(self, question: str, response: Dict) -> None:
        """Cache the query response."""
        if not self.enable_cache:
            return
        
        query_hash = self._get_query_hash(question)
        # Don't include cache metadata in stored response
        clean_response = {k: v for k, v in response.items() if k not in ['from_cache', 'cached_at']}
        self._query_cache[query_hash] = (clean_response, time.time())
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        
        return {
            'enabled': self.enable_cache,
            'cache_size': len(self._query_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'ttl_seconds': self.cache_ttl
        }
    
    def clear_cache(self) -> int:
        """Clear the query cache. Returns number of entries cleared."""
        count = len(self._query_cache)
        self._query_cache.clear()
        return count
    
    def _analyze_query_intent(self, question: str) -> Dict[str, Any]:
        """
        Analyze query to determine the best search strategy using semantic understanding.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with query analysis results:
            - strategy: 'metadata' or 'llm'
            - category: type of query (e.g., 'database_stats', 'metadata_field', 'content_search')
            - field: specific metadata field if applicable
            - confidence: confidence score (0-100)
            - reason: explanation of the routing decision
        """
        question_lower = question.lower()
        
        # Database statistics patterns (High confidence)
        db_patterns = {
            'count_queries': ['how many', 'count', 'total number', 'number of'],
            'list_queries': ['list', 'show me', 'display', 'enumerate'],
            'target_objects': ['pdf', 'document', 'file', 'paper']
        }
        
        # Check if asking about database/collection
        has_count = any(pattern in question_lower for pattern in db_patterns['count_queries'])
        has_list = any(pattern in question_lower for pattern in db_patterns['list_queries'])
        has_target = any(pattern in question_lower for pattern in db_patterns['target_objects'])
        
        if (has_count or has_list) and has_target:
            return {
                'strategy': 'metadata',
                'category': 'database_stats',
                'field': None,
                'confidence': 95,
                'reason': 'Query asks about database statistics or file listing'
            }
        
        # Metadata field patterns with semantic grouping
        metadata_patterns = {
            'pages': {
                'keywords': ['how many page', 'total page', 'number of page', 'page count', 'pages in', 'pages of', 'pages are', 'length of'],
                'semantic_indicators': ['long', 'size', 'length']
            },
            'author': {
                'keywords': ['who is the author', 'who are the author', 'who wrote', 'author of', 'authors of', 
                          'written by', 'authored by', 'created by', 'list author', 'who created'],
                'semantic_indicators': ['writer', 'creator', 'publisher']
            },
            'title': {
                'keywords': ['what is the title', 'what are the title', 'title of', 'titles of', 'document title', 
                         'name of the document', 'document name', 'list title', 'called'],
                'semantic_indicators': ['name', 'heading', 'called']
            },
            'subject': {
                'keywords': ['what is the subject', 'what are the subject', 'subject of', 'subjects of', 
                           'about what', 'topic of', 'topics of'],
                'semantic_indicators': ['theme', 'focus', 'content']
            }
        }
        
        # Check metadata fields with improved matching
        for field, patterns in metadata_patterns.items():
            # Check direct keywords
            if any(keyword in question_lower for keyword in patterns['keywords']):
                return {
                    'strategy': 'metadata',
                    'category': 'metadata_field',
                    'field': field,
                    'confidence': 92,
                    'reason': f'Query asks about {field} metadata field (keyword match)'
                }
            
            # Check semantic indicators (lower confidence)
            if any(indicator in question_lower for indicator in patterns['semantic_indicators']):
                # Additional check: must mention a document
                if has_target or any(word in question_lower for word in ['this', 'the document', 'it']):
                    return {
                        'strategy': 'metadata',
                        'category': 'metadata_field',
                        'field': field,
                        'confidence': 75,
                        'reason': f'Query likely asks about {field} (semantic match)'
                    }
        
        # Content-based query patterns (should use LLM) - Enhanced detection
        content_patterns = {
            'explanation': ['explain', 'describe', 'what does', 'what is', 'define'],
            'instruction': ['how to', 'how can', 'how do', 'steps to', 'way to'],
            'reasoning': ['why', 'reason', 'because', 'cause'],
            'comparison': ['difference between', 'compare', 'contrast', 'versus', 'vs'],
            'example': ['example', 'instance', 'demonstrate', 'show me how'],
            'process': ['process', 'procedure', 'workflow', 'method'],
            'evaluation': ['best', 'better', 'worst', 'recommend', 'suggest', 'should i'],
            'problem_solving': ['problem', 'issue', 'error', 'fix', 'troubleshoot', 'solve'],
            'understanding': ['learn', 'understand', 'know about', 'tell me about']
        }
        
        for category, patterns in content_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                # Exception: metadata questions that start with these words
                if not any(field_patterns['keywords'][0] in question_lower 
                          for field_patterns in metadata_patterns.values()):
                    return {
                        'strategy': 'llm',
                        'category': 'content_search',
                        'field': None,
                        'confidence': 88,
                        'reason': f'Query requires {category} and LLM reasoning'
                    }
        
        # Check for complex sentences (multiple clauses usually need LLM)
        clause_indicators = [' and ', ' or ', ' but ', ' when ', ' where ', ' if ']
        if sum(1 for ind in clause_indicators if ind in question_lower) >= 2:
            return {
                'strategy': 'llm',
                'category': 'complex_query',
                'field': None,
                'confidence': 82,
                'reason': 'Complex multi-clause query requires contextual understanding'
            }
        
        # Default: use LLM for ambiguous queries
        return {
            'strategy': 'llm',
            'category': 'general_query',
            'field': None,
            'confidence': 65,
            'reason': 'Query type unclear, defaulting to full RAG pipeline for comprehensive answer'
        }
        
    def chat(self, question: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Main RAG pipeline: analyze query and route to appropriate search strategy with caching.
        
        Args:
            question: User's question
            verbose: Whether to return additional information
            
        Returns:
            Dictionary containing answer and optional metadata
        """
        # Check cache first
        cached_response = self._get_cached_response(question)
        if cached_response:
            if verbose:
                print(f"\n💾 Cache HIT! (saved ~{self.cache_ttl}s processing time)")
                print(f"   Cached at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cached_response['cached_at']))}\n")
            return cached_response
        
        # Analyze query to determine search strategy
        analysis = self._analyze_query_intent(question)
        
        # Add question to chat history
        self._add_to_history("user", question)
        
        if verbose:
            print(f"\n🔍 Query Analysis:")
            print(f"   Strategy: {analysis['strategy'].upper()}")
            print(f"   Category: {analysis['category']}")
            print(f"   Confidence: {analysis['confidence']}%")
            print(f"   Reason: {analysis['reason']}\n")
        
        # Route to appropriate handler based on analysis
        response = None
        if analysis['strategy'] == 'metadata':
            if analysis['category'] == 'database_stats':
                response = self._handle_database_stats_query(question, analysis)
            elif analysis['category'] == 'metadata_field':
                response = self._handle_metadata_field_query(question, analysis)
        
        # Default: use full RAG pipeline with LLM
        if not response:
            response = self._handle_llm_query(question, analysis, verbose)
        
        # Add answer to chat history
        self._add_to_history("assistant", response['answer'])
        
        # Cache the response
        self._cache_response(question, response)
        
        return response
    
    def _handle_database_stats_query(self, question: str, analysis: Dict) -> Dict[str, Any]:
        """Handle database statistics queries."""
        stats = self.get_database_stats()
        
        if stats['total_pdfs'] == 0:
            answer = "No PDFs have been loaded into the database yet."
        else:
            answer = f"There are {stats['total_pdfs']} PDF(s) loaded in the database:\n\n"
            for i, filename in enumerate(stats['pdf_files'], 1):
                answer += f"{i}. {filename}\n"
            answer += f"\nTotal chunks: {stats['total_chunks']}"
        
        return {
            'answer': answer,
            'sources': [],
            'retrieved_chunks': 0,
            'database_stats': stats,
            'query_analysis': analysis
        }
    
    def _handle_metadata_field_query(self, question: str, analysis: Dict) -> Dict[str, Any]:
        """Handle metadata field queries (pages, author, title, subject)."""
        question_lower = question.lower()
        matched_field = analysis['field']
        
        # Check if asking about ALL documents
        all_docs_keywords = ['all document', 'all pdf', 'all file', 'every document', 'every pdf', 
                            'each document', 'each pdf', 'complete list', 'full list', 'all the']
        asking_about_all = any(keyword in question_lower for keyword in all_docs_keywords)
        
        # Also check for plural without specific filename
        stats = self.get_database_stats()
        has_specific_file = any(
            filename.lower().replace('.pdf', '') in question_lower 
            for filename in stats['pdf_files']
        )
        
        # If plural forms are used and no specific file is mentioned, assume "all"
        plural_indicators = ['authors are', 'titles are', 'authors of', 'titles of', 'pages are']
        if not has_specific_file and any(indicator in question_lower for indicator in plural_indicators):
            asking_about_all = True
        
        if asking_about_all:
            # Return metadata for all documents
            answer = f"**Metadata for all {stats['total_pdfs']} documents:**\n\n"
            
            for i, filename in enumerate(stats['pdf_files'], 1):
                doc_info = self.get_document_info(filename)
                if doc_info:
                    answer += f"**{i}. {doc_info['filename']}**\n"
                    if matched_field == 'pages':
                        answer += f"   Pages: {doc_info['total_pages']}\n"
                    elif matched_field == 'author':
                        author = doc_info['author'] if doc_info['author'] else '(anonymous)'
                        answer += f"   Author: {author}\n"
                    elif matched_field == 'title':
                        title = doc_info['title'] if doc_info['title'] else '(untitled)'
                        answer += f"   Title: {title}\n"
                    elif matched_field == 'subject':
                        subject = doc_info['subject'] if doc_info['subject'] else '(unspecified)'
                        answer += f"   Subject: {subject}\n"
                    answer += "\n"
            
            return {
                'answer': answer,
                'sources': [],
                'retrieved_chunks': 0,
                'matched_field': matched_field,
                'all_documents': True,
                'query_analysis': analysis
            }
        
        # Try to find specific file
        matched_files = []
        
        # Find all matching files
        for filename in stats['pdf_files']:
            file_base = filename.replace('.pdf', '').lower()
            filename_lower = filename.lower()
            
            # Multiple matching strategies:
            # 1. Exact filename match
            if filename_lower in question_lower or file_base in question_lower:
                matched_files.append((filename, 100))  # Priority 100
                continue
            
            # 2. Partial match of significant parts (length > 3)
            parts = [p for p in file_base.split('-') if len(p) > 3]
            if any(part in question_lower for part in parts):
                matched_files.append((filename, 90))  # Priority 90
                continue
            
            # 3. Check for numbers/versions (e.g., "2024" from "guide-2024.pdf")
            import re
            file_numbers = re.findall(r'\d+', file_base)
            question_numbers = re.findall(r'\d+', question_lower)
            if file_numbers and question_numbers:
                if any(num in file_numbers for num in question_numbers):
                    matched_files.append((filename, 80))  # Priority 80
        
        # Sort by priority and get the best match
        if matched_files:
            matched_files.sort(key=lambda x: x[1], reverse=True)
            best_match = matched_files[0][0]
            
            doc_info = self.get_document_info(best_match)
            if doc_info:
                # Format answer based on what was asked
                if matched_field == 'pages':
                    answer = f"**{doc_info['filename']}** has **{doc_info['total_pages']} pages**."
                elif matched_field == 'author':
                    author = doc_info['author'] if doc_info['author'] else '(anonymous)'
                    answer = f"**{doc_info['filename']}**\n\n**Author:** {author}"
                elif matched_field == 'title':
                    title = doc_info['title'] if doc_info['title'] else '(untitled)'
                    answer = f"**{doc_info['filename']}**\n\n**Title:** {title}"
                elif matched_field == 'subject':
                    subject = doc_info['subject'] if doc_info['subject'] else '(unspecified)'
                    answer = f"**{doc_info['filename']}**\n\n**Subject:** {subject}"
                else:
                    # Full metadata
                    answer = f"**{doc_info['filename']}**\n\n"
                    answer += f"**Title:** {doc_info['title']}\n"
                    answer += f"**Total Pages:** {doc_info['total_pages']}\n"
                    answer += f"**Author:** {doc_info['author']}\n"
                    if doc_info['subject']:
                        answer += f"**Subject:** {doc_info['subject']}\n"
                
                return {
                    'answer': answer,
                    'sources': [],
                    'retrieved_chunks': 0,
                    'document_info': doc_info,
                    'matched_field': matched_field,
                    'query_analysis': analysis
                }
        
        # If no file matched, fall back to LLM
        return self._handle_llm_query(question, analysis, verbose=False)
    
    def _handle_llm_query(self, question: str, analysis: Dict, verbose: bool = False) -> Dict[str, Any]:
        """Handle queries that require LLM and full RAG pipeline."""
        # Regular RAG query
        # Retrieve relevant context
        documents = self.retrieve_context(question)
        
        if not documents:
            return {
                'answer': "I couldn't find any relevant information in the documents.",
                'sources': [],
                'retrieved_chunks': 0,
                'query_analysis': analysis
            }
        
        # Format context
        context = self.format_context(documents)
        
        # Generate answer
        answer = self.generate_answer(question, context)
        
        # Prepare response
        response = {
            'answer': answer,
            'retrieved_chunks': len(documents),
            'query_analysis': analysis
        }
        
        # Add sources if requested
        if verbose:
            sources = []
            for doc in documents:
                source_info = {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                sources.append(source_info)
            response['sources'] = sources
        
        return response
    
    def interactive_chat(self):
        """
        Start an interactive chat session.
        """
        print("=== RAG Chat Interface ===")
        print(f"Using database: {self.db_dir}")
        
        # Handle model name for both Ollama and Azure
        model_name = getattr(self.llm, 'model', getattr(self.llm, 'deployment_name', 'Unknown'))
        print(f"LLM Model: {model_name}")
        
        print("\nType your questions (or 'quit' to exit, 'sources' to show sources)")
        print("-" * 60)
        
        show_sources = False
        
        while True:
            try:
                question = input("\nYou: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if question.lower() == 'sources':
                    show_sources = not show_sources
                    status = "enabled" if show_sources else "disabled"
                    print(f"\nSource display {status}")
                    continue
                
                # Get response
                response = self.chat(question, verbose=show_sources)
                
                # Display answer
                print(f"\nAssistant: {response['answer']}")
                
                # Display sources if enabled
                if show_sources and 'sources' in response:
                    print(f"\n--- Sources ({response['retrieved_chunks']} chunks retrieved) ---")
                    for i, source in enumerate(response['sources'], start=1):
                        print(f"  [{i}] {source['filename']} (Page {source['page']})")
                        print(f"      Preview: {source['content_preview']}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                
    def _add_to_history(self, role: str, content: str):
        """
        Add a message to the current chat history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        # Auto-create session on first message if none exists
        if not self.current_session_id:
            print(f"[DEBUG] No active session, creating new session")
            self.new_session()
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.chat_history.append(message)
        
        # Auto-save the session
        print(f"[DEBUG] Auto-saving session: {self.current_session_id}")
        result = self.save_current_session()
        print(f"[DEBUG] Auto-save result: {result}")
            
    def save_current_session(self) -> bool:
        """
        Save the current chat session to disk.
        
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.current_session_id:
            return False
        
        session_data = {
            "session_id": self.current_session_id,
            "updated_at": datetime.now().isoformat(),
            "messages": self.chat_history
        }
        
        return self._save_session(session_data)
    
    def _save_session(self, session_data: Dict[str, Any]) -> bool:
        """
        Internal method to save session data to disk.
        
        Args:
            session_data: Session data dict
            
        Returns:
            True if saved successfully
        """
        session_file = self.sessions_dir / f"{session_data['session_id']}.json"
        
        try:
            print(f"[DEBUG] Saving session to: {session_file}")
            print(f"[DEBUG] Sessions directory exists: {self.sessions_dir.exists()}")
            print(f"[DEBUG] Session ID: {session_data['session_id']}")
            
            # Load existing data if file exists
            if session_file.exists():
                print(f"[DEBUG] Session file exists, loading existing data")
                with open(session_file, 'r') as f:
                    existing_data = json.load(f)
                # Merge with new data
                existing_data.update(session_data)
                session_data = existing_data
            else:
                print(f"[DEBUG] Creating new session file")
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            print(f"[DEBUG] Session saved successfully")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save session: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_session_history(self, session_id: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Get the chat history for a session.
        
        Args:
            session_id: ID of the session (None for current session)
            
        Returns:
            List of message dicts with 'role', 'content', 'timestamp'
        """
        if session_id is None:
            return self.chat_history.copy()
        
        session_file = self.sessions_dir / f"{session_id}.json"
        
        if not session_file.exists():
            return []
        
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            return session_data.get("messages", [])
        except Exception:
            return []
        
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a saved session.
        
        Args:
            session_id: ID of the session to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        session_file = self.sessions_dir / f"{session_id}.json"
        
        if session_file.exists():
            try:
                session_file.unlink()
                
                # Clear current session if it was the deleted one
                if self.current_session_id == session_id:
                    self.current_session_id = None
                    self.chat_history = []
                
                return True
            except Exception:
                return False
        
        return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all saved sessions.
        
        Returns:
            List of session metadata dicts
        """
        sessions = []
        
        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                sessions.append({
                    "session_id": session_data["session_id"],
                    "name": session_data.get("name", "Unnamed Session"),
                    "created_at": session_data.get("created_at", "Unknown"),
                    "updated_at": session_data.get("updated_at", "Unknown"),
                    "message_count": len(session_data.get("messages", []))
                })
            except Exception:
                continue
        
        # Sort by updated_at (most recent first)
        sessions.sort(key=lambda x: x["updated_at"], reverse=True)
        
        return sessions
    
    def load_session(self, session_id: str) -> bool:
        """
        Load a chat session from disk.
        
        Args:
            session_id: ID of the session to load
            
        Returns:
            True if loaded successfully, False otherwise
        """
        session_file = self.sessions_dir / f"{session_id}.json"
        
        if not session_file.exists():
            return False
        
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            self.current_session_id = session_data["session_id"]
            self.chat_history = session_data.get("messages", [])
            
            return True
        except Exception as e:
            print(f"Error loading session: {e}")
            return False
        
    def new_session(self, session_name: Optional[str] = None) -> str:
        """
        Start a new chat session.
        
        Args:
            session_name: Optional name for the session
            
        Returns:
            Session ID
        """
        self.current_session_id = str(uuid.uuid4())
        self.chat_history = []
        
        session_data = {
            "session_id": self.current_session_id,
            "name": session_name or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": []
        }
        
        self._save_session(session_data)
        return self.current_session_id
    
def main():
    """Main entry point."""
    import sys
    
    # Parse command line arguments
    llm_model = "llama3.2"
    if len(sys.argv) > 1:
        llm_model = sys.argv[1]
    
    print(f"Initializing RAG system with model: {llm_model}")
    print("(Make sure Ollama is running and the model is pulled)")
    print()
    
    try:
        # Create RAG chat instance
        rag = RAGEngine(
            db_dir="chroma_db",
            embedding_model="nomic-embed-text",
            llm_model=llm_model,
            temperature=0.7,
            top_k=5
        )
        
        # Start interactive chat
        rag.interactive_chat()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run 'python pdf_processor.py' first to create the database.")
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        print("\nMake sure:")
        print("1. Ollama is running")
        print(f"2. The model '{llm_model}' is pulled: ollama pull {llm_model}")
        print("3. The database exists in the 'db/' directory")


if __name__ == "__main__":
    main()