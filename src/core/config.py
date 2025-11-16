"""
Configuration Management

Loading and managing configuration for Ollama and Azure OpenAI.
"""

import os
import toml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class OllamaConfig:
    """Ollama configuration."""
    base_url: str
    embedding_model: str
    default_llm: str


@dataclass
class AzureConfig:
    """Azure OpenAI configuration."""
    api_key: str
    api_version: str
    azure_endpoint: str
    embedding_deployment: str
    llm_deployment: str
    embedding_model: str
    llm_model: str


@dataclass
class Config:
    """Application configuration."""
    provider: str
    data_dir: str
    db_dir: str
    chunk_size: int
    chunk_overlap: int
    default_top_k: int
    temperature: float
    max_tokens: int
    ollama: Optional[OllamaConfig] = None
    azure: Optional[AzureConfig] = None


def load_config(config_path: str = "config.toml") -> Config:
    """
    Load configuration from TOML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Config object
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        # Return default Ollama config
        return Config(
            provider="ollama",
            data_dir="data",
            db_dir="chroma_db",
            chunk_size=1000,
            chunk_overlap=200,
            default_top_k=5,
            temperature=0.7,
            max_tokens=2000,
            ollama=OllamaConfig(
                base_url="http://localhost:11434",
                embedding_model="nomic-embed-text",
                default_llm="llama3.2"
            )
        )
    
    # Load from file
    data = toml.load(config_file)
    #print(f"Loaded configuration from {config_path}")
    #print(data)
    
    # Access nested sections using .get() with default values
    # Format: data.get("section", {}).get("key", "default_value")

    provider = data.get("provider", {}).get("type", "ollama")
    
    # Load Ollama config
    ollama_config = None
    if "ollama" in data:
        ollama_config = OllamaConfig(
            base_url=data["ollama"].get("base_url", "http://localhost:11434"),
            embedding_model=data["ollama"].get("embedding_model", "nomic-embed-text"),
            default_llm=data["ollama"].get("default_llm", "llama3.2")
        )
    
    # Load Azure config
    azure_config = None
    if "azure" in data:
        # Get API key from environment or config
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or data["azure"].get("api_key", "")
        
        azure_config = AzureConfig(
            api_key=api_key,
            api_version=data["azure"].get("api_version", "2024-02-15-preview"),
            azure_endpoint=data["azure"].get("azure_endpoint", ""),
            embedding_deployment=data["azure"].get("embedding_deployment", ""),
            llm_deployment=data["azure"].get("llm_deployment", ""),
            embedding_model=data["azure"].get("embedding_model", "text-embedding-ada-002"),
            llm_model=data["azure"].get("llm_model", "gpt-4")
        )
    
    return Config(
        provider=provider,
        data_dir=data.get("paths", {}).get("data_dir", "data"),
        db_dir=data.get("paths", {}).get("db_dir", "chroma_db"),
        chunk_size=data.get("chunking", {}).get("chunk_size", 1000),
        chunk_overlap=data.get("chunking", {}).get("chunk_overlap", 200),
        default_top_k=data.get("retrieval", {}).get("default_top_k", 5),
        temperature=data.get("llm", {}).get("temperature", 0.7),
        max_tokens=data.get("llm", {}).get("max_tokens", 2000),
        ollama=ollama_config,
        azure=azure_config
    )


def get_embedding_function(config: Config):
    """
    Get the appropriate embedding function based on provider.
    
    Args:
        config: Configuration object
        
    Returns:
        Embedding function instance
    """
    if config.provider == "azure" and config.azure:
        from langchain_openai import AzureOpenAIEmbeddings
        
        if not config.azure.api_key:
            raise ValueError(
                "Azure OpenAI API key not found."
                "Set AZURE_OPENAI_API_KEY environment variable or add to config.toml"
            )
        
        return AzureOpenAIEmbeddings(
            azure_deployment=config.azure.embedding_deployment,
            api_version=config.azure.api_version,
            azure_endpoint=config.azure.azure_endpoint,
            api_key=config.azure.api_key  # type: ignore
        )
    else:
        # Default to Ollama
        from langchain_ollama import OllamaEmbeddings
        
        base_url = config.ollama.base_url if config.ollama else "http://localhost:11434"
        model = config.ollama.embedding_model if config.ollama else "nomic-embed-text"
        
        return OllamaEmbeddings(
            model=model,
            base_url=base_url
        )


def get_llm(config: Config, model_name: Optional[str] = None):
    """
    Get the appropriate LLM based on provider.
    
    Args:
        config: Configuration object
        model_name: Optional model name override
        
    Returns:
        LLM instance
    """
    if config.provider == "azure" and config.azure:
        from langchain_openai import AzureChatOpenAI
        
        if not config.azure.api_key:
            raise ValueError(
                "Azure OpenAI API key not found. "
                "Set AZURE_OPENAI_API_KEY environment variable or add to config.toml"
            )
        
        return AzureChatOpenAI(
            azure_deployment=config.azure.llm_deployment,
            api_version=config.azure.api_version,
            azure_endpoint=config.azure.azure_endpoint,
            api_key=config.azure.api_key,  # type: ignore
            temperature=config.temperature,
            max_completion_tokens=config.max_tokens
        )
    else:
        # Default to Ollama
        from langchain_ollama import ChatOllama
        
        base_url = config.ollama.base_url if config.ollama else "http://localhost:11434"
        model = model_name or (config.ollama.default_llm if config.ollama else "llama3.2")
        
        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=config.temperature
        )


# Global config instance
_config = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config

##load_config()