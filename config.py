import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration management class."""
    
    # Azure OpenAI Configuration
    AZURE_OPENAI = {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "api_version_4o": os.getenv("AZURE_OPENAI_API_VERSION_4o"),
        "api_version_35_turbo": os.getenv("AZURE_OPENAI_API_VERSION_35_TURBO"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "deployment_gpt_4": os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT_4"),
        "deployment_gpt_4o_mini": os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT_4o_MINI"),
        "deployment_gpt_35_turbo": os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT_35_TURBO"),
        "embedding_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        "embedding_version": os.getenv("AZURE_OPENAI_EMBEDDING_VERSION")
    }
    
    # Azure Search Configuration
    AZURE_SEARCH = {
        "endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
        "key": os.getenv("AZURE_SEARCH_KEY"),
        "index_name": os.getenv("AZURE_SEARCH_INDEX")
    }
    
    # Agent Configuration
    AGENT = {
        "memory_window": int(os.getenv("MEMORY_WINDOW")),
        "max_retries": int(os.getenv("MAX_RETRIES")),
        "timeout": None  
    }
    
    MODELS= {
        "models": [
            {
            "name": "GPT-4o-mini",
            "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT_4o_MINI"),
            "version": os.getenv("AZURE_OPENAI_API_VERSION_4o")
            },
            {
            "name": "GPT-4",
            "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT_4"),
            "version":  os.getenv("AZURE_OPENAI_API_VERSION_4o")
            },
            {
            "name": "GPT-35-turbo",
            "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT_35_TURBO"),
            "version": os.getenv("AZURE_OPENAI_API_VERSION_35_TURBO")
            },
            {
            "name": "GPT-o3",
            "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT_4o_MINI"),
            "version": os.getenv("AZURE_OPENAI_API_VERSION_4o")
            }
        ]
        }
    