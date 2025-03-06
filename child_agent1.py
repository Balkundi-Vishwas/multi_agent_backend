from typing import Dict, Any, Optional, List, Tuple
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from prompt_loader import PromptLoader
from base_agent import BaseAgent
from logging import getLogger
from config import Config

logger = getLogger(__name__)

class ChildAgent1(BaseAgent):
    """Agent responsible for processing queries using vector search and chatbot."""

    def __init__(self,chatbot):
        """Initialize the child agent with necessary components."""
        self.embeddings: Optional[AzureOpenAIEmbeddings] = None
        self.vector_store: Optional[AzureSearch] = None
        self.chatbot = chatbot
        self.initialize()

        self.prompt_loader = PromptLoader()
        self.system_prompt = self.prompt_loader.load_prompt_file(
            self.prompt_loader.get_prompt_paths()['child_agent1']
        )
        print("child agent initlized")

    def initialize(self) -> None:
        """Initialize embeddings, vector store, and chatbot."""
        openai = Config.AZURE_OPENAI
        search = Config.AZURE_SEARCH
        try:
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=openai["embedding_deployment"],
                openai_api_version=openai["embedding_version"],
                azure_endpoint=openai["azure_endpoint"],
                api_key=openai["api_key"]
            )

            self.vector_store = AzureSearch(
                azure_search_endpoint=search["endpoint"],
                azure_search_key=search["key"],
                index_name=search["index_name"],          
                embedding_function=self.embeddings.embed_query
            )
        except Exception as e:
            self.handle_error(e, "initialization")

    def find_match(self, query: str, k: int = 2, threshold: float = 0.80) -> str:
        """
        Find matching documents for the query.
        
        Args:
            query (str): The search query
            k (int): Number of documents to retrieve
            threshold (float): Similarity score threshold
            
        Returns:
            str: Concatenated relevant document content
        """
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
                
            docs = self.vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                score_threshold=threshold
            )
            return "\n".join(doc[0].page_content for doc in docs)
        except Exception as e:
            self.handle_error(e, "document search")
            return ""

    def process_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the query and generate response using the chatbot.
        
        Args:
            state (Dict[str, Any]): Current state containing the query
            
        Returns:
            Dict[str, Any]: Updated state
        """
        try:
            if not self.chatbot:
                raise ValueError("Chatbot not initialized")
            query = state.get('refined_query')
            if not query:
                raise ValueError("No query found in state")

            context = self.find_match(query=query)
            self.chatbot.response_stream(context=context, query=query, system_promt= self.system_prompt)
            return state
            
        except Exception as e:
            self.handle_error(e, "query processing")
            # emit('data', {'char': "An error occurred while processing your query."})
            # emit('stream_complete')
            return state