from typing import Optional
from langchain.chains import ConversationChain
from langchain_openai import AzureChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory,ConversationSummaryMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import time
from flask_socketio import SocketIO, emit
from logging import getLogger
from config import Config

logger = getLogger(__name__)

class Chatbot:
    """Handles chat interactions using Azure OpenAI."""
    
    def __init__(self, socketio):
        """
        Initialize Chatbot with configurable memory window.
        
        Args:
            memory_window (int): Number of previous conversations to remember
        """
        self.config = Config.AZURE_OPENAI
        self.socketio = socketio
        try:
            self.memory = ConversationBufferWindowMemory(k=1, return_messages=True)
            self._initialize_llm(deployment=self.config["deployment_gpt_4o_mini"],version=self.config["api_version_4o"])
        except Exception as e:
            logger.error(f"Failed to initialize Chatbot: {str(e)}")
            raise
                

    def _initialize_llm(self,deployment,version) -> AzureChatOpenAI:
        """Initialize the Azure OpenAI LLM with configuration settings."""
        try:
            self.llm = AzureChatOpenAI(
                azure_deployment=deployment,
                api_version=version,
                temperature=0,
                max_retries=Config.AGENT["max_retries"],
                azure_endpoint=self.config["azure_endpoint"],
                api_key=self.config["api_key"],
                streaming=True
            )
            print("****************************************************")
            print("chatbot initlized with, deployment :",deployment)
            print("****************************************************")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise

    def _setup_conversation(self, system_prompt: str) -> Optional[ConversationChain]:
        """
        Set up the conversation chain with appropriate templates.
        
        Args:
            system_prompt (str): The system prompt to use
            
        Returns:
            ConversationChain: Configured conversation chain
        """
        try:
            system_template = SystemMessagePromptTemplate.from_template(
                template=system_prompt
            )
            human_template = HumanMessagePromptTemplate.from_template(
                template="{input}"
            )
            prompt_template = ChatPromptTemplate.from_messages([
                system_template,
                MessagesPlaceholder(variable_name="history"),
                human_template
            ])
            return ConversationChain(
                memory=self.memory,
                prompt=prompt_template,
                llm=self.llm
            )
        except Exception as e:
            logger.error(f"Failed to setup conversation: {str(e)}")
            return None
        
    def response_stream(self, context, query, system_promt):

        try:
            chain = self._setup_conversation(system_promt)
            self.socketio.emit('data', {'char': "response from: child_bot1, "})
                    
            for chunk in chain.predict(input=f"Context:\n {context} \n\n Query:\n{query}"):
                self.socketio.emit('data', {'char': chunk})
                time.sleep(0.01)
                
            self.socketio.emit('stream_complete')
        except:
            emit('data', {'char': "An error occurred while processing your query."})
            emit('stream_complete')

    def reset_memory(self) -> None:
        """Reset the conversation memory."""
        try:
            self.memory.clear()
            print("memory reset")
            self._initialize_llm(deployment=self.config["deployment_gpt_4o_mini"],version=self.config["api_version_4o"])
        except Exception as e:
            logger.error(f"Failed to reset memory: {str(e)}")
            raise