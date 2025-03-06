from abc import ABC, abstractmethod
from typing import Any, Dict
from logging import getLogger

logger = getLogger(__name__)

class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    @abstractmethod
    def process_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process a query and return updated state."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize any required resources."""
        pass

    def handle_error(self, error: Exception, context: str) -> None:
        """Common error handling method for all agents."""
        logger.error(f"Error in {context}: {str(error)}")
        raise error