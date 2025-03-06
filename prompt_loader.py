import os
from typing import Dict, Any

class PromptLoader:
    """Handles loading and parsing of .prompt files."""
    
    @staticmethod
    def load_prompt_file(file_path: str) -> str:
        """
        Load and return the content of a .prompt file.
        
        Args:
            file_path (str): Path to the .prompt file
            
        Returns:
            str: Content of the prompt file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Error loading prompt file {file_path}: {str(e)}")
    
    @staticmethod
    def parse_master_prompts(content: str) -> Dict[str, str]:
        """
        Parse the master agent prompt file content into separate prompts.
        
        Args:
            content (str): Content of the master prompt file
            
        Returns:
            Dict[str, str]: Dictionary containing different prompt sections
        """
            
        return content
    
    @staticmethod
    def get_prompt_paths(base_dir: str = None) -> Dict[str, str]:
        """
        Get paths to all prompt files.
        
        Args:
            base_dir (str): Base directory containing prompt files
            
        Returns:
            Dict[str, str]: Dictionary mapping agent names to prompt file paths
        """
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts')
        return {
            'child_agent1': os.path.join(base_dir, 'child_agent1.prompty'),
            'master_agent': os.path.join(base_dir, 'master_agent.prompty'),
            'refiner_agent': os.path.join(base_dir, 'refiner_agent.prompty')
        }