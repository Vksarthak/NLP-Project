from pathlib import Path
from typing import Dict, List, Any

class PromptError(Exception):
    """Custom exception for prompt template related errors."""
    pass

class PromptManager:
    """Manager for loading and rendering prompt templates."""

    def __init__(self, prompts_dir: str = 'prompts'):
        """
        Initialize the PromptManager.

        Args:
            prompts_dir: The directory containing prompt templates.
        """
        self.prompts_dir = Path(prompts_dir)
        self.cache: Dict[str, str] = {}

    def load_template(self, template_name: str) -> str:
        """
        Load a template from the prompts directory. Caches the result.

        Args:
            template_name: The name of the template file (with or without .txt extension).

        Returns:
            The template content.

        Raises:
            PromptError: If the template file does not exist or cannot be read.
        """
        if not template_name.endswith('.txt'):
            template_name += '.txt'
            
        if template_name in self.cache:
            return self.cache[template_name]
            
        template_path = self.prompts_dir / template_name
        
        if not template_path.is_file():
            raise PromptError(f"Template '{template_name}' not found at {template_path}")
            
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.cache[template_name] = content
                return content
        except Exception as e:
            raise PromptError(f"Error loading template '{template_name}': {e}")

    def render(self, template_name: str, **variables: Any) -> str:
        """
        Load a template and fill in variables.

        Args:
            template_name: The name of the template file.
            **variables: Keyword arguments to format the template with.

        Returns:
            The rendered string.

        Raises:
            PromptError: If required variables are missing or template cannot be loaded.
        """
        template = self.load_template(template_name)
        
        try:
            # Custom dictionary to handle missing keys in format_map
            class SafeDict(dict):
                def __missing__(self, key):
                    raise KeyError(key)
            
            return template.format_map(SafeDict(**variables))
            
        except KeyError as e:
            raise PromptError(f"Missing required variable for template '{template_name}': {e}")
        except ValueError as e:
            raise PromptError(f"Error formatting template '{template_name}': {e}")

    def list_templates(self) -> List[str]:
        """
        List all available templates in the prompts directory.

        Returns:
            List of template names (without extensions).
        """
        if not self.prompts_dir.is_dir():
            return []
            
        templates = []
        for file_path in self.prompts_dir.glob('*.txt'):
            templates.append(file_path.stem)
            
        return sorted(templates)
