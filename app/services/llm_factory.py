from typing import Any, Dict, List, Type
import instructor
from anthropic import Anthropic
from pydantic import BaseModel
from config.settings import get_settings


class LLMFactory:
    def __init__(self, provider: str = "anthropic"):
        self.provider = provider
        self.settings = getattr(get_settings(), provider)
        self.client = self._initialize_client()
        
    def _initialize_client(self) -> Any:
        """Initialize the appropriate client based on provider."""
        client_initializers = {
            "anthropic": lambda s: instructor.from_anthropic(
                Anthropic(
                    api_key=s.api_key,
                    timeout=s.timeout
                )
            ),
        }
        
        initializer = client_initializers.get(self.provider)
        if initializer:
            return initializer(self.settings)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def create_completion(
        self, 
        response_model: Type[BaseModel], 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Any:
        """Create a completion using the configured LLM provider."""
        completion_params = {
            "model": kwargs.get("model", self.settings.default_model),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "response_model": response_model,
            "messages": messages,
        }
        return self.client.chat.completions.create(**completion_params)