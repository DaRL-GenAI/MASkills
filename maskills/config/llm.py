"""LLM configuration module for supporting multiple model providers.

All providers use OpenAI-compatible API format.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class LLMConfig:
    """Configuration for a single LLM model."""

    name: str
    model_string: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env_var: str = "OPENAI_API_KEY"
    is_multimodal: bool = False
    max_tokens: int = 4096
    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def get_api_key(self) -> str:
        """Get API key from config or environment variable."""
        if self.api_key:
            return self.api_key
        key = os.getenv(self.api_key_env_var)
        if not key:
            raise ValueError(
                f"API key not found. Set '{self.api_key_env_var}' environment variable "
                f"or provide 'api_key' in config."
            )
        return key

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'model_string': self.model_string,
            'base_url': self.base_url,
            'api_key': self.api_key,
            'api_key_env_var': self.api_key_env_var,
            'is_multimodal': self.is_multimodal,
            'max_tokens': self.max_tokens,
            'input_price_per_million': self.input_price_per_million,
            'output_price_per_million': self.output_price_per_million,
            'extra_params': self.extra_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_preset(cls, name: str) -> 'LLMConfig':
        """Create LLMConfig from a predefined model name."""
        return get_llm_config(name)


PREDEFINED_MODELS: Dict[str, LLMConfig] = {
    # OpenAI Models
    'gpt-4o': LLMConfig(
        name='gpt-4o', model_string='gpt-4o',
        api_key_env_var='OPENAI_API_KEY', is_multimodal=True, max_tokens=4096,
        input_price_per_million=5.0, output_price_per_million=15.0,
    ),
    'gpt-4o-mini': LLMConfig(
        name='gpt-4o-mini', model_string='gpt-4o-mini',
        api_key_env_var='OPENAI_API_KEY', is_multimodal=True, max_tokens=4096,
        input_price_per_million=0.15, output_price_per_million=0.60,
    ),
    'gpt-3.5-turbo': LLMConfig(
        name='gpt-3.5-turbo', model_string='openai/gpt-3.5-turbo',
        api_key_env_var='OPENAI_API_KEY', is_multimodal=False, max_tokens=4096,
        input_price_per_million=0.5, output_price_per_million=1.5,
    ),
    'gpt-5.1': LLMConfig(
        name='gpt-5.1', model_string='openai/gpt-5.1',
        api_key_env_var='OPENAI_API_KEY', is_multimodal=True, max_tokens=8192,
        input_price_per_million=10.0, output_price_per_million=30.0,
    ),
    'gpt-5': LLMConfig(
        name='gpt-5', model_string='gpt-5',
        api_key_env_var='OPENAI_API_KEY', is_multimodal=True, max_tokens=8192,
        input_price_per_million=10.0, output_price_per_million=30.0,
    ),
    'o1': LLMConfig(
        name='o1', model_string='o1',
        api_key_env_var='OPENAI_API_KEY', is_multimodal=True, max_tokens=32768,
        input_price_per_million=15.0, output_price_per_million=60.0,
    ),
    'o1-mini': LLMConfig(
        name='o1-mini', model_string='o1-mini',
        api_key_env_var='OPENAI_API_KEY', is_multimodal=False, max_tokens=32768,
        input_price_per_million=3.0, output_price_per_million=12.0,
    ),
    # Google Gemini
    'gemini-pro': LLMConfig(
        name='gemini-pro', model_string='gemini-1.5-pro',
        base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
        api_key_env_var='GOOGLE_API_KEY', is_multimodal=True, max_tokens=8192,
        input_price_per_million=1.25, output_price_per_million=5.0,
    ),
    'gemini-flash': LLMConfig(
        name='gemini-flash', model_string='gemini-1.5-flash',
        base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
        api_key_env_var='GOOGLE_API_KEY', is_multimodal=True, max_tokens=8192,
        input_price_per_million=0.075, output_price_per_million=0.30,
    ),
    'gemini-2.0-flash': LLMConfig(
        name='gemini-2.0-flash', model_string='gemini-2.0-flash',
        base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
        api_key_env_var='GOOGLE_API_KEY', is_multimodal=True, max_tokens=8192,
        input_price_per_million=0.10, output_price_per_million=0.40,
    ),
    # Llama (via Together)
    'llama-3.1-70b': LLMConfig(
        name='llama-3.1-70b', model_string='meta-llama/llama-3.1-70b-instruct',
        base_url='https://api.together.xyz/v1', api_key_env_var='TOGETHER_API_KEY',
        max_tokens=4096, input_price_per_million=0.88, output_price_per_million=0.88,
    ),
    'llama-3.1-8b': LLMConfig(
        name='llama-3.1-8b', model_string='meta-llama/llama-3.1-8b-instruct',
        base_url='https://api.together.xyz/v1', api_key_env_var='TOGETHER_API_KEY',
        max_tokens=4096, input_price_per_million=0.18, output_price_per_million=0.18,
    ),
    'llama-3.3-70b': LLMConfig(
        name='llama-3.3-70b', model_string='meta-llama/Llama-3.3-70B-Instruct-Turbo',
        base_url='https://api.together.xyz/v1', api_key_env_var='TOGETHER_API_KEY',
        max_tokens=4096, input_price_per_million=0.88, output_price_per_million=0.88,
    ),
    # Qwen (via Together)
    'qwen-72b': LLMConfig(
        name='qwen-72b', model_string='Qwen/Qwen2.5-72B-Instruct-Turbo',
        base_url='https://api.together.xyz/v1', api_key_env_var='TOGETHER_API_KEY',
        max_tokens=4096, input_price_per_million=1.2, output_price_per_million=1.2,
    ),
    'qwen-7b': LLMConfig(
        name='qwen-7b', model_string='Qwen/Qwen2.5-7B-Instruct-Turbo',
        base_url='https://api.together.xyz/v1', api_key_env_var='TOGETHER_API_KEY',
        max_tokens=4096, input_price_per_million=0.30, output_price_per_million=0.30,
    ),
    'qwen-coder-32b': LLMConfig(
        name='qwen-coder-32b', model_string='Qwen/Qwen2.5-Coder-32B-Instruct',
        base_url='https://api.together.xyz/v1', api_key_env_var='TOGETHER_API_KEY',
        max_tokens=4096, input_price_per_million=0.80, output_price_per_million=0.80,
    ),
    # DeepSeek
    'deepseek-chat': LLMConfig(
        name='deepseek-chat', model_string='deepseek-chat',
        base_url='https://api.deepseek.com/v1', api_key_env_var='DEEPSEEK_API_KEY',
        max_tokens=4096, input_price_per_million=0.14, output_price_per_million=0.28,
    ),
    'deepseek-reasoner': LLMConfig(
        name='deepseek-reasoner', model_string='deepseek-reasoner',
        base_url='https://api.deepseek.com/v1', api_key_env_var='DEEPSEEK_API_KEY',
        max_tokens=8192, input_price_per_million=0.55, output_price_per_million=2.19,
    ),
    # Local Ollama
    'ollama-llama3': LLMConfig(
        name='ollama-llama3', model_string='llama3',
        base_url='http://localhost:11434/v1', api_key='ollama',
        max_tokens=4096,
    ),
    'ollama-qwen2': LLMConfig(
        name='ollama-qwen2', model_string='qwen2',
        base_url='http://localhost:11434/v1', api_key='ollama',
        max_tokens=4096,
    ),
}


def get_llm_config(name_or_path: str) -> LLMConfig:
    """Get LLM config by predefined name or load from JSON file."""
    if name_or_path in PREDEFINED_MODELS:
        return PREDEFINED_MODELS[name_or_path]

    path = Path(name_or_path)
    if path.exists() and path.suffix == '.json':
        with open(path) as f:
            data = json.load(f)
        return LLMConfig.from_dict(data)

    available = list(PREDEFINED_MODELS.keys())
    raise ValueError(
        f"Unknown model '{name_or_path}'. "
        f"Available predefined models: {available}. "
        f"Or provide a path to a JSON config file."
    )


def save_llm_config(config: LLMConfig, output_path: str):
    """Save LLM config to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


def list_available_models() -> Dict[str, str]:
    """List all available predefined models."""
    return {
        name: f"{cfg.model_string} via {cfg.base_url or 'OpenAI API'}"
        for name, cfg in PREDEFINED_MODELS.items()
    }
