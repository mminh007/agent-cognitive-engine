# core/settings.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, BaseModel, Field

class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="OPENAI_", extra="ignore")
    api_key: SecretStr | None = None
    base_url: str = "https://models.inference.ai.azure.com"
    # ─── LLM TIER CONFIGURATION ───
    # Tier 1: High-speed, ultra-low cost. Used for routing, classification, and background data extraction.
    tier1_fast_model: str = "gpt-4o-mini" 
    
    # Tier 2: Balanced cost/performance. Used for general software engineering and standard chat.
    tier2_balanced_model: str = "gpt-4o" 
    
    # Tier 3: High-reasoning, expensive. Reserved strictly for complex academic analysis and vision matrix calculations.
    tier3_reasoning_model: str = "o1-mini" # Or claude-3.5-sonnet if supporting multiple providers
    
    # ─── TOKEN GOVERNANCE ───
    max_completion_tokens: int = 1024  
    max_context_tokens: int = 8192



# Redis configuration for short-term memory management and session state caching
class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="REDIS_", extra="ignore")
    url: str = "redis://localhost:6379/0"
    ttl: int = 3600

# ChromaDB configuration for vector storage management
class ChromaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CHROMA_", extra="ignore")
    path: str = "./chroma_db"
    collection_name: str = "long_term_memory"
    embedding_model: str = "text-embedding-3-small"
    server_host: str = "localhost"
    server_port: str = "8000"

# RabbitMQ configuration for potential future message queue integrations (search for "RabbitMQSettings" in the codebase for usage contexts)
class RabbitMQSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RABBITMQ_", extra="ignore")
    url: str = "amqp://guest:guest@localhost:5672/"
    queue_name: str = "fact_extraction_queue"

# tool for Tavily integration (placeholder for future expansion) (search for "TavilySettings" in the codebase for usage contexts)
class TavilySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="TAVILY_", extra="ignore")
    api_key: SecretStr | None = None
    max_token_budget: int = 2000

# Logs configuration for application-wide logging management
class LogsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="LOGS_", extra="ignore")
    dir: str = "./logs"
    max_bytes: int = 10485760
    backup_count: int = 5

class Settings(BaseSettings):
    """Unified application configuration manager grouping domain-specific sub-models."""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    rabbitmq: RabbitMQSettings = Field(default_factory=RabbitMQSettings)
    logs: LogsSettings = Field(default_factory=LogsSettings)
    tavily: TavilySettings = Field(default_factory=TavilySettings)
    
@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()