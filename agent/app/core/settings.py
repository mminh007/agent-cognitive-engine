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

class GeminiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="GEMINI_", extra="ignore")
    
    # Google AI Studio API Key (AIzaSy...)
    api_key: SecretStr | None = None
    
    # Use native Google GenAI SDK by default (far more stable for LangGraph than OpenAI wrapper)
    base_url: str | None = None
    
    # ─── LLM TIER CONFIGURATION ───
    # Tier 1: High-speed, ultra-low cost. Perfect for supervisor routing and fast text/vision classifications.
    tier1_fast_model: str = "gemini-2.0-flash"
    
    # Tier 2: Balanced cost/performance. Used for core software engineering loops (general_memory) and prompt-parsed JSON.
    tier2_balanced_model: str = "gemini-2.0-flash" # Can upgrade to "gemini-1.5-pro" if you hit Tier 1 rate limits or need higher analytical depth
    
    # Tier 3: High-reasoning. Reserved for deep academic research (research_papers) with extensive context windows.
    tier3_reasoning_model: str = "gemini-2.0-pro"
    
    # ─── EMBEDDING MODEL ───
    embedding_model: str = "models/text-embedding-004"
    
    # ─── TOKEN GOVERNANCE ───
    # Gemini models support huge token outputs and context windows.
    max_completion_tokens: int = 2048   # Expanded slightly as Gemini handles verbose output well
    max_context_tokens: int = 32768     # Gemini's native context is up to 1M-2M tokens, but 32k is a very safe & fast buffer for your pruning logic


class ClaudeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CLAUDE_", extra="ignore")
    
    # Anthropic API Key (sk-ant-...)
    api_key: SecretStr | None = None
    
    base_url: str = "https://api.anthropic.com"
    
    # ─── LLM TIER CONFIGURATION ───
    # Tier 1: Fast, low-cost model (Claude 3 Haiku)
    tier1_fast_model: str = "claude-3-haiku-20240307"
    
    # Tier 2: Balanced reasoning (Claude 3.5 Sonnet)
    tier2_balanced_model: str = "claude-3-5-sonnet-20240620"
    
    # Tier 3: Highest reasoning (Claude 3 Opus)
    tier3_reasoning_model: str = "claude-3-opus-20240229"
    
    # ─── TOKEN GOVERNANCE ───
    max_completion_tokens: int = 1024
    max_context_tokens: int = 8192

class LlamaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="LLAMA_", extra="ignore")
    api_key: SecretStr | None = None
    base_url: str = "https://models.inference.ai.azure.com"
    
    tier1_fast_model: str = "Llama-3.1-8B-Instruct"
    tier2_balanced_model: str = "Llama-3.1-70B-Instruct"
    tier3_reasoning_model: str = "Llama-3.1-405B-Instruct"
    
    max_completion_tokens: int = 2048
    max_context_tokens: int = 8192

# ─── DEEPSEEK PROVIDER SETTINGS ───
class DeepSeekSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DEEPSEEK_", extra="ignore")
    api_key: SecretStr | None = None
    base_url: str = "https://models.inference.ai.azure.com"
    
    tier1_fast_model: str = "DeepSeek-R1"
    tier2_balanced_model: str = "DeepSeek-R1"
    tier3_reasoning_model: str = "DeepSeek-R1"
    
    max_completion_tokens: int = 4096
    max_context_tokens: int = 16384

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

# MCP configuration for securing tool execution
class McpSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="MCP_", extra="ignore")
    jwt_secret: str = "default_mcp_jwt_secret_key_change_me_in_prod"

# Security configuration for AI response signatures and handshake keys
class SecuritySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SECURITY_", extra="ignore")
    
    # ECDSA SECP256R1 Private Key for signing AI Response Receipts
    ai_receipt_private_key: str = (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgjFtyyJvK2e6LduA+\n"
        "Kj9T1m3noMLAIc2MP3vC1l/8BEahRANCAAR3T7bXX+jXw8E6U2y1toL7zbWINJZy\n"
        "e1Sxr229hOal6CO/mpaLIQZifVAArsmVkvIedjHz3Pstx+f6+4UA4JFs\n"
        "-----END PRIVATE KEY-----"
    )
    
    # ECDSA SECP256R1 Public Key for B2B client verification
    ai_receipt_public_key: str = (
        "-----BEGIN PUBLIC KEY-----\n"
        "MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEd0+211/o18PBOlNstbaC+821iDSW\n"
        "cntUsa9tvYTmpegjv5qWiyEGYn1QAK7JlZLyHnYx89z7Lcfn+vuFAOCRbA==\n"
        "-----END PUBLIC KEY-----"
    )

class Settings(BaseSettings):
    """Unified application configuration manager grouping domain-specific sub-models."""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    claude: ClaudeSettings = Field(default_factory=ClaudeSettings)
    llama: LlamaSettings = Field(default_factory=LlamaSettings)
    deepseek: DeepSeekSettings = Field(default_factory=DeepSeekSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    rabbitmq: RabbitMQSettings = Field(default_factory=RabbitMQSettings)
    logs: LogsSettings = Field(default_factory=LogsSettings)
    tavily: TavilySettings = Field(default_factory=TavilySettings)
    mcp: McpSettings = Field(default_factory=McpSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    
@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()