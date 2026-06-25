# app/graph/config.py
from typing import Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from app.core.settings import settings
from app.core.logger import setup_app_logger
from app.mcp.tool_registry import get_tools_by_domain

logger = setup_app_logger("GraphConfig")

# ─── DYNAMIC TIERED LLM INSTANTIATION & CACHING ───
_LLM_INSTANCE_CACHE: Dict[str, BaseChatModel] = {}
_STRUCTURED_LLM_CACHE: Dict[str, Any] = {}

def _get_provider_settings(provider: str):
    """Return the settings object for the given provider name."""
    if provider == "gemini":
        return settings.gemini
    elif provider == "claude":
        return settings.claude
    elif provider == "llama":
        return settings.llama
    elif provider == "deepseek":
        return settings.deepseek
    else:
        return settings.openai

def get_llm_instance(tier: int, config: RunnableConfig = None) -> BaseChatModel:
    """
    Dynamically instantiates and caches the LLM.
    
    From configurable (user-submitted): only `llm_provider` and `api_key`.
    From settings.py (system-configured): model tiers, base_url, max_tokens.
    """
    configurable = config.get("configurable", {}) if config else {}
    
    # 1. Determine Provider (from user config or system default)
    provider = configurable.get("llm_provider")
    if not provider:
        if settings.gemini.api_key:
            provider = "gemini"
        elif settings.claude.api_key:
            provider = "claude"
        elif settings.llama.api_key:
            provider = "llama"
        elif settings.deepseek.api_key:
            provider = "deepseek"
        else:
            provider = "openai"
    provider = provider.lower()
    
    # 2. Get provider settings (all tiers, base_url, max_tokens come from here)
    provider_cfg = _get_provider_settings(provider)
    
    # 3. Retrieve API Key (user-supplied or system default)
    api_key_str = configurable.get("api_key")
    if not api_key_str:
        api_key_str = provider_cfg.api_key.get_secret_value() if provider_cfg.api_key else None
    
    # 4. Resolve model name and token limits from settings (NOT from user input)
    base_url = provider_cfg.base_url
    
    if tier == 1:
        model_name = provider_cfg.tier1_fast_model
    elif tier == 2:
        model_name = provider_cfg.tier2_balanced_model
    else:
        model_name = provider_cfg.tier3_reasoning_model
    
    max_tokens = provider_cfg.max_completion_tokens
    
    # 5. Cache key
    import hashlib
    api_hash = hashlib.sha256(api_key_str.encode("utf-8")).hexdigest() if api_key_str else "none"
    cache_key = f"{provider}_{model_name}_{api_hash}"
    
    if cache_key not in _LLM_INSTANCE_CACHE:
        logger.info(f"==> [LLM Factory] Instantiating {provider} model: {model_name} (Tier {tier})")
        if provider == "gemini":
            if base_url and "openai" in base_url:
                _LLM_INSTANCE_CACHE[cache_key] = ChatOpenAI(
                    model=model_name, api_key=api_key_str,
                    base_url=base_url, max_tokens=max_tokens, streaming=True
                )
            else:
                _LLM_INSTANCE_CACHE[cache_key] = ChatGoogleGenerativeAI(
                    model=model_name, google_api_key=api_key_str,
                    max_tokens=max_tokens, streaming=True
                )
        elif provider == "claude":
            _LLM_INSTANCE_CACHE[cache_key] = ChatAnthropic(
                model=model_name, api_key=api_key_str,
                max_tokens=max_tokens, streaming=True
            )
        else:
            _LLM_INSTANCE_CACHE[cache_key] = ChatOpenAI(
                model=model_name, api_key=api_key_str,
                base_url=base_url, max_tokens=max_tokens, streaming=True
            )
            
    return _LLM_INSTANCE_CACHE[cache_key]

def get_structured_llm(tier: int, schema: Any, config: RunnableConfig = None) -> Any:
    """
    Dynamically creates and caches structured output runnables based on the current
    request context to prevent compilation overhead.
    """
    base_llm = get_llm_instance(tier, config)
    schema_name = getattr(schema, "__name__", str(schema))
    
    # Find the cache key of base_llm in _LLM_INSTANCE_CACHE
    base_key = "default"
    for k, v in _LLM_INSTANCE_CACHE.items():
        if v is base_llm:
            base_key = k
            break
            
    cache_key = f"{base_key}_{schema_name}"
    
    if cache_key not in _STRUCTURED_LLM_CACHE:
        logger.info(f"==> [LLM Factory] Compiling structured output for schema: {schema_name}")
        _STRUCTURED_LLM_CACHE[cache_key] = base_llm.with_structured_output(schema)
        
    return _STRUCTURED_LLM_CACHE[cache_key]

# ─── BOUND LLM CACHING MECHANISM (O(1) OPTIMIZATION) ───
_BOUND_LLM_CACHE: Dict[str, Any] = {}

def get_cached_bound_llm(domain: str, base_llm: BaseChatModel) -> BaseChatModel:
    """
    Retrieves a cached, tool-bound LLM specific to the requested domain.
    If it does not exist, it fetches ONLY the specific tools for that domain in O(1) 
    from the registry, binds them, and caches the resulting runnable.
    """
    model_name = getattr(base_llm, "model_name", None) or getattr(base_llm, "model", "default")
    cache_key = f"{domain}_{model_name}"
    
    if cache_key not in _BOUND_LLM_CACHE:
        logger.info(f"==> [Tool Binder] Compiling and caching tools for domain: {domain} on {model_name}")
        
        # O(1) Retrieval: Registry directly returns the pre-mapped tools for this domain
        domain_tools = get_tools_by_domain(domain)
        
        if domain_tools:
            _BOUND_LLM_CACHE[cache_key] = base_llm.bind_tools(domain_tools)
        else:
            _BOUND_LLM_CACHE[cache_key] = base_llm
            
    return _BOUND_LLM_CACHE[cache_key]

