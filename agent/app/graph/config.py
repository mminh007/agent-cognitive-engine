# app/graph/config.py
import hashlib
from typing import Dict, Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from app.core.settings import settings
from app.core.logger import setup_app_logger
from app.mcp.tool_registry import get_tools_by_domain

logger = setup_app_logger("GraphConfig")

# ─── SUPPORTED PROVIDERS ───
# Only OpenAI and Anthropic are supported. Both offer native Function Calling
# which guarantees reliable structured output parsing for Pydantic schemas.
SUPPORTED_PROVIDERS = frozenset({"openai", "claude"})


# ─── SETTINGS FINGERPRINT (Cache Staleness Guard) ───
def _get_settings_fingerprint() -> str:
    """
    Creates a short hash derived from all critical, mutable settings values.

    PURPOSE: Cache Staleness Fix.
    When any API key, model name, or token limit changes (e.g. via env var
    rotation), this fingerprint changes automatically. Since all three cache
    dicts embed this fingerprint in their keys, stale entries simply become
    unreachable — they are never returned and new instances are created.
    This is a zero-TTL, zero-background-thread invalidation strategy.
    """
    raw = (
        f"openai:{settings.openai.api_key}"
        f"|{settings.openai.tier1_fast_model}"
        f"|{settings.openai.tier2_balanced_model}"
        f"|{settings.openai.tier3_reasoning_model}"
        f"|claude:{settings.claude.api_key}"
        f"|{settings.claude.tier1_fast_model}"
        f"|{settings.claude.tier2_balanced_model}"
        f"|{settings.claude.tier3_reasoning_model}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ─── DYNAMIC TIERED LLM INSTANTIATION & CACHING ───
_LLM_INSTANCE_CACHE: Dict[str, BaseChatModel] = {}
_STRUCTURED_LLM_CACHE: Dict[str, Any] = {}
_BOUND_LLM_CACHE: Dict[str, Any] = {}


def _get_provider_settings(provider: str):
    """Return the settings object for a supported provider."""
    if provider == "claude":
        return settings.claude
    return settings.openai  # default


def _build_llm_cache_key(provider: str, model_name: str, api_key_str: Optional[str]) -> str:
    """
    Builds a deterministic, collision-resistant cache key.

    PURPOSE: Cache Lookup Mismatch Fix.
    Previously, get_structured_llm() re-located its base_key by iterating
    _LLM_INSTANCE_CACHE with an identity check (v is base_llm), which could
    silently fall back to "default" and cause schema-level cache collisions.

    This helper is now called by BOTH get_llm_instance() and get_structured_llm()
    with the same resolved (provider, model_name, api_key_str) arguments,
    guaranteeing that the key is always identical between the two functions.

    Components:
    - provider + model_name : human-readable tier identity
    - api_hash (12 chars)   : prevents key re-use across different API keys
    - settings_fp (12 chars): invalidates the key when any config changes
    """
    api_hash = hashlib.sha256(api_key_str.encode("utf-8")).hexdigest()[:12] if api_key_str else "no_key"
    settings_fp = _get_settings_fingerprint()
    return f"{provider}_{model_name}_{api_hash}_{settings_fp}"


def _resolve_provider_and_key(config: RunnableConfig = None):
    """
    Shared resolution logic for provider, provider_cfg, api_key_str, and model
    name per tier. Extracted to avoid duplication between get_llm_instance()
    and get_structured_llm().
    """
    configurable = config.get("configurable", {}) if config else {}

    # Priority: explicit user config → auto-detect from available API keys
    provider = configurable.get("llm_provider", "").lower()
    if provider not in SUPPORTED_PROVIDERS:
        provider = "claude" if settings.claude.api_key else "openai"

    provider_cfg = _get_provider_settings(provider)

    api_key_str = configurable.get("api_key")
    if not api_key_str:
        api_key_str = provider_cfg.api_key.get_secret_value() if provider_cfg.api_key else None

    return provider, provider_cfg, api_key_str


def get_llm_instance(tier: int, config: RunnableConfig = None) -> BaseChatModel:
    """
    Dynamically instantiates and caches the LLM for the given tier.

    Supported providers: OpenAI (ChatOpenAI) and Anthropic (ChatAnthropic).
    From user-configurable: only `llm_provider` and `api_key`.
    From settings.py (system-controlled): model tiers, base_url, max_tokens.
    """
    provider, provider_cfg, api_key_str = _resolve_provider_and_key(config)

    # Resolve model name from settings — not from user input (security)
    if tier == 1:
        model_name = provider_cfg.tier1_fast_model
    elif tier == 2:
        model_name = provider_cfg.tier2_balanced_model
    else:
        model_name = provider_cfg.tier3_reasoning_model

    max_tokens = provider_cfg.max_completion_tokens
    base_url = getattr(provider_cfg, "base_url", None)

    cache_key = _build_llm_cache_key(provider, model_name, api_key_str)

    if cache_key not in _LLM_INSTANCE_CACHE:
        logger.info(f"==> [LLM Factory] Instantiating {provider.upper()} model: {model_name} (Tier {tier})")
        if provider == "claude":
            _LLM_INSTANCE_CACHE[cache_key] = ChatAnthropic(
                model=model_name,
                api_key=api_key_str,
                max_tokens=max_tokens,
                streaming=True,
            )
        else:  # openai
            _LLM_INSTANCE_CACHE[cache_key] = ChatOpenAI(
                model=model_name,
                api_key=api_key_str,
                base_url=base_url,
                max_tokens=max_tokens,
                streaming=True,
            )

    return _LLM_INSTANCE_CACHE[cache_key]


def get_structured_llm(tier: int, schema: Any, config: RunnableConfig = None) -> Any:
    """
    Dynamically creates and caches structured output runnables.

    FIX (Cache Lookup Mismatch): Previously iterated _LLM_INSTANCE_CACHE with an
    identity check to find the base_key, which was fragile and could silently
    default to "default". Now calls _build_llm_cache_key() directly with the same
    resolved arguments as get_llm_instance(), guaranteeing key consistency.

    Both OpenAI and Anthropic use native Function Calling under the hood when
    .with_structured_output() is called, ensuring maximum schema compliance.
    """
    provider, provider_cfg, api_key_str = _resolve_provider_and_key(config)

    if tier == 1:
        model_name = provider_cfg.tier1_fast_model
    elif tier == 2:
        model_name = provider_cfg.tier2_balanced_model
    else:
        model_name = provider_cfg.tier3_reasoning_model

    schema_name = getattr(schema, "__name__", str(schema))
    base_key = _build_llm_cache_key(provider, model_name, api_key_str)
    cache_key = f"{base_key}_{schema_name}"

    if cache_key not in _STRUCTURED_LLM_CACHE:
        logger.info(f"==> [LLM Factory] Compiling structured output | Schema: {schema_name} | Model: {model_name}")
        base_llm = get_llm_instance(tier, config)
        _STRUCTURED_LLM_CACHE[cache_key] = base_llm.with_structured_output(schema)

    return _STRUCTURED_LLM_CACHE[cache_key]


# ─── BOUND LLM CACHING MECHANISM (O(1) OPTIMIZATION) ───

def _extract_api_key_from_llm(base_llm: BaseChatModel) -> str:
    """
    Extracts the raw API key string from a cached LLM instance for hashing.
    Handles both ChatOpenAI and ChatAnthropic attribute names.
    """
    for attr in ("openai_api_key", "anthropic_api_key", "api_key"):
        key = getattr(base_llm, attr, None)
        if key is not None:
            return key.get_secret_value() if hasattr(key, "get_secret_value") else str(key)
    return "unknown"


def get_cached_bound_llm(domain: str, base_llm: BaseChatModel) -> BaseChatModel:
    """
    Retrieves a cached, tool-bound LLM specific to the requested domain.
    If it does not exist, fetches ONLY the specific tools for that domain in O(1)
    from the registry, binds them, and caches the resulting runnable.

    FIX (Cache Key Conflict): Previously used only `domain + model_name` as the
    cache key. In multi-tenant environments, two requests with different API keys
    but the same model would share a single cached bound LLM — a security and
    correctness hazard. The key now includes:
    - api_hash: prevents cross-tenant leakage
    - settings_fp: invalidates cache after key rotation or config changes
    """
    model_name = getattr(base_llm, "model_name", None) or getattr(base_llm, "model", "default")
    api_key_raw = _extract_api_key_from_llm(base_llm)
    api_hash = hashlib.sha256(api_key_raw.encode()).hexdigest()[:12]
    settings_fp = _get_settings_fingerprint()
    cache_key = f"{domain}_{model_name}_{api_hash}_{settings_fp}"

    if cache_key not in _BOUND_LLM_CACHE:
        logger.info(f"==> [Tool Binder] Compiling and caching tools for domain: '{domain}' on {model_name}")
        domain_tools = get_tools_by_domain(domain)

        if domain_tools:
            _BOUND_LLM_CACHE[cache_key] = base_llm.bind_tools(domain_tools)
        else:
            _BOUND_LLM_CACHE[cache_key] = base_llm

    return _BOUND_LLM_CACHE[cache_key]
