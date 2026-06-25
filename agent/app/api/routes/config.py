# app/api/routes/config.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
from app.bootstrap.container import container
import json

router = APIRouter(prefix="/config", tags=["Configuration"])

class ProviderConfigSubmit(BaseModel):
    """
    User submits their preferred provider config.
    
    Case 1 - Custom key:   { llm_provider: "gemini", api_key: "ab.AQxxxx", default: None }
    Case 2 - Use default:  { llm_provider: None, api_key: None, default: "gemini" }
    """
    user_id: str
    llm_provider: Optional[Literal["openai", "gemini", "claude", "llama", "deepseek"]] = None
    api_key: Optional[str] = None
    default: Optional[Literal["openai", "gemini", "claude", "llama", "deepseek"]] = None

@router.post("/provider")
async def submit_provider_config(config: ProviderConfigSubmit):
    if not container.redis_client:
        raise HTTPException(status_code=500, detail="Redis client not initialized.")
    
    # Validate: user must provide either custom key OR default, not both
    if config.llm_provider and config.default:
        raise HTTPException(status_code=400, detail="Cannot set both 'llm_provider' (custom key) and 'default' at the same time.")
    
    if config.llm_provider and not config.api_key:
        raise HTTPException(status_code=400, detail="Custom provider requires an 'api_key'.")
    
    # Build minimal config to store
    config_dict = {"user_id": config.user_id}
    
    if config.default:
        # User wants to use system default key for this provider
        config_dict["llm_provider"] = config.default
        config_dict["use_default_key"] = True
    elif config.llm_provider:
        # User provides their own key
        config_dict["llm_provider"] = config.llm_provider
        config_dict["api_key"] = config.api_key
        config_dict["use_default_key"] = False
    
    # TTL: 24 hours (86400 seconds)
    await container.redis_client.setex(
        f"user_config:{config.user_id}", 
        86400, 
        json.dumps(config_dict)
    )
    
    return {"status": "success", "message": f"Configuration saved for user {config.user_id}"}
