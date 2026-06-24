# app/api/routes/chat.py
from app.services.rabbitmq_publisher import publish_extraction_task
from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import app.core.logger as logger
from app.bootstrap.container import container
from app.graph.workflow import compiled_graph
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from app.core.settings import settings
# Import Langfuse and LangSmith tracing utilities
from langfuse.langchain import CallbackHandler
from langchain_core.tracers import LangChainTracer
from app.core.metrics import SSE_ACTIVE_STREAMS, SSE_DISCONNECT_TOTAL

router = APIRouter(prefix="/chat", tags=["Agent Chat Ecosystem"])

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    prompt: str

@router.post("/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    # Check semantic cache inside Vector DB to bypass LLM if hit
    cached_reply = await container.semantic_cache.get(request.prompt)
    if cached_reply:
        async def cached_generator():
            yield f"data: {cached_reply}\n\n"
        return StreamingResponse(cached_generator(), media_type="text-event-stream")
    
    initial_state = {
        "messages": [HumanMessage(content=request.prompt)],
        "user_id": request.user_id,
        "session_id": request.session_id,
        "current_domain": "general_memory"
    }
    
    # 🚀 OPTIMIZATION 1: Pass session_id and user_id directly to Langfuse
    langfuse_handler = CallbackHandler(
        session_id=request.session_id,
        user_id=request.user_id,
        tags=["prod-stream"]
    )
    
    langsmith_tracer = LangChainTracer(project_name="agent-ecosystem-prod")

    # 🚀 OPTIMIZATION & FINOPS TRACKING:
    # Inject business classification metadata directly into the root trace span.
    # Langfuse will automatically aggregate tokens/cost grouped by these keys.
    trace_config = {
        "callbacks": [langfuse_handler, langsmith_tracer],
        "metadata": {
            "session_id": request.session_id, 
            "user_id": request.user_id,
            "business_process_codename": "Realtime_Chat_Resolution", # 🚀 For Cost per Business Transaction
            "client_tier": "Standard" 
        },
        "configurable": {
            "thread_id": f"{request.user_id}_{request.session_id}"
        }
    }

    SSE_ACTIVE_STREAMS.inc()

    is_anonymous = request.user_id.startswith("anon_")

    async def event_generator():
        final_state_messages = []
        ai_full_response_text = ""
        resolved_domain = "general_memory"
        stream_completed_cleanly = False
        
        async def run_graph_stream(graph_instance):
            nonlocal ai_full_response_text, final_state_messages, resolved_domain
            async for event in graph_instance.astream_events(initial_state, version="v2", config=trace_config):
                kind = event["event"]
                
                if kind == "on_chat_model_stream":
                    current_node = event.get("metadata", {}).get("langgraph_node", "")
                    if current_node == "supervisor_router":
                        continue
                    
                    content = event["data"]["chunk"].content
                    if content:
                        ai_full_response_text += content
                        yield f"data: {content}\n\n"
                        
                elif kind == "on_chain_end" and event["name"] == "compiled_graph":
                    output_payload = event["data"]["output"]
                    final_state_messages = output_payload["messages"]
                    resolved_domain = output_payload.get("current_domain", "general_memory")

        try:
            if is_anonymous:
                graph_run = compiled_graph.compile()
                async for chunk in run_graph_stream(graph_run):
                    yield chunk
            else:
                async with AsyncRedisSaver(redis_url=settings.redis.url) as saver:
                    graph_run = compiled_graph.compile(checkpointer=saver)
                    async for chunk in run_graph_stream(graph_run):
                        yield chunk
            
            # ─── SECURE CRYPTOGRAPHIC AI RESPONSE RECEIPT GENERATION ───
            if ai_full_response_text:
                import hashlib
                import time
                import json
                from app.core.asymmetric_helper import sign_data_es256

                response_hash = hashlib.sha256(ai_full_response_text.encode('utf-8')).hexdigest()
                timestamp = int(time.time())
                data_to_sign = f"{request.session_id}:{timestamp}:{response_hash}"
                
                signature = sign_data_es256(data_to_sign, settings.security.ai_receipt_private_key)
                
                receipt_json = {
                    "type": "receipt",
                    "session_id": request.session_id,
                    "timestamp": timestamp,
                    "response_hash": response_hash,
                    "signature": signature,
                    "key_id": "secp256r1-default-key"
                }
                yield f"data: {json.dumps(receipt_json)}\n\n"

            stream_completed_cleanly = True    
        finally:
            # 🚀 METRIC: Decrement active streams and evaluate connection termination health
            SSE_ACTIVE_STREAMS.dec()
            if stream_completed_cleanly:
                SSE_DISCONNECT_TOTAL.labels(reason="completed").inc()
            else:
                SSE_DISCONNECT_TOTAL.labels(reason="abrupt_client_disconnect").inc()

            # 🚀 OPTIMIZATION 2: Ensure all Langfuse traces are flushed to the server,
            # even upon successful stream completion or abrupt client disconnection.
            langfuse_handler.flush()

        # Cache the completed response text if available
        if ai_full_response_text:
            await container.semantic_cache.set(request.prompt, ai_full_response_text)

        # Offload post-processing/extraction tasks asynchronously via RabbitMQ
        if final_state_messages and not is_anonymous:
            background_tasks.add_task(
                publish_extraction_task,
                request.user_id, 
                request.session_id, 
                resolved_domain
            )

    return StreamingResponse(event_generator(), media_type="text-event-stream")