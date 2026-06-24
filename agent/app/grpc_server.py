# app/grpc_server.py
import asyncio
import grpc
from grpc import aio
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "grpc_layer"))
import chat_pb2
import chat_pb2_grpc
from langchain_core.messages import HumanMessage
from app.services.rabbitmq_publisher import publish_extraction_task
from app.core.logger import setup_app_logger
from app.graph.workflow import compiled_graph
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from app.core.settings import settings
from app.mcp.mcp_client import mcp_manager
from app.bootstrap.startup import startup, shutdown

from prometheus_client import Gauge, Counter
from prometheus_client import start_http_server

logger = setup_app_logger("GrpcServerCore")

GRPC_ACTIVE_STREAMS = Gauge(
    'grpc_active_streams', 
    'Number of concurrent active gRPC chat streams currently processing'
)

GRPC_ERRORS_TOTAL = Counter(
    'grpc_errors_total', 
    'Total number of gRPC runtime pipeline exceptions caught'
)

class AgentServiceServicer(chat_pb2_grpc.AgentServiceServicer):
    def __init__(self):
        self.graph = compiled_graph

    async def StreamChat(self, request: chat_pb2.ChatRequest, context: grpc.aio.ServicerContext):
        logger.info(f"==> [gRPC] Received request from User: {request.user_id}, Session: {request.session_id}")
        
        # 🚀 METRIC: Increment active stream gauge
        GRPC_ACTIVE_STREAMS.inc()

        initial_state = {
            "messages": [HumanMessage(content=request.prompt)],
            "user_id": request.user_id,
            "session_id": request.session_id,
            "current_domain": "general_memory" # Fallback seed value
        }

        trace_config = {
            "metadata": {
                "session_id": request.session_id, 
                "user_id": request.user_id,
                "source": "grpc"
            },
            "configurable": {
                "thread_id": f"{request.user_id}_{request.session_id}" 
            }
        }

        final_state_messages = []
        ai_full_response_text = ""
        resolved_routing_domain = "general_memory"
        
        is_anonymous = request.user_id.startswith("anon_")

        async def run_stream(compiled_graph_instance):
            nonlocal ai_full_response_text, final_state_messages, resolved_routing_domain
            async for event in compiled_graph_instance.astream_events(initial_state, version="v2", config=trace_config):
                if context.cancelled():
                    logger.info("==> [gRPC] Stream dropped by upstream proxy.")
                    break

                kind = event["event"]
                
                if kind == "on_chat_model_stream":
                    current_node = event.get("metadata", {}).get("langgraph_node", "")
                    if current_node == "supervisor_router":
                        continue

                    content = event["data"]["chunk"].content
                    if content:
                        ai_full_response_text += content
                        yield chat_pb2.ChatResponse(chunk=content)
                        
                elif kind == "on_chain_end" and event["name"] == "compiled_graph":
                    output_payload = event["data"]["output"]
                    final_state_messages = output_payload["messages"]
                    # 🚀 Intercept the terminal state domain configuration generated dynamically by the Supervisor
                    resolved_routing_domain = output_payload.get("current_domain", "general_memory")

        try:
            if is_anonymous:
                graph = self.graph.compile()
                async for chunk_response in run_stream(graph):
                    yield chunk_response
            else:
                async with AsyncRedisSaver(redis_url=settings.redis.url) as saver:
                    graph = self.graph.compile(checkpointer=saver)
                    async for chunk_response in run_stream(graph):
                        yield chunk_response

            # ─── ASYNCHRONOUS BACKGROUND LONG-TERM FACT EXTRACTION ORCHESTRATION ───
            if final_state_messages and not is_anonymous:
                # 🚀 Pass the resolved dynamic routing tag down the message pipeline task definition
                asyncio.create_task(
                    publish_extraction_task(
                        request.user_id, 
                        request.session_id, 
                        resolved_routing_domain
                    )
                )

            # ─── SECURE CRYPTOGRAPHIC AI RESPONSE RECEIPT GENERATION (SOLUTION A) ───
            if ai_full_response_text:
                import hashlib
                import time
                from app.core.asymmetric_helper import sign_data_es256

                # Compute SHA-256 hash of the fully accumulated response text
                response_hash = hashlib.sha256(ai_full_response_text.encode('utf-8')).hexdigest()
                timestamp = int(time.time())

                # Data pattern to sign: session_id + timestamp + response_hash
                data_to_sign = f"{request.session_id}:{timestamp}:{response_hash}"
                
                # Sign the data
                signature = sign_data_es256(data_to_sign, settings.security.ai_receipt_private_key)

                # Yield the final message containing the Receipt envelope
                receipt_msg = chat_pb2.Receipt(
                    session_id=request.session_id,
                    timestamp=timestamp,
                    response_hash=response_hash,
                    signature=signature,
                    key_id="secp256r1-default-key"
                )
                yield chat_pb2.ChatResponse(chunk="", receipt=receipt_msg)
                logger.info(f"==> [gRPC Receipt] Successfully generated and yielded AI Response Receipt for Session: {request.session_id}")

        except Exception as e:
            GRPC_ERRORS_TOTAL.inc()
            logger.error(f"==> [gRPC Error] Runtime exception caught in pipeline: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

        finally:
            # METRIC: Decrement stream gauge upon termination
            GRPC_ACTIVE_STREAMS.dec()

async def serve():
    await asyncio.to_thread(start_http_server, 8001)
    logger.info("📊 gRPC Prometheus metrics exporter server listening securely on port 8001\n")

    await startup()
    
    server = aio.server()
    chat_pb2_grpc.add_AgentServiceServicer_to_server(AgentServiceServicer(), server)
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    logger.info(f"🚀 gRPC Core Engine started on {listen_addr}")
    
    logger.info("⚙️ Connecting to external Upstream MCP Servers from gRPC Process...")
    await mcp_manager.initialize_all_servers()

    await server.start()
    await server.wait_for_termination()

if __name__ == '__main__':
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(serve())
