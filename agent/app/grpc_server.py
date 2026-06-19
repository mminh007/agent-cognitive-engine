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
from app.api.deps import get_agent_graph
from app.services.rabbitmq_publisher import publish_extraction_task
from app.core.logger import setup_app_logger
from app.graph.workflow import compiled_graph
from app.mcp.mcp_client import mcp_manager

logger = setup_app_logger("GrpcServerCore")

class AgentServiceServicer(chat_pb2_grpc.AgentServiceServicer):
    def __init__(self):
        self.graph = compiled_graph

    async def StreamChat(self, request: chat_pb2.ChatRequest, context: grpc.aio.ServicerContext):
        logger.info(f"==> [gRPC] Received request from User: {request.user_id}, Session: {request.session_id}")
        
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
        
        try:
            async for event in self.graph.astream_events(initial_state, version="v2", config=trace_config):
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

            # ─── ASYNCHRONOUS BACKGROUND LONG-TERM FACT EXTRACTION ORCHESTRATION ───
            if final_state_messages:
                # 🚀 Pass the resolved dynamic routing tag down the message pipeline task definition
                asyncio.create_task(
                    publish_extraction_task(
                        request.user_id, 
                        request.session_id, 
                        resolved_routing_domain, 
                        final_state_messages
                    )
                )

        except Exception as e:
            logger.error(f"==> [gRPC Error] Runtime exception caught in pipeline: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

async def serve():
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
