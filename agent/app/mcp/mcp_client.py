# agent/app/mcp/mcp_client.py
import os
import json
import asyncio
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from app.core.logger import setup_app_logger

logger = setup_app_logger("McpRuntimeGateway")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "mcp_servers_config.json")

class DynamicMcpClientManager:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self.langchain_tools = []
        self._lock = asyncio.Lock()
        self._is_initialized = False

    async def initialize_all_servers(self):
        """
        Reads configuration, launches MCP sub-processes via Stdio,
        and seamlessly loads them into the LangChain Tools standard via load_mcp_tools.
        """
        async with self._lock:
            if self._is_initialized:
                return self.langchain_tools

            if not os.path.exists(CONFIG_PATH):
                logger.warning(f"MCP Servers config missing at {CONFIG_PATH}. Emitting empty tool list.")
                return []

            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    servers_config = json.load(f)
            except Exception as e:
                logger.error(f"❌ Failed to parse MCP config JSON: {e}")
                return []

            all_converted_tools = []

            for config in servers_config:
                server_name = config.get("name")
                command = config.get("command")
                args = config.get("args", [])
                env = os.environ.copy()
                if config.get("env"):
                    env.update(config.get("env"))

                logger.info(f"==> [MCP Client] Connecting to external server: '{server_name}' via stdio...")
                
                try:
                    server_params = StdioServerParameters(
                        command=command, args=args, env=env
                    )
                    
                    # 1. Boot up the background Stdio stream
                    read_stream, write_stream = await self.exit_stack.enter_async_context(
                        stdio_client(server_params)
                    )
                    
                    # 2. Initialize standard MCP Client Session and execute Handshake
                    session = await self.exit_stack.enter_async_context(
                        ClientSession(read_stream, write_stream)
                    )
                    await session.initialize()
                    
                    self.sessions[server_name] = session
                    logger.info(f"==> [MCP Client] Handshake OK with server: '{server_name}'")

                    # 3. Load tools directly using the async API from version 0.3.0
                    langchain_tools = await load_mcp_tools(session)
                    all_converted_tools.extend(langchain_tools)
                    
                    logger.info(f" Successfully loaded {len(langchain_tools)} tools from '{server_name}'")

                except Exception as server_err:
                    logger.error(f"❌ [MCP Client Connect Failure] Server '{server_name}' failed: {server_err}")

            self.langchain_tools = all_converted_tools
            self._is_initialized = True
            return self.langchain_tools

    async def shutdown(self):
        """Safely closes connections upon container termination."""
        logger.info("🛑 Closing all upstream MCP Stdio Server connections...")
        await self.exit_stack.aclose()


mcp_manager = DynamicMcpClientManager()

def get_mcp_tools() -> list:
    """Returns the cached list of pre-loaded tools."""
    return mcp_manager.langchain_tools