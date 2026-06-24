# app/mcp/server.py
import sys
import os
# Ensure absolute imports resolve correctly across the project structure
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 🚀 Security Handshake Validation Layer (Solution A: HS256 JWT validation)
from app.core.jwt_helper import verify_jwt
client_token = os.environ.get("MCP_CLIENT_TOKEN")
jwt_secret = os.environ.get("MCP_JWT_SECRET") or "default_mcp_jwt_secret_key_change_me_in_prod"

if not client_token:
    print("🚨 [Security Alert] Access Denied: Missing client authentication token.", file=sys.stderr)
    sys.exit(1)

payload = verify_jwt(client_token, jwt_secret)
if not payload or payload.get("sub") != "mcp-client":
    print("🚨 [Security Alert] Access Denied: Invalid or expired client authentication token.", file=sys.stderr)
    sys.exit(1)

from mcp.server.fastmcp import FastMCP

from app.mcp.domains.core_tools import calculate_execution_time_logic
from app.mcp.domains.web_tools import search_web_logic

# Initialize the FastMCP service broker
mcp_server = FastMCP("Unified Domain Action Gateway")

@mcp_server.tool()
def calculate_execution_time(milliseconds: int) -> str:
    """Converts a system execution runtime duration from milliseconds into a highly readable format consisting of minutes and seconds."""
    return calculate_execution_time_logic(milliseconds)

@mcp_server.tool()
def search_web(query: str) -> str:
    """Executes a live search query against open web indexes to retrieve real-time data or verify facts."""
    return search_web_logic(query)

if __name__ == "__main__":
    # Mount the server using the standard input/output transport channel
    mcp_server.run(transport="stdio")