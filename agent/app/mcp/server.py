# app/mcp/server.py
import sys
import os
from mcp.server.fastmcp import FastMCP

# Ensure absolute imports resolve correctly across the project structure
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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