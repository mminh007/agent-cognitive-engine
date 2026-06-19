# app/mcp/tool_registry.py
from typing import List
from langchain_core.tools import BaseTool
from app.mcp.mcp_client import get_mcp_tools

# Define tool boundaries for each agent domain.
# When the system grows to 100+ tools, only this registry needs to be updated.
AGENT_TOOL_REGISTRY = {
    "general_memory": [
        "calculate_execution_time",
        "search_web"
    ],
    "research_papers": [
        "search_web"
        # Future additions: "arxiv_search", "read_pdf"
    ],
    "vision_detection": [
        # Future additions: "ocr_extract", "analyze_bounding_box"
    ]
}

def get_tools_by_domain(domain: str) -> List[BaseTool]:
    """
    Optimized O(1) tool lookup path. Resolves registered tools for a given 
    domain against the master list of cached MCP tools.
    
    This fulfills the exact import signature required by app/graph/nodes.py.
    """
    # Fetch the master pre-loaded tools list from the client manager cache
    all_mcp_tools = get_mcp_tools()
    allowed_tool_names = AGENT_TOOL_REGISTRY.get(domain, [])

    # Map names directly to base tool schemas
    return [
        tool for tool in all_mcp_tools
        if tool.name in allowed_tool_names
    ]


def get_tools_for_agent(
    domain: str,
    all_available_tools: List[BaseTool]
) -> List[BaseTool]:
    """
    Filters the available MCP tools based on the active agent domain.

    If the domain is not registered, an empty list is returned
    to prevent unnecessary token consumption and tool exposure.
    """
    allowed_tool_names = AGENT_TOOL_REGISTRY.get(domain, [])

    filtered_tools = [
        tool for tool in all_available_tools
        if tool.name in allowed_tool_names
    ]

    return filtered_tools