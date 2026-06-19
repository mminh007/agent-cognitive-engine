# app/graph/workflow.py
import os
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
import redis
import hashlib
from app.graph.state import AgentState
from app.graph.nodes import (
    node_supervisor_router, 
    node_general_agent, 
    node_research_paper_agent, 
    node_vision_detection_agent,
    node_planner_agent
)
from app.mcp.mcp_client import get_mcp_tools
from app.core.logger import setup_app_logger
from app.core.settings import settings

logger = setup_app_logger("WorkflowOrchestrator")

# Initialize and configure the central tool interface
mcp_tools = get_mcp_tools()
tool_node = ToolNode(mcp_tools)
workflow = StateGraph(AgentState)

# Register the Master Router along with all domain-isolated specialist nodes
workflow.add_node("supervisor_router", node_supervisor_router)
workflow.add_node("general_memory", node_general_agent)
workflow.add_node("research_papers", node_research_paper_agent)
workflow.add_node("vision_detection", node_vision_detection_agent)
workflow.add_node("planner", node_planner_agent)

# Enforce entry points through the Intent Supervisor
workflow.set_entry_point("supervisor_router")

# ─── PHASE 2: CONFIGURABLE HARD LIMITS ───
MAX_ITERATIONS = 8        # Prevent infinite Thought/Action loops
MAX_TOOL_CALLS = 10       # Prevent budget drain

def conditional_agent_routing(state: AgentState) -> str:
    """Evaluates the calculated intent domain parameter to switch processing lanes."""
    target_destination = state.get("current_domain", "general_memory")
    # 🚀 PHASE 2: Route Research requests to the Planner first
    if target_destination == "research_papers":
        return "planner"
        
    if target_destination in ["general_memory", "vision_detection"]:
        return target_destination
        
    return "general_memory"

# Wire the conditional logic pathways linking the Supervisor node to the specialist agents
workflow.add_conditional_edges(
    "supervisor_router",
    conditional_agent_routing,
    {
        "general_memory": "general_memory",
        "planner": "planner",  # Planner takes over Research lane entry
        "vision_detection": "vision_detection"
    }
)

# 🚀 PHASE 2: Planner strictly transitions to the Research Agent to execute tasks
workflow.add_edge("planner", "research_papers")


def generate_tool_hash(tool_name: str, args: dict) -> str:
    """Creates a unique deterministic hash for a tool call to detect exact duplicates."""
    raw_str = f"{tool_name}_{str(sorted(args.items()))}"
    return hashlib.md5(raw_str.encode()).hexdigest()

def evaluate_tool_hooks(state: AgentState) -> str:
    """
    Detects active tool calls while enforcing strict programmatic safeguards:
    Max Iterations, Budget Limits, and Duplicate Detection.
    """
    last_message = state["messages"][-1]
    session_id = state.get("session_id", "UNKNOWN_SESSION")
    
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return END

    # 1. MAX ITERATION & BUDGET SAFEGUARD
    if state.get("iteration_count", 0) >= MAX_ITERATIONS or state.get("tool_call_count", 0) >= MAX_TOOL_CALLS:
        logger.warning(f"🛑 [Safeguard Tripped] Session {session_id}: Max loops/tools reached. Forcing termination.")
        return END

    action_history = state.get("action_history", [])
    
    # 2. DUPLICATE ACTION DETECTION
    for tool_call in last_message.tool_calls:
        t_hash = generate_tool_hash(tool_call['name'], tool_call['args'])
        if t_hash in action_history:
            logger.warning(f"🛑 [Duplicate Detection] Blocked redundant tool call: {tool_call['name']}")
            return END
            
    tool_names = [t['name'] for t in last_message.tool_calls]
    logger.info(f"==> [Tool Action Dispatched] Session: {session_id} -> Executing: {tool_names}")
    return "execute_tools"

async def action_tracker_node(state: AgentState):
    """
    Interceptor node that updates workflow memory (action history & counters) 
    BEFORE executing the actual tools.
    """
    last_msg = state["messages"][-1]
    action_history = state.get("action_history", [])
    tool_call_count = state.get("tool_call_count", 0)
    
    new_hashes = []
    if hasattr(last_msg, "tool_calls"):
        for tc in last_msg.tool_calls:
            new_hashes.append(generate_tool_hash(tc['name'], tc['args']))
            tool_call_count += 1
            
    return {
        "action_history": action_history + new_hashes,
        "tool_call_count": tool_call_count
    }

workflow.add_node("action_tracker", action_tracker_node)
workflow.add_node("tools", tool_node)

# Bind tool validation endpoints across all worker components
workflow.add_conditional_edges("general_memory", evaluate_tool_hooks, {"execute_tools": "action_tracker", END: END})
workflow.add_conditional_edges("research_papers", evaluate_tool_hooks, {"execute_tools": "action_tracker", END: END})
workflow.add_conditional_edges("vision_detection", evaluate_tool_hooks, {"execute_tools": "action_tracker", END: END})

# Link tracker directly to the actual LangChain ToolNode
workflow.add_edge("action_tracker", "tools")

def route_back_to_agent(state: AgentState) -> str:
    """
    Reads the active domain from the state context to guide 
    the ToolMessage payload cleanly back into the issuing Agent.
    """
    target = state.get("current_domain", "general_memory")
    if target in ["general_memory", "research_papers", "vision_detection"]:
        return target
    return "general_memory"

workflow.add_conditional_edges(
    "tools",
    route_back_to_agent,
    {
        "general_memory": "general_memory",
        "research_papers": "research_papers",
        "vision_detection": "vision_detection"
    }
)

compiled_graph = workflow.compile(name="compiled_graph") 

