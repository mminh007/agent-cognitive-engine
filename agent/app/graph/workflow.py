# app/graph/workflow.py
import os
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
#from langgraph.checkpoint.redis.aio import AsyncRedisSaver
#import redis
import hashlib
from app.graph.state import AgentState
from app.graph.nodes import (
    node_supervisor_router, 
    node_general_agent, 
    node_research_paper_agent, 
    node_vision_detection_agent,
    node_planner_agent,
    node_critic_agent,
    node_final_synthesizer,
    node_reflection_agent,
    node_task_manager
)
from app.mcp.mcp_client import get_mcp_tools
from app.core.logger import setup_app_logger
from app.core.metrics import FORCED_TERMINATION_TOTAL, DUPLICATE_TOOL_CALL_TOTAL, GRAPH_ITERATIONS

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
workflow.add_node("critic_agent", node_critic_agent)
workflow.add_node("reflection_agent", node_reflection_agent)
workflow.add_node("task_manager", node_task_manager)
workflow.add_node("final_synthesizer", node_final_synthesizer)

# Enforce entry points through the Intent Supervisor
workflow.set_entry_point("supervisor_router")

# ─── PHASE 1: ROUTING & PLANNING ───
# Supervisor always dictates the objective and delegates to Planner
workflow.add_edge("supervisor_router", "planner")

def route_from_planner(state: AgentState) -> str:
    """Delegates the planned tasks to the appropriate specific execution lane."""
    target_destination = state.get("current_domain", "general_memory")
    if target_destination in ["general_memory", "research_papers", "vision_detection"]:
        return target_destination
    return "general_memory"

workflow.add_conditional_edges(
    "planner",
    route_from_planner,
    {
        "general_memory": "general_memory",
        "research_papers": "research_papers",
        "vision_detection": "vision_detection"
    }
)


# ─── PHASE 2: CONFIGURABLE HARD LIMITS ───
MAX_ITERATIONS = 8        # Prevent infinite Thought/Action loops
MAX_TOOL_CALLS = 10       # Prevent budget drain

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
    domain = state.get("current_domain", "general_memory")
    
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "critic_agent"

    # 1. MAX ITERATION & BUDGET SAFEGUARD
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        logger.warning(f"🛑 [Safeguard Tripped] Session {session_id}: Max loops reached. Forcing termination.\n")
        FORCED_TERMINATION_TOTAL.labels(domain=domain, reason="max_iterations").inc() # 🚀 METRIC
        return "critic_agent"
    
    if state.get("tool_call_count", 0) >= MAX_TOOL_CALLS:
        logger.warning(f"🛑 [Safeguard Tripped] Session {session_id}: Max tools budget reached. Forcing termination.\n")
        FORCED_TERMINATION_TOTAL.labels(domain=domain, reason="budget_depleted").inc() # 🚀 METRIC
        return "critic_agent"

    action_history = state.get("action_history", [])
    
    # 2. DUPLICATE ACTION DETECTION
    for tool_call in last_message.tool_calls:
        t_hash = generate_tool_hash(tool_call['name'], tool_call['args'])
        if t_hash in action_history:
            logger.warning(f"🛑 [Duplicate Detection] Blocked redundant tool call: {tool_call['name']}\n")
            DUPLICATE_TOOL_CALL_TOTAL.labels(tool_name=tool_call['name']).inc() # 🚀 METRIC
            FORCED_TERMINATION_TOTAL.labels(domain=domain, reason="duplicate_action").inc() # 🚀 METRIC
            return "critic_agent"
            
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
executor_map = {"execute_tools": "action_tracker", "critic_agent": "critic_agent"}
workflow.add_conditional_edges("general_memory", evaluate_tool_hooks, executor_map)
workflow.add_conditional_edges("research_papers", evaluate_tool_hooks, executor_map)
workflow.add_conditional_edges("vision_detection", evaluate_tool_hooks, executor_map)

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

# ─── PHASE 3: EVALUATION & SYNTHESIS PIPELINE ───
workflow.add_edge("critic_agent", "reflection_agent")

def route_from_reflection(state: AgentState) -> str:
    if state.get("needs_rework"):
        domain = state.get("current_domain", "general_memory")
        if domain in ["general_memory", "research_papers", "vision_detection"]:
            return domain
        return "general_memory"
    return "task_manager"

workflow.add_conditional_edges(
    "reflection_agent",
    route_from_reflection,
    {
        "general_memory": "general_memory",
        "research_papers": "research_papers",
        "vision_detection": "vision_detection",
        "task_manager": "task_manager"
    }
)

def route_from_task_manager(state: AgentState) -> str:
    if state.get("current_task_id") is not None:
        domain = state.get("current_domain", "general_memory")
        if domain in ["general_memory", "research_papers", "vision_detection"]:
            return domain
        return "general_memory"
    return "final_synthesizer"

workflow.add_conditional_edges(
    "task_manager",
    route_from_task_manager,
    {
        "general_memory": "general_memory",
        "research_papers": "research_papers",
        "vision_detection": "vision_detection",
        "final_synthesizer": "final_synthesizer"
    }
)

workflow.add_edge("final_synthesizer", END)

compiled_graph = workflow.compile(name="compiled_graph") 

