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
    node_direct_executor_init,
    node_critic_agent,
    node_final_synthesizer,
    node_reflection_agent,
    node_task_manager,
    ExecutorOutput
)
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
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
workflow.add_node("direct_executor_init", node_direct_executor_init)  # Low-complexity bypass
workflow.add_node("critic_agent", node_critic_agent)
workflow.add_node("reflection_agent", node_reflection_agent)
workflow.add_node("task_manager", node_task_manager)
workflow.add_node("final_synthesizer", node_final_synthesizer)

async def node_self_correct(state: AgentState):
    """Injects a self-correction prompt when confidence is too low."""
    logger.info("==> [Self-Correct] Confidence < 0.6. Forcing executor to rethink.")
    return {"messages": [HumanMessage(content="Your last output had low confidence (< 0.6). Please use tools to gather more solid evidence before answering.")]}

workflow.add_node("self_correct", node_self_correct)

# Enforce entry points through the Intent Supervisor
workflow.set_entry_point("supervisor_router")

# ─── PHASE 1: COMPLEXITY-AWARE ROUTING ───
# Low complexity: bypass Planner → direct_executor_init → domain executor
# Medium/High complexity: Supervisor → Planner → domain executor
def route_from_supervisor(state: AgentState) -> str:
    """
    Complexity-aware routing gate after Supervisor.
    'low' complexity skips Planner entirely to save LLM cost.
    'medium' / 'high' route through full Planner decomposition.
    """
    complexity = state.get("complexity", "medium")
    if complexity == "low":
        logger.info(f"==> [Router] Complexity=LOW — bypassing Planner, routing to direct_executor_init.")
        return "direct_executor_init"
    logger.info(f"==> [Router] Complexity={complexity.upper()} — routing to Planner for decomposition.")
    return "planner"

workflow.add_conditional_edges(
    "supervisor_router",
    route_from_supervisor,
    {
        "direct_executor_init": "direct_executor_init",
        "planner": "planner"
    }
)

def route_from_planner(state: AgentState) -> str:
    """Delegates the planned tasks to the appropriate specific execution lane."""
    tasks = state.get("tasks", [])
    if tasks:
        target_destination = tasks[0].get("target_agent", state.get("current_domain", "general_memory"))
    else:
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

# direct_executor_init routes to the same domain lanes as Planner
workflow.add_conditional_edges(
    "direct_executor_init",
    route_from_planner,   # Reuse same domain-routing logic
    {
        "general_memory": "general_memory",
        "research_papers": "research_papers",
        "vision_detection": "vision_detection"
    }
)


# ─── PHASE 2: CONFIGURABLE HARD LIMITS ───
MAX_ITERATIONS   = 8    # Prevent infinite Thought/Action loops within a single task
MAX_TOOL_CALLS   = 10   # Prevent budget drain across the execution lifecycle
MAX_REWORK_CYCLES = 2   # Cap Critic→Reflection→Executor rework loops per task

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
        try:
            parser = JsonOutputParser(pydantic_object=ExecutorOutput)
            parsed_payload = parser.parse(last_message.content)
            findings = parsed_payload.get("findings", [])
            if findings:
                total_conf = sum(f.get("confidence", 0.0) if isinstance(f, dict) else getattr(f, "confidence", 0.0) for f in findings)
                avg_conf = total_conf / len(findings)
                if avg_conf < 0.6:
                    logger.warning(f"🛑 [Fast-Fail] Session {session_id}: Avg Confidence {avg_conf:.2f} < 0.6. Self-Correcting without Critic.")
                    return "self_correct"
        except Exception as e:
            pass # Let critic handle formatting or empty findings errors
            
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
executor_map = {"execute_tools": "action_tracker", "critic_agent": "critic_agent", "self_correct": "self_correct"}
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

workflow.add_conditional_edges(
    "self_correct",
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
    """
    Bounded rework gate: if Reflection demands rework but the rework counter
    has already hit MAX_REWORK_CYCLES, force-advance to task_manager instead
    of looping back to the executor indefinitely.
    """
    if state.get("needs_rework"):
        rework_count = state.get("rework_count", 0)
        if rework_count >= MAX_REWORK_CYCLES:
            domain = state.get("current_domain", "general_memory")
            logger.warning(
                f"🛑 [Rework Safeguard] Max rework cycles ({MAX_REWORK_CYCLES}) reached "
                f"for domain '{domain}'. Force-advancing to task_manager."
            )
            FORCED_TERMINATION_TOTAL.labels(domain=domain, reason="max_rework_cycles").inc()
            return "task_manager"
        
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
    current_task_id = state.get("current_task_id")
    if current_task_id is not None:
        tasks = state.get("tasks", [])
        for t in tasks:
            if t.get("id") == current_task_id:
                domain = t.get("target_agent", state.get("current_domain", "general_memory"))
                if domain in ["general_memory", "research_papers", "vision_detection"]:
                    return domain
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

