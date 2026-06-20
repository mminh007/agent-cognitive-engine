# app/graph/nodes.py
import os
import tiktoken
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.graph.state import AgentState
from app.services.vector_db import vector_db_service
from app.services.query_transformer import transform_user_query

# We now import a highly optimized O(1) tool retrieval function
from app.mcp.tool_registry import get_tools_by_domain 
from app.core.settings import settings
from app.core.logger import setup_app_logger

logger = setup_app_logger("CognitiveNodes")

# ─── TIERED LLM INSTANTIATION ───
base_llm_kwargs = {
    "api_key": settings.openai.api_key.get_secret_value() if settings.openai.api_key else None,
    "base_url": settings.openai.base_url,
    "streaming": True
}

llm_tier1_fast = ChatOpenAI(model=settings.openai.tier1_fast_model, **base_llm_kwargs)

llm_tier2_balanced = ChatOpenAI(
    model=settings.openai.tier2_balanced_model, 
    max_tokens=settings.openai.max_completion_tokens,
    **base_llm_kwargs
)

llm_tier3_reasoning = ChatOpenAI(
    model=settings.openai.tier3_reasoning_model,
    max_tokens=settings.openai.max_completion_tokens,
    **base_llm_kwargs
)

# ─── BOUND LLM CACHING MECHANISM (O(1) OPTIMIZATION) ───
# Caches LLM instances that have already had their specific tools bound to them.
# Prevents O(N) tool filtering and costly .bind_tools() schema re-compilation per request.
_BOUND_LLM_CACHE: Dict[str, Any] = {}

def get_cached_bound_llm(domain: str, base_llm: ChatOpenAI) -> ChatOpenAI:
    """
    Retrieves a cached, tool-bound LLM specific to the requested domain.
    If it does not exist, it fetches ONLY the specific tools for that domain in O(1) 
    from the registry, binds them, and caches the resulting runnable.
    """
    # Create a unique cache key based on the domain and the specific LLM tier being used
    cache_key = f"{domain}_{base_llm.model_name}"
    
    if cache_key not in _BOUND_LLM_CACHE:
        logger.info(f"==> [Tool Binder] Compiling and caching tools for domain: {domain} on {base_llm.model_name}")
        
        # O(1) Retrieval: Registry directly returns the pre-mapped tools for this domain
        domain_tools = get_tools_by_domain(domain)
        
        if domain_tools:
            _BOUND_LLM_CACHE[cache_key] = base_llm.bind_tools(domain_tools)
        else:
            _BOUND_LLM_CACHE[cache_key] = base_llm
            
    return _BOUND_LLM_CACHE[cache_key]


# ─── STRUCTURED OUTPUT SCHEMAS ───
class SupervisorRouterOutput(BaseModel):
    target_agent: str = Field(description="The target domain: 'general_memory', 'research_papers', 'vision_detection', or 'coding_agent'.")
    complexity: str = Field(description="Level of complexity: 'low', 'medium', 'high'.")
    required_agents: List[str] = Field(description="List of agents needed to complete the tasks.")
    objective: str = Field(description="The overarching execution objective for the Planner.")
    rationale: str = Field(description="Internal chain-of-thought justification.")

class ComplexityAssessment(BaseModel):
    requires_heavy_reasoning: bool = Field(description="Set to true if the request requires intense mathematics, complex transformer formulations, or o1-level chain of thought.")
    rationale: str = Field(description="Brief reason for this complexity assignment.")

class TaskItem(BaseModel):
    id: int = Field(description="Unique incremental ID for the task.")
    description: str = Field(description="Clear, actionable description of the sub-task.")

class PlannerOutput(BaseModel):
    tasks: List[TaskItem] = Field(description="List of sequential tasks required to fulfill the user's research request.")

class CriticOutput(BaseModel):
    objective_met: bool = Field(description="Assess if the executor fully achieved the initial objective.")
    feedback: str = Field(description="Detailed feedback: Point out flaws, missing data, or provide positive reinforcement if optimal.")

class ReflectionOutput(BaseModel):
    needs_rework: bool = Field(description="Determine if the executor needs to rerun based on the critic's severity.")
    actionable_advice: str = Field(description="Strict, actionable instructions for the executor or synthesis notes if passing.")
    
structured_planner_llm = llm_tier2_balanced.with_structured_output(PlannerOutput)

structured_router_llm = llm_tier1_fast.with_structured_output(SupervisorRouterOutput)


# ─── UTILITY FUNCTIONS ───
def prune_messages_by_token_limit(messages: list, max_tokens: int, model_name: str) -> list:
    """Ensures message history strictly respects context window boundaries."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
        
    total_tokens = 0
    keep_messages = []
    
    for msg in reversed(messages):
        msg_tokens = len(encoding.encode(msg.content))
        if total_tokens + msg_tokens > max_tokens:
            break
        keep_messages.insert(0, msg)
        total_tokens += msg_tokens
        
    return keep_messages

def load_agent_manifest_instructions(file_path: str = "AGENTS.md") -> str:
    """Extracts baseline operational principles from the external Markdown manifest."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return "You are a highly capable engineering AI assistant."


# ─── GRAPH NODES ───

async def node_supervisor_router(state: AgentState):
    """
    Node 0: Supervisor. Identifies the intent and routes to the correct isolated domain lane.
    """
    user_latest_message = state["messages"][-1].content
    manifest = load_agent_manifest_instructions()
    
    try:
        decision: SupervisorRouterOutput = await structured_router_llm.ainvoke([
            SystemMessage(content=f"{manifest}\n\nAnalyze the current human message and extract the absolute routing domain."),
            HumanMessage(content=user_latest_message)
        ])
        logger.info(f"\n\n==> [Process: SUPERVISOR_ROUTER] Initializing...")
        logger.info(f"    Slot: [{decision.target_agent.upper()}] | Complexity: {decision.complexity.upper()}")
        logger.info(f"    Objective: {decision.objective}\n")
        
        return {
            "current_domain": decision.target_agent,
            "complexity": decision.complexity,
            "required_agents": decision.required_agents,
            "objective": decision.objective
        }
    except Exception as route_err:
        logger.error(f"❌ [ROUTER FAILURE] Defaulting to general framework -> Trace: {str(route_err)}")
        return {"current_domain": "general_memory"}


async def node_planner_agent(state: AgentState):
    """
    Node: PLANNER_AGENT.
    Acts as the project manager. Takes a complex research query, evaluates feasibility,
    and breaks it down into a linear execution plan stored in Workflow Memory.
    """
    user_initial_prompt = state["messages"][0].content if state["messages"] else ""
    logger.info(f"==> [Planner] Breaking down research request: '{user_initial_prompt[:50]}...'")
    
    # Generate the plan using Tier 2
    plan: PlannerOutput = await structured_planner_llm.ainvoke([
        SystemMessage(content="You are the Master Planner. Break down the user's research request into 2-4 concrete, sequential sub-tasks. Focus on logical progression: e.g., Task 1: Find context, Task 2: Analyze limitations, Task 3: Compare results."),
        HumanMessage(content=user_initial_prompt)
    ])
    
    # Convert Pydantic models to dicts for State compatibility
    tasks_state = [{"id": t.id, "desc": t.description, "status": "pending", "result": None} for t in plan.tasks]
    
    logger.info(f"==> [Planner] Generated {len(tasks_state)} tasks successfully.")
    
    # Reset loop counters for the upcoming execution phase
    return {
        "tasks": tasks_state,
        "current_task_id": tasks_state[0]["id"] if tasks_state else None,
        "iteration_count": 0,
        "tool_call_count": 0,
        "action_history": []
    }


async def node_general_agent(state: AgentState):
    """
    Node 1: CORE_ENGINE_AGENT operating as a true topological ReAct Agent.
    Utilizes the cached bound-LLM mechanism for extremely fast inference loops.
    """
    user_id = state.get("user_id", "UNKNOWN_USER")
    current_domain = state.get("current_domain", "general_memory")
    
    # Context injected strictly against the root query to prevent context drifting
    user_initial_prompt = state["messages"][0].content if state["messages"] else ""
    context = await vector_db_service.retrieve_context(user_id, user_initial_prompt, collection_name=current_domain)
    manifest = load_agent_manifest_instructions()
    
    # Explicit topology instructions guiding the LLM how to parse ToolMessages natively
    system_content = (
        f"{manifest}\n\n"
        "You are the CORE_ENGINE_AGENT operating in a Reason-and-Act (ReAct) loop.\n"
        "You must follow this exact execution cycle:\n"
        "1. THOUGHT: Always begin by thinking step-by-step about the user's request and current state.\n"
        "2. ACTION: If you need external data, invoke the appropriate tool.\n"
        "3. OBSERVATION: The system will execute your tool and append a ToolMessage to the conversation history.\n"
        "4. EVALUATION: Read the latest observation. If insufficient, formulate a new THOUGHT and ACTION.\n"
        "5. FINAL ANSWER: If you have gathered all necessary info, provide the final response.\n\n"
        f"<long_term_memory>\n{context}\n</long_term_memory>"
    )
    
    compiled_messages = [SystemMessage(content=system_content)]
    
    # Crucial: Preserves the entire graph message stream (including intermediary ToolMessages)
    compiled_messages.extend(state["messages"])
    
    # Retrieve the pre-compiled, tool-bound LLM via O(1) cache lookup
    llm_with_tools = get_cached_bound_llm(current_domain, llm_tier2_balanced)
        
    compiled_messages = prune_messages_by_token_limit(
        compiled_messages, 
        settings.openai.max_context_tokens, 
        settings.openai.tier2_balanced_model
    )
    
    response = await llm_with_tools.ainvoke(compiled_messages)
    return {"messages": [response]}


async def node_research_paper_agent(state: AgentState):
    """
    Node 2: REASONING_RESEARCH_AGENT (Powered by Tier 3 / o1-mini).
    Executes tasks iteratively. Incorporates Critic Before Tool and Confidence-Based Stopping.
    """
    user_id = state.get("user_id", "UNKNOWN_USER")
    current_domain = state.get("current_domain", "research_papers")
    
    # ─── WORKFLOW MEMORY INJECTION ───
    tasks = state.get("tasks", [])
    current_task_id = state.get("current_task_id")
    
    # Find the active task description
    active_task = next((t for t in tasks if t["id"] == current_task_id), None)
    task_context = f"\n\nCURRENT ACTIVE TASK: {active_task['desc']}" if active_task else ""
    
    # Retrieve base RAG context
    user_initial_prompt = state["messages"][0].content if state["messages"] else ""
    optimized_query = await transform_user_query(user_initial_prompt)
    rag_context = await vector_db_service.retrieve_context(user_id, optimized_query, collection_name=current_domain)
    
    # 🚀 REASONING & LOOP CONTROL PROMPT ENGINEERING
    system_content = (
        "You are the DEEP_RESEARCH_AGENT, a highly autonomous Reasoning AI.\n"
        "You operate in a strict loop to complete your CURRENT ACTIVE TASK.\n\n"
        "CRITICAL RULES FOR EXECUTION:\n"
        "1. CRITIC BEFORE TOOL: Before calling any tool, internally ask yourself: 'Do I actually need external data for this, or can I deduce it from the existing context?' Only call tools if strictly necessary.\n"
        "2. CONFIDENCE-BASED STOPPING: If you have gathered sufficient high-quality information to fulfill the CURRENT ACTIVE TASK, DO NOT call any more tools. Output your final synthesized answer for this task.\n"
        "3. AVOID DUPLICATES: Review the message history. Do not repeat a search query you have already executed.\n"
        f"{task_context}\n"
        f"<research_context>\n{rag_context}\n</research_context>"
    )
    
    compiled_messages = [SystemMessage(content=system_content)]
    compiled_messages.extend(state["messages"])
    
    # Use Tier 3 (o1-mini/gpt-4o) for heavy reasoning tasks
    llm_engine = get_cached_bound_llm(current_domain, llm_tier3_reasoning)
    
    compiled_messages = prune_messages_by_token_limit(
        compiled_messages, 
        settings.openai.max_context_tokens, 
        settings.openai.tier3_reasoning_model
    )
    
    # Increment iteration count
    current_iteration = state.get("iteration_count", 0) + 1
    
    response = await llm_engine.ainvoke(compiled_messages)
    
    return {
        "messages": [response],
        "iteration_count": current_iteration
    }


async def node_vision_detection_agent(state: AgentState):
    """
    Node 3: VISION_DETECTION_AGENT. 
    Operates without heavy tools initially, leveraging high-speed Tier 1 bounds.
    """
    user_id = state.get("user_id", "UNKNOWN_USER")
    current_domain = state.get("current_domain", "vision_detection")
    user_initial_prompt = state["messages"][0].content if state["messages"] else ""
    
    context = await vector_db_service.retrieve_context(user_id, user_initial_prompt, collection_name=current_domain)
    manifest = load_agent_manifest_instructions()

    system_content = (
        f"{manifest}\n\n"
        "You are currently acting as the VISION_DETECTION_AGENT.\n"
        "Resolve tasks regarding object detection, bounding box configurations, or YOLO model variations.\n"
        f"<vision_knowledge_base>\n{context}\n</vision_knowledge_base>"
    )
    
    compiled_messages = [SystemMessage(content=system_content)]
    compiled_messages.extend(state["messages"])
    
    # Retrieve cached LLM (will bind any future vision tools automatically)
    llm_with_tools = get_cached_bound_llm(current_domain, llm_tier1_fast)
    
    compiled_messages = prune_messages_by_token_limit(
        compiled_messages, 
        settings.openai.max_context_tokens, 
        settings.openai.tier1_fast_model
    )
    
    response = await llm_with_tools.ainvoke(compiled_messages)
    return {"messages": [response]}


# --- EVALUATION NODE --- 
async def node_critic_agent(state: AgentState):
    """
    Node 3: CRITIC_AGENT. Evaluates executor output against the overarching objective.
    """
    logger.info(f"\n\n==> [Process: CRITIC_AGENT] Auditing execution trajectory...")
    
    objective = state.get("objective", "Provide a comprehensive answer.")
    executor_payload = state["messages"][-1].content
    
    structured_critic = llm_tier2_balanced.with_structured_output(CriticOutput)
    
    evaluation_prompt = (
        f"You are the strict CRITIC_AGENT.\n"
        f"MASTER OBJECTIVE: {objective}\n\n"
        f"EXECUTOR'S PAYLOAD:\n{executor_payload}\n\n"
        f"Critically analyze if the Executor fulfilled the objective. Expose missing data or hallucinated parameters."
    )
    
    evaluation: CriticOutput = await structured_critic.ainvoke([HumanMessage(content=evaluation_prompt)])
    
    logger.info(f"    Status: {'✅ MET' if evaluation.objective_met else '❌ DEFICIENT'}")
    logger.info(f"    Feedback: {evaluation.feedback}\n")
    
    return {"critic_feedback": evaluation.feedback}


async def node_reflection_agent(state: AgentState):
    """
    Node 4: REFLECTION_AGENT. Synthesizes critic feedback into actionable directives.
    """
    logger.info(f"\n\n==> [Process: REFLECTION_AGENT] Formulating operational reflection...")
    
    critic_feedback = state.get("critic_feedback", "No anomalies detected.")
    executor_payload = state["messages"][-1].content
    
    structured_reflection = llm_tier2_balanced.with_structured_output(ReflectionOutput)
    
    reflection_prompt = (
        f"You are the REFLECTION_AGENT.\n"
        f"CRITIC'S AUDIT:\n{critic_feedback}\n\n"
        f"EXECUTOR'S PAYLOAD:\n{executor_payload}\n\n"
        f"Determine if rework is absolutely required. If yes, generate strict instructions. If no, draft a synthesis memo."
    )
    
    reflection: ReflectionOutput = await structured_reflection.ainvoke([HumanMessage(content=reflection_prompt)])
    
    logger.info(f"    Correction Required: {reflection.needs_rework}")
    logger.info(f"    Directive: {reflection.actionable_advice}\n")
    
    return {"reflection_notes": reflection.actionable_advice}


async def node_final_synthesizer(state: AgentState):
    """
    Node 5: FINAL_SYNTHESIZER. Compiles the ultimate formatted response.
    """
    logger.info(f"\n\n==> [Process: FINAL_SYNTHESIZER] Constructing final payload...")
    
    original_prompt = state["messages"][0].content
    executor_payload = state["messages"][-1].content
    reflection_notes = state.get("reflection_notes", "")
    
    synthesis_prompt = (
        "You are the FINAL_SYNTHESIZER. Integrate the core data and reflection notes into a highly polished, markdown-formatted response.\n"
        f"<user_query>\n{original_prompt}\n</user_query>\n"
        f"<raw_data>\n{executor_payload}\n</raw_data>\n"
        f"<reflection>\n{reflection_notes}\n</reflection>"
    )
    
    final_response = await llm_tier2_balanced.ainvoke([HumanMessage(content=synthesis_prompt)])
    
    logger.info(f"    Process Complete. Handshake ready.\n\n")
    
    return {"messages": [final_response]}