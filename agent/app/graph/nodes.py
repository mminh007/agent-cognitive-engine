# app/graph/nodes.py
import os
import functools
import tiktoken
from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel

from app.graph.state import AgentState
from app.services.query_transformer import transform_user_query
from app.mcp.tool_registry import get_tools_by_domain
from app.core.settings import settings
from app.core.logger import setup_app_logger
from app.core.metrics import GRAPH_ITERATIONS
from app.bootstrap.container import container
from app.graph.config import get_cached_bound_llm, get_structured_llm, get_llm_instance

logger = setup_app_logger("CognitiveNodes")


# ─── STRUCTURED OUTPUT SCHEMAS ───

class SupervisorRouterOutput(BaseModel):
    target_agent: str = Field(description="The target domain: 'general_memory' (default for web search, itineraries, general tasks), 'research_papers' (STRICTLY for academic papers/PDFs), 'vision_detection' (images/YOLO).")
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
    target_agent: Literal["general_memory", "research_papers", "vision_detection"] = Field(description="The agent lane. Use 'general_memory' for almost everything (web search, planning). Use 'research_papers' ONLY for academic literature/PDFs. Use 'vision_detection' ONLY for images.")

class PlannerOutput(BaseModel):
    tasks: List[TaskItem] = Field(description="List of sequential tasks required to fulfill the user's research request.")

class Finding(BaseModel):
    statement: str = Field(description="A concise factual statement.")
    evidence: Optional[str] = Field(default=None, description="Evidence or citation supporting this statement.")
    confidence: float = Field(description="Confidence level in this finding (0.0 to 1.0).")

class ExecutorOutput(BaseModel):
    """
    Schema used exclusively by node_finding_extractor (not by executor nodes directly).
    Executor nodes now produce free-form Markdown; this schema is applied in a
    dedicated downstream extraction step to avoid JSON formatting failures.
    """
    result_summary: str = Field(description="A concise 1-3 sentence summary of the main answer or result.")
    findings: List[Finding] = Field(description="Key factual findings extracted from the executor's response.")

class CriticOutput(BaseModel):
    objective_met: bool = Field(description="Assess if the executor fully achieved the initial objective.")
    tool_quality_score: int = Field(description="Score (1-10) evaluating the appropriate use of tools and context.")
    evidence_quality_score: int = Field(description="Score (1-10) evaluating the strength of evidence supporting the findings.")
    citation_quality_score: int = Field(description="Score (1-10) evaluating proper source citations.")
    freshness_score: int = Field(description="Score (1-10) evaluating the recency/freshness of the information.")
    feedback: str = Field(description="Detailed feedback synthesizing the evaluations into a final verdict.")

class ReflectionOutput(BaseModel):
    needs_rework: bool = Field(description="Determine if the executor needs to rerun based on the critic's severity.")
    actionable_advice: str = Field(description="Strict, actionable instructions for the executor or synthesis notes if passing.")


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


def _build_task_context(state: AgentState) -> str:
    """
    Builds a plain-text task context block for injection into executor system prompts.

    REPLACES: _build_executor_instructions() which previously appended JSON formatting
    directives alongside task context — the root cause of JSONDecodeError failures
    when executors generated Markdown or code snippet responses.

    Executors now receive ONLY task context; JSON formatting is never requested here.
    Structured extraction is performed downstream by node_finding_extractor.
    """
    tasks = state.get("tasks", [])
    current_task_id = state.get("current_task_id")

    completed_findings = []
    for t in tasks:
        if t.get("status") == "completed" and t.get("findings"):
            for f in t["findings"]:
                confidence = f.get("confidence", 0.0) if isinstance(f, dict) else getattr(f, "confidence", 0.0)
                stmt = f.get("statement", "") if isinstance(f, dict) else getattr(f, "statement", "")
                completed_findings.append(f"  - Task {t['id']}: {stmt} (confidence: {confidence:.2f})")

    active_task = next((t for t in tasks if t["id"] == current_task_id), None) if current_task_id else None

    parts = []
    if active_task:
        parts.append(f"CURRENT ACTIVE TASK: {active_task['desc']}")
    if completed_findings:
        parts.append("PRIOR TASK FINDINGS (for context):\n" + "\n".join(completed_findings))

    return "\n\n".join(parts) if parts else ""


@functools.lru_cache(maxsize=1)
def load_agent_manifest_instructions(file_path: str = "AGENTS.md") -> str:
    """
    Extracts baseline operational principles from the external Markdown manifest.
    Cached after first read — manifest is static for the lifetime of the process.
    """
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            logger.info("==> [Manifest] Loaded AGENTS.md into cache.")
            return f.read()
    return "You are a highly capable engineering AI assistant."


# ─── GRAPH NODES ───

async def node_supervisor_router(state: AgentState, config: RunnableConfig = None):
    """
    Node 0: Supervisor. Identifies the intent and routes to the correct isolated domain lane.
    Persists complexity tier and required_agents list into State for downstream routing.
    """
    user_latest_message = state["messages"][-1].content
    manifest = load_agent_manifest_instructions()

    try:
        structured_router_llm = get_structured_llm(1, SupervisorRouterOutput, config)
        decision: SupervisorRouterOutput = await structured_router_llm.ainvoke([
            SystemMessage(content=f"{manifest}\n\nAnalyze the current human message and extract the absolute routing domain."),
            HumanMessage(content=user_latest_message)
        ])
        logger.info(f"\n\n==> [PROCESS] SUPERVISOR_ROUTER Initializing...")
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
        return {
            "current_domain": "general_memory",
            "complexity": "low",
            "required_agents": []
        }


async def node_planner_agent(state: AgentState, config: RunnableConfig = None):
    """
    Node: PLANNER_AGENT.
    Acts as the project manager. Takes a complex research query, evaluates feasibility,
    and breaks it down into a linear execution plan stored in Workflow Memory.
    Only invoked for complexity='medium' or 'high' — low complexity bypasses this node.
    """
    user_initial_prompt = state["messages"][0].content if state["messages"] else ""
    objective = state.get("objective", user_initial_prompt)
    logger.info(f"==> [Planner] Breaking down research request: '{user_initial_prompt[:50]}...'")

    try:
        required_agents = state.get("required_agents", ["general_memory"])
        agents_str = ", ".join(required_agents) if required_agents else "general_memory"

        structured_planner_llm = get_structured_llm(2, PlannerOutput, config)
        plan: PlannerOutput = await structured_planner_llm.ainvoke([
            SystemMessage(
                content=(
                    "You are the Master Planner. Break down the user's request into 2-4 "
                    "concrete, sequential sub-tasks. Focus on logical progression.\n\n"
                    f"Overarching Objective: {objective}\n"
                    f"Available Agent Domains: {agents_str}\n"
                    "Assign the most appropriate agent domain to each task.\n"
                    "CRITICAL ROUTING RULES:\n"
                    "- 'general_memory': Default for web searches, weather forecasts, itineraries, coding, and standard data gathering.\n"
                    "- 'research_papers': STRICTLY for reading academic literature, scientific papers, or PDFs. Do NOT use for regular web searches.\n"
                    "- 'vision_detection': STRICTLY for image processing."
                )
            ),
            HumanMessage(content=user_initial_prompt)
        ])

        # Convert Pydantic models to dicts for State compatibility
        tasks_state = [
            {"id": t.id, "desc": t.description, "status": "pending",
             "target_agent": t.target_agent, "result": None, "findings": []}
            for t in plan.tasks
        ]
        logger.info(f"==> [Planner] Generated {len(tasks_state)} tasks successfully.")

    except Exception as plan_err:
        # Graceful fallback: create a single synthetic task from the raw user prompt
        logger.error(f"❌ [PLANNER FAILURE] Structured parse failed — falling back to single task. Trace: {str(plan_err)}")
        target_agent = state.get("current_domain", "general_memory")
        tasks_state = [{"id": 1, "desc": user_initial_prompt, "status": "pending",
                        "target_agent": target_agent, "result": None, "findings": []}]

    # Reset loop counters for the upcoming execution phase; initialize rework_count
    return {
        "tasks": tasks_state,
        "current_task_id": tasks_state[0]["id"] if tasks_state else None,
        "iteration_count": 0,
        "tool_call_count": 0,
        "action_history": [],
        "rework_count": 0
    }


async def node_general_agent(state: AgentState, config: RunnableConfig = None):
    """
    Node 1: CORE_ENGINE_AGENT operating as a true topological ReAct Agent.

    OUTPUT CONTRACT: Returns plain Markdown — no JSON wrapping required or expected.
    Structured finding extraction is delegated entirely to node_finding_extractor,
    which runs downstream after the tool loop completes. This eliminates the class
    of JSONDecodeError bugs caused by code snippets embedded in JSON strings.
    """
    user_id = state.get("user_id", "UNKNOWN_USER")
    current_domain = state.get("current_domain", "general_memory")

    if not container.hybrid_search:
        await container.initialize()

    # Context injected strictly against the root query to prevent context drifting
    user_initial_prompt = state["messages"][0].content if state["messages"] else ""
    context = await container.hybrid_search.retrieve(user_id, user_initial_prompt, collection=current_domain)
    manifest = load_agent_manifest_instructions()
    task_context = _build_task_context(state)

    system_content = (
        f"{manifest}\n\n"
        "You are the CORE_ENGINE_AGENT operating in a Reason-and-Act (ReAct) loop.\n"
        "Follow this execution cycle:\n"
        "1. THOUGHT: Think step-by-step about the user's request and current state.\n"
        "2. ACTION: If you need external data, invoke the appropriate tool.\n"
        "3. OBSERVATION: The system will execute your tool and append a ToolMessage.\n"
        "4. EVALUATION: If the observation is insufficient, form a new THOUGHT and ACTION.\n"
        "5. FINAL ANSWER: Once you have sufficient information, write a complete, well-structured "
        "Markdown response. Do NOT format your answer as JSON.\n\n"
    )

    if task_context:
        system_content += f"<current_task>\n{task_context}\n</current_task>\n\n"

    system_content += f"<long_term_memory>\n{context}\n</long_term_memory>"

    compiled_messages = [SystemMessage(content=system_content)]
    # Crucial: Preserves the entire graph message stream (including intermediary ToolMessages)
    compiled_messages.extend(state["messages"])

    # Retrieve the pre-compiled, tool-bound LLM via O(1) cache lookup
    base_llm = get_llm_instance(2, config)
    llm_with_tools = get_cached_bound_llm(current_domain, base_llm)

    compiled_messages = prune_messages_by_token_limit(
        compiled_messages,
        settings.openai.max_context_tokens,
        settings.openai.tier2_balanced_model
    )

    current_iteration = state.get("iteration_count", 0) + 1

    response = await llm_with_tools.ainvoke(compiled_messages)
    return {
        "messages": [response],
        "iteration_count": current_iteration
    }


async def node_research_paper_agent(state: AgentState, config: RunnableConfig = None):
    """
    Node 2: REASONING_RESEARCH_AGENT (Powered by Tier 3).
    Executes tasks iteratively using Critic Before Tool and Confidence-Based Stopping.

    OUTPUT CONTRACT: Returns plain Markdown — no JSON wrapping required or expected.
    """
    user_id = state.get("user_id", "UNKNOWN_USER")
    current_domain = state.get("current_domain", "research_papers")

    if not container.hybrid_search:
        await container.initialize()

    task_context = _build_task_context(state)

    # Retrieve base RAG context against optimized query
    user_initial_prompt = state["messages"][0].content if state["messages"] else ""
    optimized_query = await transform_user_query(user_initial_prompt)
    rag_context = await container.hybrid_search.retrieve(user_id, optimized_query, collection=current_domain)

    system_content = (
        "You are the DEEP_RESEARCH_AGENT, a highly autonomous Reasoning AI.\n"
        "You operate in a strict loop to complete your CURRENT ACTIVE TASK.\n\n"
        "CRITICAL RULES FOR EXECUTION:\n"
        "1. CRITIC BEFORE TOOL: Before calling any tool, internally ask: 'Do I actually need external "
        "data for this, or can I deduce it from the existing context?' Only call tools if strictly necessary.\n"
        "2. CONFIDENCE-BASED STOPPING: If you have gathered sufficient high-quality information, "
        "DO NOT call any more tools. Output your final synthesized answer for this task.\n"
        "3. AVOID DUPLICATES: Review the message history. Do not repeat a search query you have already executed.\n"
        "4. FINAL ANSWER FORMAT: Write a complete, well-structured Markdown response. "
        "Do NOT format your answer as JSON.\n\n"
    )

    if task_context:
        system_content += f"<current_task>\n{task_context}\n</current_task>\n\n"

    system_content += f"<research_context>\n{rag_context}\n</research_context>"

    compiled_messages = [SystemMessage(content=system_content)]
    compiled_messages.extend(state["messages"])

    # Use Tier 3 (high-reasoning model) for heavy research tasks
    base_llm = get_llm_instance(3, config)
    llm_engine = get_cached_bound_llm(current_domain, base_llm)

    compiled_messages = prune_messages_by_token_limit(
        compiled_messages,
        settings.openai.max_context_tokens,
        settings.openai.tier3_reasoning_model
    )

    current_iteration = state.get("iteration_count", 0) + 1

    response = await llm_engine.ainvoke(compiled_messages)
    return {
        "messages": [response],
        "iteration_count": current_iteration
    }


async def node_vision_detection_agent(state: AgentState, config: RunnableConfig = None):
    """
    Node 3: VISION_DETECTION_AGENT.
    Operates without heavy tools initially, leveraging high-speed Tier 1.

    OUTPUT CONTRACT: Returns plain Markdown — no JSON wrapping required or expected.
    """
    user_id = state.get("user_id", "UNKNOWN_USER")
    current_domain = state.get("current_domain", "vision_detection")

    if not container.hybrid_search:
        await container.initialize()

    user_initial_prompt = state["messages"][0].content if state["messages"] else ""
    context = await container.hybrid_search.retrieve(user_id, user_initial_prompt, collection=current_domain)
    manifest = load_agent_manifest_instructions()
    task_context = _build_task_context(state)

    system_content = (
        f"{manifest}\n\n"
        "You are the VISION_DETECTION_AGENT.\n"
        "Resolve tasks regarding object detection, bounding box configurations, or YOLO model variations.\n"
        "Write a complete, well-structured Markdown response. Do NOT format your answer as JSON.\n\n"
    )

    if task_context:
        system_content += f"<current_task>\n{task_context}\n</current_task>\n\n"

    system_content += f"<vision_knowledge_base>\n{context}\n</vision_knowledge_base>"

    compiled_messages = [SystemMessage(content=system_content)]
    compiled_messages.extend(state["messages"])

    # Retrieve cached LLM (will bind any future vision tools automatically)
    base_llm = get_llm_instance(1, config)
    llm_with_tools = get_cached_bound_llm(current_domain, base_llm)

    compiled_messages = prune_messages_by_token_limit(
        compiled_messages,
        settings.openai.max_context_tokens,
        settings.openai.tier1_fast_model
    )

    current_iteration = state.get("iteration_count", 0) + 1

    response = await llm_with_tools.ainvoke(compiled_messages)
    return {
        "messages": [response],
        "iteration_count": current_iteration
    }


async def node_finding_extractor(state: AgentState, config: RunnableConfig = None):
    """
    Node: FINDING_EXTRACTOR.

    PURPOSE — Separation of Content Generation from Information Extraction:
    Executor nodes (general_agent, research_paper_agent, vision_detection_agent) now
    produce free-form Markdown responses. This node reads that plain text and applies
    a dedicated, low-cost Tier-1 structured-output call to extract:
      - result_summary: a concise 1-3 sentence digest of the executor's answer
      - findings: a typed List[Finding] with statements, evidence, and confidence scores

    WHY THIS DESIGN:
    Previously, executors were prompted to output the entire answer inside a JSON string
    field (result: str). Code snippets, markdown headers, and unescaped quotes in that
    field caused frequent JSONDecodeError failures. By separating generation from
    extraction, executors can produce any format naturally; extraction is handled here
    where a short, well-scoped prompt reliably produces valid JSON via Function Calling.

    RESULT: Writes to state['raw_executor_output'] and state['extracted_findings'],
    which are consumed by node_task_manager (no JSON parsing from messages needed).
    """
    logger.info("==> [PROCESS] FINDING_EXTRACTOR Extracting structured findings from executor output...")

    executor_raw = state["messages"][-1].content

    extraction_prompt = (
        "You are a precise Information Extraction engine. Your ONLY job is to read the "
        "following executor response and extract structured data.\n\n"
        "Extract:\n"
        "1. result_summary — a concise 1-3 sentence digest capturing the main answer or conclusion.\n"
        "2. findings — a list of key factual claims, each with:\n"
        "   - statement: a single, self-contained factual sentence\n"
        "   - evidence: a quote or citation from the text (optional)\n"
        "   - confidence: a float from 0.0 (uncertain) to 1.0 (certain)\n\n"
        f"<executor_output>\n{executor_raw}\n</executor_output>"
    )

    try:
        # Tier 1 (fast/cheap model) is sufficient — extraction is a short, focused call
        structured_extractor = get_structured_llm(1, ExecutorOutput, config)
        extraction: ExecutorOutput = await structured_extractor.ainvoke([
            HumanMessage(content=extraction_prompt)
        ])
        raw_summary = extraction.result_summary
        findings = [f.model_dump() for f in extraction.findings]
        logger.info(f"    Extracted {len(findings)} finding(s). Summary: '{raw_summary[:80]}...'")
    except Exception as e:
        logger.error(f"❌ [EXTRACTOR FAILURE] Falling back to raw executor text. Trace: {str(e)}")
        raw_summary = executor_raw
        findings = []

    return {
        "raw_executor_output": raw_summary,
        "extracted_findings": findings
    }


async def node_direct_executor_init(state: AgentState):
    """
    Node: DIRECT_EXECUTOR_INIT.
    Lightweight initializer for the complexity='low' bypass path.
    Skips Planner to save LLM cost, creates a single synthetic task from the user prompt,
    and sets all required loop counters to their initial values.
    """
    user_initial_prompt = state["messages"][0].content if state["messages"] else ""
    objective = state.get("objective", user_initial_prompt)
    target_agent = state.get("current_domain", "general_memory")

    # Single synthetic task — no decomposition needed for simple queries
    tasks_state = [{"id": 1, "desc": objective, "status": "pending",
                    "target_agent": target_agent, "result": None, "findings": []}]

    logger.info("==> [DirectInit] Low-complexity bypass: skipping Planner, single task created.")

    return {
        "tasks": tasks_state,
        "current_task_id": 1,
        "iteration_count": 0,
        "tool_call_count": 0,
        "action_history": [],
        "rework_count": 0
    }


# ─── EVALUATION NODES ───

async def node_critic_agent(state: AgentState, config: RunnableConfig = None):
    """
    Node: CRITIC_AGENT. Evaluates executor output against the overarching objective.
    Reads from state['raw_executor_output'] and state['extracted_findings'] as set
    by node_finding_extractor — no message parsing required here.
    """
    logger.info(f"\n\n==> [PROCESS] CRITIC_AGENT Auditing execution trajectory...")

    objective = state.get("objective", "Provide a comprehensive answer.")
    # Read clean plain-text payload from state (set by node_finding_extractor)
    executor_payload = state.get("raw_executor_output") or state["messages"][-1].content
    findings = state.get("extracted_findings", [])
    action_history = state.get("action_history", [])

    evaluation_prompt = (
        f"You are the strict CRITIC_AGENT.\n"
        f"MASTER OBJECTIVE: {objective}\n\n"
        f"Critically analyze if the Executor fulfilled the objective.\n"
        f"Provide integer scores (1-10) for tool_quality_score, evidence_quality_score, "
        f"citation_quality_score, and freshness_score.\n"
        f"Expose missing data or hallucinated parameters in your feedback.\n\n"
        f"<executor_payload>\n{executor_payload}\n</executor_payload>\n\n"
        f"<extracted_findings>\n{findings}\n</extracted_findings>\n\n"
        f"<action_history>\n{action_history}\n</action_history>"
    )

    try:
        structured_critic_llm = get_structured_llm(2, CriticOutput, config)
        evaluation: CriticOutput = await structured_critic_llm.ainvoke([HumanMessage(content=evaluation_prompt)])
    except Exception as e:
        logger.error(f"❌ [CRITIC FAILURE] Parse error: {str(e)}")
        evaluation = CriticOutput(
            objective_met=True,
            tool_quality_score=5, evidence_quality_score=5, citation_quality_score=5, freshness_score=5,
            feedback="[Fallback] Parsing failed, proceeding automatically."
        )

    logger.info(f"    Status: {'✅ MET' if evaluation.objective_met else '❌ DEFICIENT'}")
    logger.info(f"    Scores: Tools={evaluation.tool_quality_score}, Evidence={evaluation.evidence_quality_score}, "
                f"Citations={evaluation.citation_quality_score}, Freshness={evaluation.freshness_score}")
    logger.info(f"    Feedback: {evaluation.feedback}\n")

    return {"critic_feedback": evaluation.feedback}


async def node_reflection_agent(state: AgentState, config: RunnableConfig = None):
    """
    Node: REFLECTION_AGENT. Synthesizes critic feedback into actionable directives.
    Increments rework_count when a rework cycle is approved, enabling the bounded
    rework safeguard in route_from_reflection().
    """
    logger.info(f"\n\n==> [PROCESS] REFLECTION_AGENT Formulating operational reflection...")

    critic_feedback = state.get("critic_feedback", "No anomalies detected.")
    executor_payload = state.get("raw_executor_output") or state["messages"][-1].content
    current_rework_count = state.get("rework_count", 0)

    reflection_prompt = (
        f"You are the REFLECTION_AGENT.\n"
        f"Determine if rework is absolutely required. If yes, generate strict instructions. "
        f"If no, draft a synthesis memo.\n\n"
        f"<critic_audit>\n{critic_feedback}\n</critic_audit>\n\n"
        f"<executor_payload>\n{executor_payload}\n</executor_payload>"
    )

    try:
        structured_reflection_llm = get_structured_llm(2, ReflectionOutput, config)
        reflection: ReflectionOutput = await structured_reflection_llm.ainvoke([HumanMessage(content=reflection_prompt)])
    except Exception as e:
        logger.error(f"❌ [REFLECTION FAILURE] Parse error: {str(e)}")
        reflection = ReflectionOutput(needs_rework=False, actionable_advice="[Fallback] Parsing failed, proceeding without rework.")

    # Increment rework_count only when a rework cycle is actually triggered
    new_rework_count = current_rework_count + 1 if reflection.needs_rework else current_rework_count

    logger.info(f"    Correction Required: {reflection.needs_rework} | Rework Cycle: {new_rework_count}")
    logger.info(f"    Directive: {reflection.actionable_advice}\n")

    return {
        "reflection_notes": reflection.actionable_advice,
        "needs_rework": reflection.needs_rework,
        "rework_count": new_rework_count
    }


async def node_final_synthesizer(state: AgentState, config: RunnableConfig = None):
    """
    Node: FINAL_SYNTHESIZER. Compiles the ultimate formatted response.
    Reads from completed task results and findings to produce a polished Markdown answer.
    """
    logger.info(f"\n\n==> [PROCESS] FINAL_SYNTHESIZER Constructing final payload...")

    current_objective = state.get("objective", state["messages"][-1].content if state["messages"] else "")
    tasks = state.get("tasks", [])

    all_findings = []
    all_results = []
    for t in tasks:
        if t.get("findings"):
            for f in t["findings"]:
                stmt = f.get("statement", "") if isinstance(f, dict) else getattr(f, "statement", "")
                all_findings.append(f"- {stmt}")
        if t.get("result"):
            all_results.append(f"### Task: {t['desc']}\n{t['result']}")

    findings_text = "\n".join(all_findings) if all_findings else "No key findings extracted."
    results_text = "\n\n".join(all_results) if all_results else "No detailed results available."

    synthesis_prompt = (
        "You are the FINAL_SYNTHESIZER. Create a highly polished, professional Markdown response "
        "that directly addresses the user's request.\n"
        "Synthesize the outputs from the completed tasks into a coherent and unified answer. "
        "Do not artificially separate the response into 'Key Findings' and 'Detailed Analysis' unless appropriate.\n"
        "Extract, deduplicate, and compile any sources/citations into a 'Bibliography' at the end if applicable.\n\n"
        f"<user_objective>\n{current_objective}\n</user_objective>\n\n"
        f"<task_findings>\n{findings_text}\n</task_findings>\n\n"
        f"<task_results>\n{results_text}\n</task_results>\n"
    )

    final_response = await get_llm_instance(2, config).ainvoke([HumanMessage(content=synthesis_prompt)])

    total_iterations = state.get("iteration_count", 0)
    GRAPH_ITERATIONS.labels(domain=state.get("current_domain", "general_memory")).observe(total_iterations)

    logger.info("    Process Complete. Handshake ready.\n\n")

    return {"messages": [final_response]}


async def node_task_manager(state: AgentState):
    """
    Node: TASK_MANAGER.
    Manages iterating through Planner tasks. Marks current as complete, advances pointer.

    REFACTORED: No longer performs JSON parsing from executor message content.
    Instead reads directly from state['raw_executor_output'] and state['extracted_findings'],
    which are set by node_finding_extractor. This eliminates the fragile try/except
    JSON parse that previously lost findings on any formatting error.
    """
    logger.info(f"\n\n==> [PROCESS] TASK_MANAGER Auditing task registry...")

    tasks = state.get("tasks", [])
    current_task_id = state.get("current_task_id")

    if not tasks:
        logger.info("    No tasks registered. Proceeding.")
        return {"current_task_id": None}

    # Read structured data directly from state — set by node_finding_extractor
    extracted_result = state.get("raw_executor_output") or state["messages"][-1].content
    extracted_findings = state.get("extracted_findings", [])

    updated_tasks = []
    for t in tasks:
        if t["id"] == current_task_id:
            t["status"] = "completed"
            t["result"] = extracted_result
            t["findings"] = extracted_findings
        updated_tasks.append(t)

    next_task = next((t for t in updated_tasks if t.get("status") == "pending"), None)

    if next_task:
        logger.info(f"    Advancing to Next Task -> [{next_task['id']}]: {next_task['desc']}")
        # Context Sandboxing: clear all messages except the original prompt
        messages_to_remove = [RemoveMessage(id=m.id) for m in state["messages"][1:] if getattr(m, "id", None)]

        return {
            "tasks": updated_tasks,
            "current_task_id": next_task["id"],
            "iteration_count": 0,
            "tool_call_count": 0,
            "rework_count": 0,
            # Reset extraction state for the next task cycle
            "raw_executor_output": None,
            "extracted_findings": [],
            "messages": messages_to_remove
        }
    else:
        logger.info("    All tasks completed. Proceeding to synthesis.")
        return {
            "tasks": updated_tasks,
            "current_task_id": None
        }