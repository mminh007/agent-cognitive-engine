# app/core/metrics.py
"""
Centralized Prometheus Metrics Registry.
Defines all observable parameters for Traffic, Token/Cost Governance, and RAG/Agent Performance.
Allows separation of monitoring logic from core business operations.
"""
from prometheus_client import Counter, Gauge, Histogram

# ==========================================
# 1. TRAFFIC & STREAMING METRICS
# ==========================================
SSE_ACTIVE_STREAMS = Gauge(
    'sse_active_streams', 
    'Number of concurrent active Server-Sent Event (SSE) HTTP streams'
)

SSE_DISCONNECT_TOTAL = Counter(
    'sse_disconnect_total', 
    'Total number of closed SSE connections categorized by termination reason',
    ['reason'] # e.g., "completed", "abrupt_client_disconnect"
)

# ==========================================
# 2. LANGGRAPH SAFEGUARDS & LOOP CONTROL
# ==========================================
GRAPH_ITERATIONS = Histogram(
    'langgraph_iteration_count', 
    'Distribution of topological iterations (cycles) per graph execution session',
    ['domain'],
    buckets=(1, 2, 3, 4, 5, 6, 7, 8, 10, 15)
)

FORCED_TERMINATION_TOTAL = Counter(
    'langgraph_forced_termination_total', 
    'Total occurrences where a procedural safeguard tripped and forced a hard termination',
    ['domain', 'reason'] # e.g., "max_iterations", "budget_depleted", "duplicate_action", "max_rework_cycles"
)

DUPLICATE_TOOL_CALL_TOTAL = Counter(
    'langgraph_duplicate_tool_call_total',
    'Total number of redundant tool invocations blocked by the action tracker mechanism',
    ['tool_name']
)

# ==========================================
# 3. HYBRID RAG & CACHE METRICS
# ==========================================
SEMANTIC_CACHE_LOOKUPS = Counter(
    'semantic_cache_requests_total', 
    'Total semantic cache evaluation lookups categorized by resolution status', 
    ['status'] # e.g., "hit", "miss", "error"
)