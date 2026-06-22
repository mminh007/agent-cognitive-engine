# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from app.mcp.mcp_client import mcp_manager

from app.api.router_registry import build_master_router
from app.services.pruning_policy import run_pruning_policy
from app.core.logger import setup_app_logger

logger = setup_app_logger("MainApp")
scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the background processes throughout the lifecycle of FastAPI."""

    logger.info("⚙️ Connecting to external Upstream MCP Servers...")
    await mcp_manager.initialize_all_servers()

    logger.info("⚙️ Booting up internal background schedulers...")
    
    # Configure the ChromaDB pruning function to run every 24 hours with safeguards against overlapping executions
    scheduler.add_job(
        run_pruning_policy,
        trigger=IntervalTrigger(hours=24),
        id="chromadb_pruning_job",
        replace_existing=True,
        max_instances=1
    )
    scheduler.start()
    
    yield
    
    logger.info("🛑 Shutting down background schedulers...")
    scheduler.shutdown(wait=False)

    logger.info("🛑 Shutting down MCP Server connections...")
    await mcp_manager.shutdown()

# Core application engine setup
app = FastAPI(
    title="Core High-Velocity Multi-Agent API Engine",
    version="2.0.0",
    lifespan=lifespan
)

# 🚀 EXPOSE PROMETHEUS METRICS ENDPOINT
@app.get("/metrics", tags=["Observability"])
def get_prometheus_metrics():
    """
    Scrape target endpoint for Prometheus server.
    Emits centralized application health and process metrics formatted via text standards.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Fetch the aggregated routing map generated automatically via runtime scanning loops
api_gateway_router = build_master_router()

# Mount the consolidated registry onto the global app pipeline instance
app.include_router(api_gateway_router)
