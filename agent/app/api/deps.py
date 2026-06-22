# app/api/deps.py
from collections.abc import AsyncGenerator
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from app.graph.workflow import compiled_graph
from app.core.settings import settings
#from app.graph.vision_workflow import vision_graph

async def get_agent_graph()-> AsyncGenerator:
    """
    Dependency provider that yields the compiled LangGraph execution instance.
    This lifecycle isolate allows easy mock injections during automated unit tests.
    """
    async with AsyncRedisSaver(redis_url=settings.redis.url) as saver:
        runtime_graph = compiled_graph.compile(checkpointer=saver)
        yield runtime_graph


# async def get_vision_graph()-> AsyncGenerator:
#     """
#     Dependency provider for the Vision Agent's specific graph instance.
#     This allows for clean separation of concerns and easy swapping of graph logic if needed.
#     """
#     yield vision_graph