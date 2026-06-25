# app/boostrap/startup.py
from app.bootstrap.container import container

async def startup():
    await container.initialize()

async def shutdown():
    await container.shutdown()