# app/boostrap/startup.py
from app.bootstrap.container import Container

container = Container()

async def startup():
    await container.initialize()

async def shutdown():
    await container.shutdown()