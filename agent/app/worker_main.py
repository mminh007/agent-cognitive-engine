# app/worker_main.py
import asyncio
import json
import sys
import aio_pika
from app.core.settings import settings
from langchain_core.messages import messages_from_dict
from app.services import MemoryWorker
from app.core.logger import setup_app_logger
from bootstrap.container import container

logger = setup_app_logger("WorkerMainCore")

async def process_message(message: aio_pika.IncomingMessage):
    """
    Consumer callback handler unpacking AMQP envelopes and pushing payloads to dedicated collection workers.
    """
    async with message.process():
        try:
            payload = json.loads(message.body.decode())
            user_id = payload.get("user_id")
            session_id = payload.get("session_id")
            target_collection = payload.get("target_rag_domain", "general_memory")  # 🚀 Extract dynamic collection destination
            raw_messages = payload.get("messages")
            codename = payload.get("codename_process")
            
            logger.info(f"==> [Worker Event] Processing Payload | Action: {codename} | Target Partition: {target_collection}")
            
            messages = messages_from_dict(raw_messages)
            
            memory_worker = MemoryWorker(
                memory_service= container.memory_service,
                extractor= container.extractor
            )
            # Pass collection name downstream to isolate storage tasks
            await memory_worker.process_extraction_task(user_id, target_collection, messages)
            
            logger.info(f"==> [Worker Resolution] Task resolved successfully for Session: {session_id}\n")
            
            await message.ack()  # Explicit ACK after successful processing
        except Exception as e:
            logger.error(f"❌ [Worker Core Loop Error] Failed to handle runtime message frame: {str(e)}\n")

            await message.reject(requeue=False)
        logger.info("==> [DLQ] Message has been routed to Dead Letter Queue for manual review.\n")

async def main():
    logger.info("⚙️ Memory Worker is booting and attempting connection hooks with RabbitMQ...")
    
    await container.initialize()

    connection = await aio_pika.connect_robust(settings.rabbitmq.url)
    
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        dlx_name = "dlx_memory_exchange"
        dlq_name = "fact_extraction_dlq"

        dlx = await channel.declare_exchange(dlx_name, aio_pika.ExchangeType.DIRECT)
        dlq = await channel.declare_queue(dlq_name, durable=True)
        await dlq.bind(dlx, routing_key=dlq_name)
        
        queue_arguments = {
            "x-dead-letter-exchange": dlx_name,
            "x-dead-letter-routing-key": dlq_name
        }
        queue = await channel.declare_queue(
            settings.rabbitmq.queue_name,
            durable=True,
            arguments=queue_arguments)
        
        logger.info(f"🎧 Worker is actively blocking on event loop queue parameters: '{settings.rabbitmq.queue_name}'...\n")
        await queue.consume(process_message)
        await asyncio.Future()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
