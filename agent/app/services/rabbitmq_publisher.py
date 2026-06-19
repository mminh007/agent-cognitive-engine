# app/services/rabbitmq_publisher.py
import json
import aio_pika
from langchain_core.messages import messages_to_dict
from app.core.settings import settings
from app.core.logger import setup_app_logger

logger = setup_app_logger("RabbitMqPublisher")

async def publish_extraction_task(user_id: str, session_id: str, target_rag_domain: str, messages: list):
    """
    Serializes conversation history payload and posts it along with its resolved domain routing tag.
    """
    try:
        connection = await aio_pika.connect_robust(settings.rabbitmq.url)
        
        async with connection:
            channel = await connection.channel()

            # 1. Declare Dead Letter Exchange and Dead Letter Queue
            dlx_name = "dlx_memory_exchange"
            dlq_name = "fact_extraction_dlq"
            
            dlx = await channel.declare_exchange(dlx_name, aio_pika.ExchangeType.DIRECT)
            dlq = await channel.declare_queue(dlq_name, durable=True)
            await dlq.bind(dlx, routing_key=dlq_name)
            
            # 2. Declare the main queue and bind it to the DLX
            queue_arguments = {
                "x-dead-letter-exchange": dlx_name,
                "x-dead-letter-routing-key": dlq_name
            }
            await channel.declare_queue(
                settings.rabbitmq.queue_name, 
                durable=True, 
                arguments=queue_arguments
            )
            
            messages_dict = messages_to_dict(messages)
            
            # 🚀 Injected explicit target_rag_domain parameter inside the queue payload envelope
            payload = {
                "codename_process": "FactExtractionProcess",
                "user_id": user_id,
                "session_id": session_id,
                "target_rag_domain": target_rag_domain or "general_memory",
                "messages": messages_dict
            }
            
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(payload).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                ),
                routing_key=settings.rabbitmq.queue_name,
            )
            
            # Trailing layout whitespace separation boundary preserved
            logger.info(f"==> [RabbitMQ] Publish Successfully - Codename: FactExtractionProcess - Assigned Target: {target_rag_domain}\n")
            
    except Exception as e:
        logger.error(f"❌ [RabbitMQ Publisher Failure] Failed to marshal task parameters: {str(e)}\n")
