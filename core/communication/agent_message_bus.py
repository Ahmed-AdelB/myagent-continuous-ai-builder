"""
Agent Communication Bus - GPT-5 Recommendation #3
Centralized event-driven messaging system for agent coordination

Replaces ad-hoc inter-agent communication with structured message bus
to prevent inconsistency and race conditions.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Standard message types for agent communication"""
    # Test and quality related
    TEST_RESULT = "test_result"
    TEST_REQUEST = "test_request"
    QUALITY_UPDATE = "quality_update"

    # Code and development
    CODE_CHANGE = "code_change"
    BUILD_STATUS = "build_status"
    DEPLOY_STATUS = "deploy_status"

    # Design and architecture
    DESIGN_REVIEW = "design_review"
    ARCHITECTURE_UPDATE = "architecture_update"

    # Error and debugging
    ERROR_DETECTED = "error_detected"
    ERROR_RESOLVED = "error_resolved"
    ERROR_PATTERN = "error_pattern"

    # Agent coordination
    TASK_REQUEST = "task_request"
    TASK_COMPLETED = "task_completed"
    AGENT_STATUS = "agent_status"

    # System control
    ITERATION_START = "iteration_start"
    ITERATION_END = "iteration_end"
    EMERGENCY_STOP = "emergency_stop"

    # Custom messages
    CUSTOM = "custom"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class AgentMessage:
    """Standard message format for agent communication"""
    id: str
    sender_agent: str
    recipient_agent: Optional[str]  # None for broadcast
    message_type: MessageType
    priority: MessagePriority
    payload: Dict[str, Any]
    correlation_id: Optional[str]  # For request-response patterns
    timestamp: datetime
    ttl_seconds: int = 300  # 5 minutes default TTL
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class MessageSubscription:
    """Subscription configuration for agents"""
    agent_id: str
    message_types: Set[MessageType]
    callback: Callable[[AgentMessage], None]
    queue_name: str


class AgentMessageBus:
    """
    Centralized message bus for agent communication

    Provides:
    - Event-driven messaging
    - Message routing and filtering
    - Guaranteed delivery
    - Message persistence
    - Dead letter handling
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.subscriptions: Dict[str, MessageSubscription] = {}
        self.running = False

        # Message statistics
        self.message_stats = {
            "sent": 0,
            "received": 0,
            "failed": 0,
            "retries": 0
        }

        logger.info("Agent Message Bus initialized")

    async def start(self):
        """Start the message bus"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            self.running = True

            # Start background tasks
            asyncio.create_task(self._message_processor())
            asyncio.create_task(self._retry_processor())
            asyncio.create_task(self._cleanup_processor())

            logger.info("Agent Message Bus started successfully")

        except Exception as e:
            logger.error(f"Failed to start message bus: {e}")
            raise

    async def stop(self):
        """Stop the message bus"""
        self.running = False
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Agent Message Bus stopped")

    async def subscribe(
        self,
        agent_id: str,
        message_types: List[MessageType],
        callback: Callable[[AgentMessage], None]
    ) -> bool:
        """
        Subscribe an agent to specific message types

        Args:
            agent_id: Unique identifier for the agent
            message_types: List of message types to subscribe to
            callback: Function to call when message is received

        Returns:
            bool: True if subscription successful
        """
        try:
            queue_name = f"agent:{agent_id}:queue"

            subscription = MessageSubscription(
                agent_id=agent_id,
                message_types=set(message_types),
                callback=callback,
                queue_name=queue_name
            )

            self.subscriptions[agent_id] = subscription

            # Create Redis stream for agent if it doesn't exist
            try:
                await self.redis_client.xgroup_create(
                    queue_name,
                    f"{agent_id}_group",
                    id="0",
                    mkstream=True
                )
            except redis.exceptions.ResponseError as e:
                if "BUSYGROUP" not in str(e):  # Group already exists
                    raise

            logger.info(f"Agent {agent_id} subscribed to {len(message_types)} message types")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe agent {agent_id}: {e}")
            return False

    async def unsubscribe(self, agent_id: str) -> bool:
        """Unsubscribe an agent from all message types"""
        try:
            if agent_id in self.subscriptions:
                del self.subscriptions[agent_id]

            # Clean up Redis resources
            queue_name = f"agent:{agent_id}:queue"
            try:
                await self.redis_client.xgroup_destroy(queue_name, f"{agent_id}_group")
            except redis.exceptions.ResponseError:
                pass  # Group might not exist

            logger.info(f"Agent {agent_id} unsubscribed")
            return True

        except Exception as e:
            logger.error(f"Failed to unsubscribe agent {agent_id}: {e}")
            return False

    async def send_message(
        self,
        sender_agent: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        recipient_agent: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        ttl_seconds: int = 300
    ) -> str:
        """
        Send a message through the bus

        Args:
            sender_agent: ID of the sending agent
            message_type: Type of message
            payload: Message payload
            recipient_agent: Specific recipient (None for broadcast)
            priority: Message priority
            correlation_id: For request-response correlation
            ttl_seconds: Message time to live

        Returns:
            str: Message ID
        """
        message_id = str(uuid.uuid4())

        message = AgentMessage(
            id=message_id,
            sender_agent=sender_agent,
            recipient_agent=recipient_agent,
            message_type=message_type,
            priority=priority,
            payload=payload,
            correlation_id=correlation_id,
            timestamp=datetime.now(),
            ttl_seconds=ttl_seconds
        )

        try:
            # Serialize message
            message_data = json.dumps(asdict(message), default=str)

            # Determine delivery strategy
            if recipient_agent:
                # Direct message to specific agent
                await self._deliver_to_agent(recipient_agent, message_data)
            else:
                # Broadcast to all subscribed agents
                await self._broadcast_message(message_type, message_data)

            # Store message for audit and retry
            await self._store_message(message_id, message_data)

            self.message_stats["sent"] += 1
            logger.debug(f"Message {message_id} sent from {sender_agent}")

            return message_id

        except Exception as e:
            logger.error(f"Failed to send message {message_id}: {e}")
            self.message_stats["failed"] += 1
            raise

    async def send_request(
        self,
        sender_agent: str,
        recipient_agent: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        timeout_seconds: int = 30
    ) -> Optional[AgentMessage]:
        """
        Send a request and wait for response

        Args:
            sender_agent: ID of the requesting agent
            recipient_agent: ID of the agent to request from
            message_type: Type of request
            payload: Request payload
            timeout_seconds: How long to wait for response

        Returns:
            AgentMessage: Response message or None if timeout
        """
        correlation_id = str(uuid.uuid4())

        # Send request
        await self.send_message(
            sender_agent=sender_agent,
            recipient_agent=recipient_agent,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id,
            priority=MessagePriority.HIGH
        )

        # Wait for response
        response_queue = f"response:{correlation_id}"

        try:
            # Wait for response with timeout
            response_data = await asyncio.wait_for(
                self.redis_client.brpop([response_queue], timeout=timeout_seconds),
                timeout=timeout_seconds
            )

            if response_data:
                _, message_json = response_data
                message_dict = json.loads(message_json)

                # Reconstruct AgentMessage
                message_dict["timestamp"] = datetime.fromisoformat(message_dict["timestamp"])
                message_dict["message_type"] = MessageType(message_dict["message_type"])
                message_dict["priority"] = MessagePriority(message_dict["priority"])

                return AgentMessage(**message_dict)

        except asyncio.TimeoutError:
            logger.warning(f"Request {correlation_id} timed out after {timeout_seconds}s")
        except Exception as e:
            logger.error(f"Failed to get response for {correlation_id}: {e}")

        return None

    async def send_response(
        self,
        sender_agent: str,
        original_message: AgentMessage,
        response_payload: Dict[str, Any]
    ) -> str:
        """
        Send a response to a request message

        Args:
            sender_agent: ID of the responding agent
            original_message: The original request message
            response_payload: Response data

        Returns:
            str: Response message ID
        """
        if not original_message.correlation_id:
            raise ValueError("Cannot respond to message without correlation_id")

        response_queue = f"response:{original_message.correlation_id}"

        response_message = AgentMessage(
            id=str(uuid.uuid4()),
            sender_agent=sender_agent,
            recipient_agent=original_message.sender_agent,
            message_type=original_message.message_type,
            priority=MessagePriority.HIGH,
            payload=response_payload,
            correlation_id=original_message.correlation_id,
            timestamp=datetime.now()
        )

        response_data = json.dumps(asdict(response_message), default=str)

        # Send response to correlation queue
        await self.redis_client.lpush(response_queue, response_data)

        # Set TTL on response queue
        await self.redis_client.expire(response_queue, 60)  # 1 minute TTL

        logger.debug(f"Response sent for correlation {original_message.correlation_id}")
        return response_message.id

    async def _deliver_to_agent(self, agent_id: str, message_data: str):
        """Deliver message to a specific agent"""
        queue_name = f"agent:{agent_id}:queue"

        try:
            # Add to Redis stream
            await self.redis_client.xadd(queue_name, {"data": message_data})
            logger.debug(f"Message delivered to agent {agent_id}")

        except Exception as e:
            logger.error(f"Failed to deliver message to agent {agent_id}: {e}")
            # Add to retry queue
            retry_queue = "message_bus:retry_queue"
            retry_data = {
                "agent_id": agent_id,
                "message_data": message_data,
                "retry_count": 0,
                "timestamp": time.time()
            }
            await self.redis_client.lpush(retry_queue, json.dumps(retry_data))

    async def _broadcast_message(self, message_type: MessageType, message_data: str):
        """Broadcast message to all subscribed agents"""
        interested_agents = [
            agent_id for agent_id, sub in self.subscriptions.items()
            if message_type in sub.message_types
        ]

        for agent_id in interested_agents:
            await self._deliver_to_agent(agent_id, message_data)

    async def _store_message(self, message_id: str, message_data: str):
        """Store message for audit and potential retry"""
        audit_key = f"message_bus:audit:{message_id}"
        await self.redis_client.setex(audit_key, 86400, message_data)  # 24 hour retention

    async def _message_processor(self):
        """Background task to process incoming messages"""
        while self.running:
            try:
                for agent_id, subscription in self.subscriptions.items():
                    try:
                        # Read messages from agent queue
                        messages = await self.redis_client.xreadgroup(
                            f"{agent_id}_group",
                            agent_id,
                            {subscription.queue_name: ">"},
                            count=10,
                            block=100
                        )

                        for stream, msgs in messages:
                            for msg_id, fields in msgs:
                                try:
                                    message_data = fields[b"data"].decode()
                                    message_dict = json.loads(message_data)

                                    # Reconstruct AgentMessage
                                    message_dict["timestamp"] = datetime.fromisoformat(message_dict["timestamp"])
                                    message_dict["message_type"] = MessageType(message_dict["message_type"])
                                    message_dict["priority"] = MessagePriority(message_dict["priority"])

                                    agent_message = AgentMessage(**message_dict)

                                    # Call agent callback
                                    await asyncio.get_event_loop().run_in_executor(
                                        None, subscription.callback, agent_message
                                    )

                                    # Acknowledge message
                                    await self.redis_client.xack(
                                        subscription.queue_name,
                                        f"{agent_id}_group",
                                        msg_id
                                    )

                                    self.message_stats["received"] += 1

                                except Exception as e:
                                    logger.error(f"Failed to process message {msg_id}: {e}")

                    except Exception as e:
                        if "NOGROUP" not in str(e):  # Expected error if no group
                            logger.error(f"Error processing messages for {agent_id}: {e}")

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Message processor error: {e}")
                await asyncio.sleep(1)

    async def _retry_processor(self):
        """Background task to retry failed message deliveries"""
        retry_queue = "message_bus:retry_queue"

        while self.running:
            try:
                # Get failed message for retry
                retry_data = await self.redis_client.brpop([retry_queue], timeout=5)

                if retry_data:
                    _, retry_json = retry_data
                    retry_info = json.loads(retry_json)

                    retry_count = retry_info["retry_count"]
                    if retry_count < 3:  # Max 3 retries
                        # Exponential backoff
                        delay = 2 ** retry_count
                        await asyncio.sleep(delay)

                        try:
                            await self._deliver_to_agent(
                                retry_info["agent_id"],
                                retry_info["message_data"]
                            )
                            logger.info(f"Retry successful for agent {retry_info['agent_id']}")

                        except Exception as e:
                            # Increment retry count and requeue
                            retry_info["retry_count"] = retry_count + 1
                            await self.redis_client.lpush(retry_queue, json.dumps(retry_info))
                            self.message_stats["retries"] += 1
                            logger.warning(f"Retry {retry_count + 1} failed for agent {retry_info['agent_id']}: {e}")
                    else:
                        # Move to dead letter queue
                        dead_letter_queue = "message_bus:dead_letter_queue"
                        await self.redis_client.lpush(dead_letter_queue, retry_json)
                        logger.error(f"Message moved to dead letter queue after max retries: {retry_info['agent_id']}")

            except Exception as e:
                logger.error(f"Retry processor error: {e}")
                await asyncio.sleep(1)

    async def _cleanup_processor(self):
        """Background task to clean up expired messages"""
        while self.running:
            try:
                # Clean up expired audit messages
                current_time = time.time()
                pattern = "message_bus:audit:*"

                async for key in self.redis_client.scan_iter(match=pattern, count=100):
                    try:
                        ttl = await self.redis_client.ttl(key)
                        if ttl == -1:  # No TTL set
                            await self.redis_client.expire(key, 86400)  # Set 24 hour TTL
                    except Exception:
                        pass

                # Sleep for 1 hour before next cleanup
                await asyncio.sleep(3600)

            except Exception as e:
                logger.error(f"Cleanup processor error: {e}")
                await asyncio.sleep(60)

    def get_statistics(self) -> Dict[str, Any]:
        """Get message bus statistics"""
        return {
            "running": self.running,
            "subscribed_agents": len(self.subscriptions),
            "message_stats": self.message_stats.copy(),
            "subscription_details": {
                agent_id: {
                    "message_types": [mt.value for mt in sub.message_types],
                    "queue_name": sub.queue_name
                }
                for agent_id, sub in self.subscriptions.items()
            }
        }