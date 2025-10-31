"""
Message Bus Simulation for SignalMesh Local Agent.

This script simulates how the Local Agent would integrate with a message bus
in the full SignalMesh system, demonstrating the publish/subscribe pattern
and inter-agent communication.

Prerequisites:
    pip install -r requirements.txt

Run with: python tests/test_message_bus_simulation.py
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List
from uuid import uuid4

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for required dependencies
try:
    from src.agent import LocalAgent
except ImportError as e:
    print("\n❌ Error: Required dependencies not installed!")
    print("\nPlease install dependencies first:")
    print("  cd signalmesh-local-agent")
    print("  pip install -r requirements.txt")
    print(f"\nOriginal error: {e}\n")
    sys.exit(1)


# ANSI color codes for beautiful output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class MockMessageBus:
    """
    Mock message bus simulating RabbitMQ/Redis pub-sub functionality.

    In production, this would be replaced with actual message bus client
    (e.g., RabbitMQ, Redis Pub/Sub, Apache Kafka, etc.).
    """

    def __init__(self):
        """Initialize the mock message bus."""
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_log: List[Dict[str, Any]] = []
        print(f"{Colors.OKGREEN}✓ MockMessageBus initialized{Colors.ENDC}")

    async def publish(self, topic: str, message: Dict[str, Any]) -> None:
        """
        Publish a message to a topic.

        Args:
            topic: Topic name (e.g., "local/task", "local/progress")
            message: Message payload
        """
        timestamp = datetime.now().isoformat()

        # Log the message
        log_entry = {
            "timestamp": timestamp,
            "topic": topic,
            "message": message,
            "message_id": str(uuid4())
        }
        self.message_log.append(log_entry)

        # Print message flow
        self._print_message("PUBLISH", topic, message)

        # Deliver to subscribers
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                # Simulate async delivery
                await asyncio.sleep(0.1)
                await callback(topic, message)

    def subscribe(self, topic: str, callback: Callable) -> None:
        """
        Subscribe to a topic.

        Args:
            topic: Topic name to subscribe to
            callback: Async callback function to handle messages
        """
        if topic not in self.subscribers:
            self.subscribers[topic] = []

        self.subscribers[topic].append(callback)
        print(f"{Colors.OKCYAN}📡 Subscribed to topic: {topic}{Colors.ENDC}")

    def _print_message(self, action: str, topic: str, message: Dict[str, Any]) -> None:
        """
        Print formatted message.

        Args:
            action: Action type (PUBLISH/RECEIVE)
            topic: Topic name
            message: Message content
        """
        print(f"\n{Colors.BOLD}{'─' * 80}{Colors.ENDC}")

        if action == "PUBLISH":
            icon = "📤"
            color = Colors.WARNING
        else:
            icon = "📥"
            color = Colors.OKCYAN

        print(f"{color}{Colors.BOLD}{icon} {action}: {topic}{Colors.ENDC}")

        # Format message
        message_str = json.dumps(message, indent=2, default=str)
        for line in message_str.split('\n'):
            print(f"  {line}")

        print(f"{Colors.BOLD}{'─' * 80}{Colors.ENDC}")

    def get_message_log(self) -> List[Dict[str, Any]]:
        """Get the complete message log."""
        return self.message_log


class LocalAgentWithBus:
    """
    Wrapper for LocalAgent that integrates with message bus.

    This demonstrates how the Local Agent would be deployed in the
    full SignalMesh system.
    """

    def __init__(self, agent: LocalAgent, message_bus: MockMessageBus):
        """
        Initialize the agent with message bus integration.

        Args:
            agent: LocalAgent instance
            message_bus: Message bus instance
        """
        self.agent = agent
        self.message_bus = message_bus
        self.agent_id = "local-agent-001"

        # Subscribe to task topic
        self.message_bus.subscribe("local/task", self._handle_task_message)

        print(f"{Colors.OKGREEN}✓ LocalAgentWithBus initialized (ID: {self.agent_id}){Colors.ENDC}")

    async def _handle_task_message(self, topic: str, message: Dict[str, Any]) -> None:
        """
        Handle incoming task messages.

        Args:
            topic: Topic the message was received on
            message: Message payload
        """
        print(f"\n{Colors.OKCYAN}🤖 Local Agent received task message{Colors.ENDC}")

        try:
            # Extract task details and context
            task_ticket = message.get("task_ticket", {})
            context = message.get("context", {})
            task_id = task_ticket.get("task_id", "unknown")

            # Publish progress: started
            await self.message_bus.publish("local/progress", {
                "task_id": task_id,
                "agent_id": self.agent_id,
                "status": "started",
                "message": "Local Agent started processing task",
                "timestamp": datetime.now().isoformat()
            })

            # Process task with context
            result = await self.agent.handle_task(task_ticket, context)

            # Publish progress: completed
            await self.message_bus.publish("local/progress", {
                "task_id": task_id,
                "agent_id": self.agent_id,
                "status": "completed",
                "message": "Local Agent completed processing",
                "result": result,
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            # Publish progress: error
            await self.message_bus.publish("local/progress", {
                "task_id": task_ticket.get("task_id", "unknown"),
                "agent_id": self.agent_id,
                "status": "error",
                "message": f"Error processing task: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })


class MockReplyAgent:
    """
    Mock Reply Agent that receives Local Agent results and drafts emails.

    In production, this would be a full Reply Agent with LLM capabilities
    for drafting human-friendly responses.
    """

    def __init__(self, message_bus: MockMessageBus):
        """
        Initialize the Reply Agent.

        Args:
            message_bus: Message bus instance
        """
        self.message_bus = message_bus
        self.agent_id = "reply-agent-001"

        # Subscribe to progress topic
        self.message_bus.subscribe("local/progress", self._handle_progress_message)

        print(f"{Colors.OKGREEN}✓ MockReplyAgent initialized (ID: {self.agent_id}){Colors.ENDC}")

    async def _handle_progress_message(self, topic: str, message: Dict[str, Any]) -> None:
        """
        Handle progress messages from Local Agent.

        Args:
            topic: Topic the message was received on
            message: Message payload
        """
        status = message.get("status")

        if status == "started":
            print(f"\n{Colors.OKCYAN}📧 Reply Agent: Task started, waiting for completion...{Colors.ENDC}")

        elif status == "completed":
            print(f"\n{Colors.OKGREEN}📧 Reply Agent: Task completed, drafting email...{Colors.ENDC}")

            # Extract result
            result = message.get("result", {})

            # Draft email
            email = self._draft_email(result)

            # Publish draft email
            await self.message_bus.publish("reply/draft", {
                "task_id": result.get("task_id", "unknown"),
                "agent_id": self.agent_id,
                "email_draft": email,
                "timestamp": datetime.now().isoformat()
            })

        elif status == "error":
            print(f"\n{Colors.FAIL}📧 Reply Agent: Error occurred, drafting error notification...{Colors.ENDC}")

    def _draft_email(self, result: Dict[str, Any]) -> Dict[str, str]:
        """
        Draft an email based on Local Agent results.

        In production, this would use an LLM to generate human-friendly content.

        Args:
            result: Task result from Local Agent

        Returns:
            Dictionary with email fields
        """
        files_processed = result.get("files_processed", 0)
        summary = result.get("summary", "No summary available")
        processing_time = result.get("processing_time", 0)

        # Simple template-based email (in production, use LLM)
        subject = f"Workspace Activity Summary - {files_processed} Files Analyzed"

        body = f"""Hi there,

I've completed the analysis of your workspace. Here's what I found:

SUMMARY
{'-' * 60}
{summary}

PROCESSING DETAILS
{'-' * 60}
• Files Analyzed: {files_processed}
• Processing Time: {processing_time:.2f} seconds
• Status: Successfully completed

"""

        if result.get("file_list"):
            body += "\nFILES PROCESSED\n"
            body += "-" * 60 + "\n"
            for i, file_info in enumerate(result["file_list"][:10], 1):  # Limit to 10
                name = file_info.get("name", "unknown")
                file_type = file_info.get("file_type", "unknown")
                body += f"{i}. {name} ({file_type})\n"

            if len(result["file_list"]) > 10:
                body += f"... and {len(result['file_list']) - 10} more files\n"

        body += f"""
If you need more details or have questions about any of the files,
just let me know!

Best regards,
SignalMesh Local Agent
"""

        return {
            "subject": subject,
            "body": body,
            "to": "requester@example.com",
            "from": "signalmesh@example.com"
        }


class MockEmailPublisher:
    """
    Mock email publisher that displays draft emails.

    In production, this would send actual emails or create drafts
    in an email system.
    """

    def __init__(self, message_bus: MockMessageBus):
        """
        Initialize the email publisher.

        Args:
            message_bus: Message bus instance
        """
        self.message_bus = message_bus

        # Subscribe to draft topic
        self.message_bus.subscribe("reply/draft", self._handle_draft_message)

        print(f"{Colors.OKGREEN}✓ MockEmailPublisher initialized{Colors.ENDC}")

    async def _handle_draft_message(self, topic: str, message: Dict[str, Any]) -> None:
        """
        Handle draft email messages.

        Args:
            topic: Topic the message was received on
            message: Message payload
        """
        email_draft = message.get("email_draft", {})

        print(f"\n{Colors.OKGREEN}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}{Colors.BOLD}📧 DRAFT EMAIL READY{Colors.ENDC}")
        print(f"{Colors.OKGREEN}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")

        print(f"{Colors.BOLD}From:{Colors.ENDC} {email_draft.get('from', 'N/A')}")
        print(f"{Colors.BOLD}To:{Colors.ENDC} {email_draft.get('to', 'N/A')}")
        print(f"{Colors.BOLD}Subject:{Colors.ENDC} {email_draft.get('subject', 'N/A')}\n")

        print(f"{Colors.OKCYAN}{'─' * 78}{Colors.ENDC}")
        body_lines = email_draft.get('body', '').split('\n')
        for line in body_lines:
            print(f"  {line}")
        print(f"{Colors.OKCYAN}{'─' * 78}{Colors.ENDC}\n")


def print_header(text: str) -> None:
    """Print a formatted header."""
    width = 80
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * width}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(width)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * width}{Colors.ENDC}\n")


def print_flow_visualization() -> None:
    """Print a visual representation of the message flow."""
    print(f"\n{Colors.BOLD}📊 MESSAGE FLOW VISUALIZATION{Colors.ENDC}\n")

    flow = """
    ┌─────────────────┐
    │  User Request   │  (Simulated)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Message Bus    │  Topic: local/task
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Local Agent    │  Processes task
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Message Bus    │  Topic: local/progress
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Reply Agent    │  Drafts email
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Message Bus    │  Topic: reply/draft
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Email System   │  Displays draft
    └─────────────────┘
    """

    for line in flow.split('\n'):
        print(f"{Colors.OKCYAN}{line}{Colors.ENDC}")

    print()


async def run_simulation() -> None:
    """Run the complete message bus simulation."""
    print_header("SignalMesh Message Bus Simulation")

    print(f"{Colors.BOLD}This simulation demonstrates how the Local Agent integrates{Colors.ENDC}")
    print(f"{Colors.BOLD}with other SignalMesh components via a message bus.{Colors.ENDC}\n")

    print_flow_visualization()

    # Initialize components
    print(f"{Colors.BOLD}🚀 Initializing Components{Colors.ENDC}\n")

    # 1. Message Bus
    message_bus = MockMessageBus()

    # 2. Local Agent
    workspace_path = Path(__file__).parent.parent / "data" / "simulated_workspace"
    agent = LocalAgent(
        workspace_path=str(workspace_path),
        rag_storage_path="./data/qdrant_storage"
    )
    local_agent_with_bus = LocalAgentWithBus(agent, message_bus)

    # 3. Reply Agent
    reply_agent = MockReplyAgent(message_bus)

    # 4. Email Publisher
    email_publisher = MockEmailPublisher(message_bus)

    print(f"\n{Colors.OKGREEN}✓ All components initialized{Colors.ENDC}\n")

    await asyncio.sleep(1)

    # Create task message
    print_header("SIMULATION START")

    print(f"{Colors.BOLD}👤 Simulating User Request{Colors.ENDC}\n")
    print(f"  User: Sarah Chen (Team Lead)")
    print(f"  Question: 'Can you give me a summary of what the team worked on this week?'")
    print(f"  Channel: Email\n")

    await asyncio.sleep(1)

    # Publish task to message bus
    print(f"{Colors.BOLD}📤 Publishing task to message bus...{Colors.ENDC}\n")

    task_message = {
        "task_ticket": {
            "task_id": "DEMO-TASK-001",
            "workspace_path": str(workspace_path),
            "start_date": datetime.now() - timedelta(days=7),
            "end_date": datetime.now(),
            "scan_all": False,
            "reindex": True
        },
        "context": {
            "requester": "Sarah Chen",
            "requester_email": "sarah.chen@example.com",
            "original_question": "Can you give me a summary of what the team worked on this week?",
            "channel": "Email",
            "urgency": "Normal"
        },
        "timestamp": datetime.now().isoformat()
    }

    await message_bus.publish("local/task", task_message)

    # Wait for processing
    print(f"\n{Colors.WARNING}⏳ Waiting for agents to process...{Colors.ENDC}\n")
    await asyncio.sleep(2)

    # Show message log
    print_header("MESSAGE LOG")

    print(f"{Colors.BOLD}📋 Complete Message Flow{Colors.ENDC}\n")

    message_log = message_bus.get_message_log()
    for i, log_entry in enumerate(message_log, 1):
        topic = log_entry["topic"]
        timestamp = log_entry["timestamp"]
        message_id = log_entry["message_id"][:8]

        print(f"  {i}. [{timestamp}]")
        print(f"     Topic: {topic}")
        print(f"     Message ID: {message_id}...")

        # Show key info
        msg = log_entry["message"]
        if "status" in msg:
            print(f"     Status: {msg['status']}")

        print()

    # Statistics
    print_header("SIMULATION STATISTICS")

    print(f"{Colors.BOLD}📊 Message Bus Statistics{Colors.ENDC}\n")
    print(f"  Total Messages: {len(message_log)}")

    topic_counts = {}
    for entry in message_log:
        topic = entry["topic"]
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    print(f"  Messages by Topic:")
    for topic, count in sorted(topic_counts.items()):
        print(f"    • {topic}: {count}")

    print()

    print(f"{Colors.BOLD}🔗 Component Interactions{Colors.ENDC}\n")
    print(f"  Local Agent → Message Bus: Published progress updates")
    print(f"  Reply Agent → Message Bus: Published email draft")
    print(f"  Email Publisher: Displayed final draft")

    print()

    # Integration notes
    print(f"{Colors.BOLD}💡 Production Deployment Notes{Colors.ENDC}\n")
    print("  In a production SignalMesh deployment:")
    print("  1. Replace MockMessageBus with RabbitMQ, Redis, or Kafka")
    print("  2. Deploy agents as separate services/containers")
    print("  3. Add message persistence and retry logic")
    print("  4. Implement authentication and authorization")
    print("  5. Add monitoring and observability (Prometheus, Grafana)")
    print("  6. Use actual LLM for Reply Agent email generation")
    print("  7. Integrate with real email system (SendGrid, SES, etc.)")
    print("  8. Add circuit breakers and rate limiting")

    print()

    print_header("SIMULATION COMPLETE")


def main():
    """Main entry point."""
    try:
        asyncio.run(run_simulation())
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Simulation interrupted by user{Colors.ENDC}\n")
    except Exception as e:
        print(f"\n\n{Colors.FAIL}Simulation failed with error: {e}{Colors.ENDC}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
