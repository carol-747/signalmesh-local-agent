"""
End-to-End Pipeline Test for SignalMesh Local Agent.

This script demonstrates how the Local Agent processes different types of task tickets
in an end-to-end workflow, simulating real-world usage scenarios.

Prerequisites:
    pip install -r requirements.txt

Run with: python tests/test_pipeline.py
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

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


def print_header(text: str) -> None:
    """Print a formatted header."""
    width = 80
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * width}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(width)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * width}{Colors.ENDC}\n")


def print_section(text: str) -> None:
    """Print a formatted section header."""
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}{'─' * 80}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{Colors.BOLD}📋 {text}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{Colors.BOLD}{'─' * 80}{Colors.ENDC}\n")


def print_info(label: str, value: Any) -> None:
    """Print formatted info line."""
    print(f"{Colors.OKBLUE}  {label}:{Colors.ENDC} {value}")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


class TestScenario:
    """Represents a test scenario with context."""

    def __init__(
        self,
        ticket_id: str,
        name: str,
        description: str,
        workspace_path: str,
        date_range: Dict[str, Any],
        context: Dict[str, str]
    ):
        self.ticket_id = ticket_id
        self.name = name
        self.description = description
        self.workspace_path = workspace_path
        self.date_range = date_range
        self.context = context

    def to_task_ticket(self) -> Dict[str, Any]:
        """Convert scenario to task ticket format."""
        return {
            "task_id": self.ticket_id,
            "workspace_path": self.workspace_path,
            **self.date_range
        }


def create_test_scenarios() -> List[TestScenario]:
    """
    Create test scenarios simulating different types of workspace scans.

    Returns:
        List of TestScenario objects
    """
    workspace_path = str(Path(__file__).parent.parent / "data" / "simulated_workspace")
    now = datetime.now()

    scenarios = [
        TestScenario(
            ticket_id="TICKET-001",
            name="Weekly Update Request",
            description="Team lead requests weekly activity summary",
            workspace_path=workspace_path,
            date_range={
                "start_date": now - timedelta(days=7),
                "end_date": now,
                "scan_all": False,
                "reindex": True
            },
            context={
                "requester": "Sarah Chen (Team Lead)",
                "requester_email": "sarah.chen@example.com",
                "original_question": "Can you give me a summary of what the team worked on this week?",
                "urgency": "Normal",
                "channel": "Email"
            }
        ),
        TestScenario(
            ticket_id="TICKET-002",
            name="Recent Changes Check",
            description="Developer needs context on last 3 days of changes",
            workspace_path=workspace_path,
            date_range={
                "start_date": now - timedelta(days=3),
                "end_date": now,
                "scan_all": False,
                "reindex": True
            },
            context={
                "requester": "Mike Rodriguez (Senior Developer)",
                "requester_email": "mike.rodriguez@example.com",
                "original_question": "I've been on vacation for 3 days. What did I miss?",
                "urgency": "High",
                "channel": "Slack"
            }
        ),
        TestScenario(
            ticket_id="TICKET-003",
            name="Full Workspace Index",
            description="Initial workspace scan for new project member",
            workspace_path=workspace_path,
            date_range={
                "scan_all": True,
                "reindex": True
            },
            context={
                "requester": "Alex Kim (New Team Member)",
                "requester_email": "alex.kim@example.com",
                "original_question": "I just joined the team. Can you help me understand what's in our workspace?",
                "urgency": "Normal",
                "channel": "Email"
            }
        )
    ]

    return scenarios


def print_input_message(scenario: TestScenario) -> None:
    """
    Print formatted input message showing the incoming request.

    Args:
        scenario: Test scenario to display
    """
    print_section(f"INPUT MESSAGE - {scenario.name}")

    print(f"{Colors.BOLD}📨 Incoming Request{Colors.ENDC}")
    print_info("Ticket ID", scenario.ticket_id)
    print_info("Description", scenario.description)
    print()

    print(f"{Colors.BOLD}👤 Requester Context{Colors.ENDC}")
    print_info("Name", scenario.context["requester"])
    print_info("Email", scenario.context["requester_email"])
    print_info("Channel", scenario.context["channel"])
    print_info("Urgency", scenario.context["urgency"])
    print()

    print(f"{Colors.BOLD}💬 Original Question{Colors.ENDC}")
    print(f'  "{scenario.context["original_question"]}"')
    print()

    print(f"{Colors.BOLD}⚙️ Task Parameters{Colors.ENDC}")
    print_info("Workspace", scenario.workspace_path)

    if scenario.date_range.get("scan_all"):
        print_info("Scan Mode", "Full workspace scan (all files)")
    else:
        start = scenario.date_range["start_date"]
        end = scenario.date_range["end_date"]
        print_info("Scan Mode", "Date range filter")
        print_info("Start Date", start.strftime("%Y-%m-%d %H:%M:%S"))
        print_info("End Date", end.strftime("%Y-%m-%d %H:%M:%S"))
        days = (end - start).days
        print_info("Date Range", f"{days} days")

    print_info("Reindex RAG", "Yes" if scenario.date_range.get("reindex") else "No")


def print_processing_status() -> None:
    """Print processing status indicator."""
    print(f"\n{Colors.WARNING}⚙️  Processing request...{Colors.ENDC}")
    print(f"{Colors.WARNING}   [1/4] Scanning workspace for files...{Colors.ENDC}")
    print(f"{Colors.WARNING}   [2/4] Parsing file contents...{Colors.ENDC}")
    print(f"{Colors.WARNING}   [3/4] Indexing in RAG system...{Colors.ENDC}")
    print(f"{Colors.WARNING}   [4/4] Generating summary...{Colors.ENDC}\n")


def print_output_message(result: Dict[str, Any], scenario: TestScenario) -> None:
    """
    Print formatted output message showing the agent's response.

    Args:
        result: Task result from LocalAgent
        scenario: Original test scenario
    """
    print_section(f"OUTPUT MESSAGE - {scenario.name}")

    # Status
    status = result.get("status", "unknown")
    if status == "success":
        print_success(f"Task completed successfully")
    else:
        print_error(f"Task failed: {result.get('message', 'Unknown error')}")

    print()

    # Statistics
    print(f"{Colors.BOLD}📊 Processing Statistics{Colors.ENDC}")
    stats = result.get("results", {}).get("statistics", {})
    perf = result.get("performance", {})

    print_info("Files Scanned", stats.get("files_scanned", result.get("files_processed", 0)))
    print_info("Files Indexed", stats.get("files_indexed", result.get("files_indexed", 0)))
    print_info("Processing Time", f"{perf.get('processing_time_seconds', result.get('processing_time', 0)):.2f} seconds")
    print_info("Errors Encountered", len(result.get("errors", [])))
    print()

    # File breakdown
    files_list = result.get("results", {}).get("files", result.get("file_list", []))
    breakdown = result.get("results", {}).get("breakdown_by_type", {})

    if breakdown:
        print(f"{Colors.BOLD}📁 Files by Type{Colors.ENDC}")

        for file_type, type_info in sorted(breakdown.items()):
            emoji = {
                "code": "🐍",
                "data": "📊",
                "notebook": "📓",
                "note": "📝"
            }.get(file_type, "📄")
            count = type_info.get("count", 0)
            print(f"  {emoji} {file_type.capitalize()}: {count} file(s)")

            # Show type-specific details
            if file_type == "code" and "total_lines" in type_info:
                print(f"     Lines: {type_info['total_lines']}, Functions: {type_info.get('functions', 0)}, Classes: {type_info.get('classes', 0)}")
            elif file_type == "data" and "total_rows" in type_info:
                print(f"     Rows: {type_info['total_rows']}, Columns: {type_info.get('total_columns', 0)}")

        print()

    # List files
    if files_list:
        print(f"{Colors.BOLD}📄 File Details{Colors.ENDC}")
        for i, file_info in enumerate(files_list[:10], 1):  # Limit to 10
            name = file_info.get("name", "unknown")
            file_type = file_info.get("type", file_info.get("file_type", "other"))
            size_kb = file_info.get("size_bytes", file_info.get("size", 0)) / 1024
            print(f"  {i}. {name} ({file_type}, {size_kb:.1f} KB)")

        if len(files_list) > 10:
            print(f"  ... and {len(files_list) - 10} more files")

        print()

    # Summary
    summary = result.get("results", {}).get("summary", result.get("summary", ""))
    if summary:
        print(f"{Colors.BOLD}📝 Generated Summary{Colors.ENDC}")
        print(f"{Colors.OKCYAN}{'─' * 78}{Colors.ENDC}")
        summary_lines = summary.split("\n")
        for line in summary_lines:
            print(f"  {line}")
        print(f"{Colors.OKCYAN}{'─' * 78}{Colors.ENDC}")
        print()

    # Insights
    insights = result.get("insights", {})
    if insights and insights.get("highlights"):
        print(f"{Colors.BOLD}💡 Key Insights{Colors.ENDC}")
        for highlight in insights.get("highlights", [])[:5]:
            print(f"  • {highlight}")
        print()

    # Errors
    if result.get("errors"):
        print(f"{Colors.BOLD}⚠️  Errors{Colors.ENDC}")
        for error in result["errors"]:
            print_warning(error)
        print()


async def run_scenario(agent: LocalAgent, scenario: TestScenario) -> Dict[str, Any]:
    """
    Run a single test scenario.

    Args:
        agent: LocalAgent instance
        scenario: Test scenario to run

    Returns:
        Task result dictionary
    """
    print_header(f"SCENARIO: {scenario.name}")

    # Show input
    print_input_message(scenario)

    # Show processing
    print_processing_status()

    # Execute task with context
    task_ticket = scenario.to_task_ticket()
    result = await agent.handle_task(task_ticket, scenario.context)

    # Show output
    print_output_message(result, scenario)

    return result


async def run_all_scenarios() -> None:
    """Run all test scenarios and display results."""
    print_header("SignalMesh Local Agent - End-to-End Pipeline Test")

    print(f"{Colors.BOLD}This test demonstrates how the Local Agent processes different types{Colors.ENDC}")
    print(f"{Colors.BOLD}of task tickets in a realistic workflow.{Colors.ENDC}\n")

    # Initialize agent
    workspace_path = Path(__file__).parent.parent / "data" / "simulated_workspace"
    print(f"🚀 Initializing Local Agent...")
    print_info("Workspace", str(workspace_path))
    print_info("RAG Storage", "./data/qdrant_storage")
    print()

    try:
        agent = LocalAgent(
            workspace_path=str(workspace_path),
            rag_storage_path="./data/qdrant_storage"
        )
        print_success("Agent initialized successfully\n")
    except Exception as e:
        print_error(f"Failed to initialize agent: {e}")
        return

    # Create scenarios
    scenarios = create_test_scenarios()
    print(f"📋 Created {len(scenarios)} test scenarios\n")

    # Run each scenario
    results = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{Colors.BOLD}Running scenario {i}/{len(scenarios)}...{Colors.ENDC}\n")
        result = await run_scenario(agent, scenario)
        results.append((scenario, result))

        # Pause between scenarios for readability
        if i < len(scenarios):
            print(f"\n{Colors.OKCYAN}{'═' * 80}{Colors.ENDC}\n")
            await asyncio.sleep(0.5)

    # Final summary
    print_header("OVERALL SUMMARY")

    print(f"{Colors.BOLD}📊 Aggregate Statistics{Colors.ENDC}\n")

    total_files = sum(r.get("files_processed", 0) for _, r in results)
    total_indexed = sum(r.get("files_indexed", 0) for _, r in results)
    total_time = sum(r.get("processing_time", 0) for _, r in results)
    total_errors = sum(len(r.get("errors", [])) for _, r in results)
    successful = sum(1 for _, r in results if r.get("status") == "success")

    print_info("Total Scenarios Executed", len(scenarios))
    print_info("Successful Executions", f"{successful}/{len(scenarios)}")
    print_info("Total Files Processed", total_files)
    print_info("Total Files Indexed", total_indexed)
    print_info("Total Processing Time", f"{total_time:.2f} seconds")
    print_info("Average Time per Scenario", f"{total_time/len(scenarios):.2f} seconds")
    print_info("Total Errors", total_errors)
    print()

    # Per-scenario summary
    print(f"{Colors.BOLD}📋 Scenario Results{Colors.ENDC}\n")
    for i, (scenario, result) in enumerate(results, 1):
        status = result.get("status", "unknown")
        if status == "success":
            status_icon = f"{Colors.OKGREEN}✓{Colors.ENDC}"
        else:
            status_icon = f"{Colors.FAIL}✗{Colors.ENDC}"

        files = result.get("files_processed", 0)
        time = result.get("processing_time", 0)

        print(f"  {status_icon} {scenario.name}")
        print(f"     {scenario.ticket_id} | {files} files | {time:.2f}s")

    print()

    # RAG Status
    print(f"{Colors.BOLD}🧠 RAG System Status{Colors.ENDC}\n")
    try:
        rag_status = agent.get_rag_status()
        print_info("Collection", rag_status.get("collection_name", "N/A"))
        print_info("Total Indexed Files", rag_status.get("points_count", 0))
        print_info("Status", rag_status.get("status", "unknown"))
    except Exception as e:
        print_warning(f"Could not retrieve RAG status: {e}")

    print()

    # Integration notes
    print(f"{Colors.BOLD}🔗 SignalMesh Integration Notes{Colors.ENDC}\n")
    print("  In a full SignalMesh deployment, this Local Agent would:")
    print("  1. Subscribe to task messages on a message bus (e.g., RabbitMQ)")
    print("  2. Process tasks asynchronously with these same handlers")
    print("  3. Publish progress updates and results back to the bus")
    print("  4. Enable the Reply Agent to draft responses to requesters")
    print("  5. Coordinate with other agents (Research Agent, Web Agent, etc.)")
    print()

    print_header("TEST COMPLETE")


def main():
    """Main entry point."""
    try:
        asyncio.run(run_all_scenarios())
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Test interrupted by user{Colors.ENDC}\n")
    except Exception as e:
        print(f"\n\n{Colors.FAIL}Test failed with error: {e}{Colors.ENDC}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
