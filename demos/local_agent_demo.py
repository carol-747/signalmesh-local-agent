"""
Streamlit Demo App for SignalMesh Local Agent.

This demo application showcases the Local Agent's capabilities including
file scanning, content parsing, RAG indexing, and semantic search.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
from loguru import logger

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import LocalAgent


# Configure logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")


def init_session_state():
    """Initialize Streamlit session state."""
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "scan_results" not in st.session_state:
        st.session_state.scan_results = None
    if "search_results" not in st.session_state:
        st.session_state.search_results = None


def create_agent(workspace_path: str) -> LocalAgent:
    """
    Create or get cached LocalAgent instance.

    Args:
        workspace_path: Path to workspace

    Returns:
        LocalAgent instance
    """
    try:
        if (st.session_state.agent is None or
            str(st.session_state.agent.workspace_path) != workspace_path):
            st.session_state.agent = LocalAgent(workspace_path)
            logger.info(f"Created LocalAgent for workspace: {workspace_path}")
        return st.session_state.agent
    except Exception as e:
        st.error(f"Error creating agent: {e}")
        return None


async def run_scan(
    agent: LocalAgent,
    start_date: datetime,
    end_date: datetime,
    scan_all: bool
):
    """
    Run workspace scan asynchronously.

    Args:
        agent: LocalAgent instance
        start_date: Start date for scan
        end_date: End date for scan
        scan_all: Whether to scan all files
    """
    task_ticket = {
        "task_id": f"demo_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "workspace_path": str(agent.workspace_path),
        "start_date": start_date,
        "end_date": end_date,
        "scan_all": scan_all,
        "reindex": True
    }

    with st.spinner("Processing workspace..."):
        result = await agent.handle_task(task_ticket)
        st.session_state.scan_results = result


async def run_search(agent: LocalAgent, query: str, limit: int):
    """
    Run semantic search asynchronously.

    Args:
        agent: LocalAgent instance
        query: Search query
        limit: Maximum results
    """
    with st.spinner(f"Searching for: {query}..."):
        results = await agent.search_workspace(query, limit=limit)
        st.session_state.search_results = results


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="SignalMesh Local Agent Demo",
        page_icon="🤖",
        layout="wide"
    )

    init_session_state()

    st.title("🤖 SignalMesh Local Agent Demo")
    st.markdown("""
    This demo showcases the Local Agent's capabilities for workspace analysis
    and semantic search.
    """)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    workspace_path = st.sidebar.text_input(
        "Workspace Path",
        value=str(Path.cwd() / "data" / "simulated_workspace"),
        help="Path to the workspace directory to scan"
    )

    # Main content area
    tab1, tab2, tab3 = st.tabs(["📁 File Scanner", "🔍 Semantic Search", "📊 RAG Status"])

    # Tab 1: File Scanner
    with tab1:
        st.header("File Scanner & Indexer")

        col1, col2 = st.columns(2)

        with col1:
            scan_mode = st.radio(
                "Scan Mode",
                ["Scan All Files", "Scan by Date Range"],
                help="Choose whether to scan all files or filter by date"
            )

        with col2:
            if scan_mode == "Scan by Date Range":
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now().date()
                )
                start_date = st.date_input(
                    "Start Date",
                    value=(datetime.now() - timedelta(days=7)).date()
                )
            else:
                start_date = datetime.now().date()
                end_date = datetime.now().date()

        if st.button("🚀 Scan Workspace", type="primary"):
            agent = create_agent(workspace_path)
            if agent:
                scan_all = (scan_mode == "Scan All Files")
                asyncio.run(run_scan(
                    agent,
                    datetime.combine(start_date, datetime.min.time()),
                    datetime.combine(end_date, datetime.max.time()),
                    scan_all
                ))

        # Display scan results
        if st.session_state.scan_results:
            results = st.session_state.scan_results

            st.divider()

            # Status message
            if results["status"] == "success":
                st.success(results["message"])
            else:
                st.error(results["message"])

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Files Processed", results["files_processed"])
            with col2:
                st.metric("Files Indexed", results["files_indexed"])
            with col3:
                st.metric("Processing Time", f"{results['processing_time']:.2f}s")
            with col4:
                error_count = len(results.get("errors", []))
                st.metric("Errors", error_count)

            # Summary
            st.subheader("📝 Summary")
            st.text(results.get("summary", "No summary available"))

            # File list
            if results.get("file_list"):
                st.subheader("📄 Changed Files")

                # Create DataFrame for better display
                import pandas as pd
                df = pd.DataFrame(results["file_list"])

                # Format columns
                if "modified" in df.columns:
                    df["modified"] = pd.to_datetime(df["modified"]).dt.strftime("%Y-%m-%d %H:%M")
                if "size" in df.columns:
                    df["size_kb"] = (df["size"] / 1024).round(2)

                # Display table
                display_cols = ["name", "file_type", "modified", "size_kb"]
                display_cols = [col for col in display_cols if col in df.columns]

                st.dataframe(
                    df[display_cols],
                    use_container_width=True,
                    hide_index=True
                )

            # Errors
            if results.get("errors"):
                with st.expander("⚠️ View Errors"):
                    for error in results["errors"]:
                        st.warning(error)

    # Tab 2: Semantic Search
    with tab2:
        st.header("Semantic Search")

        st.markdown("""
        Search your workspace using natural language queries. The RAG system
        will find semantically similar content.
        """)

        col1, col2 = st.columns([3, 1])

        with col1:
            search_query = st.text_input(
                "Search Query",
                placeholder="e.g., machine learning code, data analysis notebook, etc.",
                help="Enter a natural language search query"
            )

        with col2:
            search_limit = st.number_input(
                "Max Results",
                min_value=1,
                max_value=20,
                value=5,
                help="Maximum number of results to return"
            )

        if st.button("🔍 Search", type="primary"):
            agent = create_agent(workspace_path)
            if agent and search_query:
                asyncio.run(run_search(agent, search_query, search_limit))
            elif not search_query:
                st.warning("Please enter a search query")

        # Display search results
        if st.session_state.search_results:
            results = st.session_state.search_results

            st.divider()

            if not results:
                st.info("No results found. Try scanning the workspace first.")
            else:
                st.success(f"Found {len(results)} results")

                for i, result in enumerate(results, 1):
                    with st.container():
                        st.markdown(f"### Result {i}")

                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown(f"**File:** `{result['file_path']}`")
                            st.markdown(f"**Score:** {result['score']:.4f}")

                        with col2:
                            metadata = result.get("metadata", {})
                            if metadata.get("file_type"):
                                st.badge(metadata["file_type"].upper())

                        # Content preview
                        if result.get("content_preview"):
                            with st.expander("Preview"):
                                st.text(result["content_preview"])

                        # Metadata
                        if metadata:
                            with st.expander("Metadata"):
                                st.json(metadata)

                        st.divider()

    # Tab 3: RAG Status
    with tab3:
        st.header("RAG System Status")

        if st.button("🔄 Refresh Status"):
            agent = create_agent(workspace_path)
            if agent:
                st.rerun()

        agent = create_agent(workspace_path)
        if agent:
            status = agent.get_rag_status()

            if "error" in status:
                st.error(f"Error getting RAG status: {status['error']}")
            else:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Collection Name", status.get("collection_name", "N/A"))
                with col2:
                    st.metric("Files Indexed", status.get("points_count", 0))
                with col3:
                    st.metric("Status", status.get("status", "unknown"))

                st.subheader("Details")
                st.json(status)

                # Clear collection button
                st.divider()
                st.warning("⚠️ Danger Zone")

                if st.button("🗑️ Clear All Data", type="secondary"):
                    if st.checkbox("I understand this will delete all indexed data"):
                        try:
                            agent.rag_manager.clear_collection()
                            st.success("Collection cleared successfully")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error clearing collection: {e}")

    # Footer
    st.sidebar.divider()
    st.sidebar.markdown("""
    ### About
    **SignalMesh Local Agent**

    A component of the SignalMesh multi-agent research operations system.

    Features:
    - 📁 Workspace file scanning
    - 🔍 Content parsing & analysis
    - 🧠 RAG-based semantic search
    - 💾 Local vector storage
    """)


if __name__ == "__main__":
    main()
