# Local Agent Integration Guide for Reply Agent

## Overview

This guide helps the Reply Agent team integrate with the Local Agent's comprehensive JSON output to draft personalized email responses.

## Quick Start

### What You Need to Know

The Local Agent processes workspace files and returns a **comprehensive JSON** with everything needed to draft intelligent replies:

- ✅ **Original requester context** (who asked, what they asked, how they asked)
- ✅ **Processed file results** with detailed analysis
- ✅ **Key insights and highlights** to feature in responses
- ✅ **Suggested follow-up actions** to offer users
- ✅ **Performance metrics** and error handling

## Output Schema

### Full JSON Structure

```json
{
  // ═══════════════════════════════════════════════════════════
  // TASK IDENTIFICATION
  // ═══════════════════════════════════════════════════════════
  "task_id": "DEMO-TASK-001",
  "agent_id": "local-agent-001",
  "timestamp": "2025-10-31T15:33:52.125283",
  "status": "success",  // or "error"
  "message": "Successfully processed 4 files",

  // ═══════════════════════════════════════════════════════════
  // ORIGINAL REQUEST CONTEXT (Critical for Reply Agent!)
  // ═══════════════════════════════════════════════════════════
  "context": {
    "requester": {
      "name": "Sarah Chen",                    // ← Use in email greeting
      "email": "sarah.chen@example.com",       // ← Send reply to
      "role": "Team Lead",                     // ← Adjust tone
      "timezone": "America/New_York"           // ← Time formatting
    },
    "request": {
      "original_question": "Can you give me a summary...",  // ← Reference in reply
      "channel": "Email",                      // ← Format accordingly
      "thread_id": "email-thread-12345",       // ← For threading
      "urgency": "Normal",                     // ← Prioritize response
      "requested_at": "2025-10-31T10:15:00Z",
      "requested_format": "summary"            // ← summary, detailed, list
    },
    "scope": {
      "time_range": "last_7_days",
      "start_date": "2025-10-24T15:33:51Z",
      "end_date": "2025-10-31T15:33:51Z"
    }
  },

  // ═══════════════════════════════════════════════════════════
  // PROCESSING RESULTS (Content for email body)
  // ═══════════════════════════════════════════════════════════
  "results": {
    "summary": "Processed 4 files from workspace...",  // ← Use as overview
    "statistics": {
      "files_processed": 4,
      "files_indexed": 4,
      "total_size_mb": 0.005
    },
    "breakdown_by_type": {
      "code": {
        "count": 1,
        "languages": ["python"],
        "total_lines": 67,
        "functions": 5,
        "classes": 1
      },
      // ... other types
    },
    "files": [
      {
        "path": "/workspace/sample_analysis.py",
        "name": "sample_analysis.py",
        "type": "code",
        "analysis": {
          "summary": "Python file with 1 class(es), 5 function(s)",
          "tags": ["analysis", "pandas", "data-processing"],  // ← Use for context
          "complexity": "medium"
        }
      }
    ]
  },

  // ═══════════════════════════════════════════════════════════
  // KEY INSIGHTS (Feature these in email!)
  // ═══════════════════════════════════════════════════════════
  "insights": {
    "highlights": [                            // ← Bullet points for email
      "New code added (sample_analysis.py)",
      "Data file updated (sample_data.csv)",
      "Analysis notebook created (sample_notebook.ipynb)"
    ],
    "activity_summary": "Active development with code changes",  // ← Lead paragraph
    "notable_changes": [                       // ← Important items to call out
      {
        "type": "code",
        "file": "sample_analysis.py",
        "description": "...",
        "impact": "high"
      }
    ],
    "recommendations": [                       // ← Action items for user
      "Review sample_analysis.py implementation",
      "Execute sample_notebook.ipynb to generate visualizations"
    ]
  },

  // ═══════════════════════════════════════════════════════════
  // SUGGESTED ACTIONS (Offer to user in email)
  // ═══════════════════════════════════════════════════════════
  "suggested_actions": [
    {
      "action": "semantic_search",
      "description": "Search for specific topics in the workspace",
      "example": "Find all code related to 'data processing'"
    },
    {
      "action": "deep_dive",
      "description": "Get detailed analysis of specific code file",
      "example": "Analyze sample_analysis.py in detail"
    }
  ],

  // ═══════════════════════════════════════════════════════════
  // PERFORMANCE & ERRORS
  // ═══════════════════════════════════════════════════════════
  "performance": {
    "processing_time_seconds": 0.261843
  },
  "errors": [],                                // ← Check before drafting
  "warnings": [],

  // ═══════════════════════════════════════════════════════════
  // METADATA
  // ═══════════════════════════════════════════════════════════
  "metadata": {
    "agent_version": "0.1.0",
    "schema_version": "1.0",
    "reply_agent_compatible": true,            // ← Always check this
    "requires_user_action": false,
    "is_final": true
  }
}
```

## Integration Options

### Option 1: Message Bus (Recommended for Production)

**Subscribe to topic:** `local/progress`

```python
# Your Reply Agent subscribes to results
message_bus.subscribe("local/progress", handle_local_agent_results)

async def handle_local_agent_results(topic, message):
    if message.get("status") == "completed":
        result = message.get("result", {})

        # Extract key fields
        context = result.get("context", {})
        requester = context.get("requester", {})
        insights = result.get("insights", {})

        # Draft email
        email = draft_email(
            to=requester.get("email"),
            name=requester.get("name"),
            original_question=context.get("request", {}).get("original_question"),
            highlights=insights.get("highlights", []),
            summary=result.get("results", {}).get("summary"),
            suggestions=result.get("suggested_actions", [])
        )

        # Publish draft
        await message_bus.publish("reply/draft", email)
```

**Topics:**
- Subscribe: `local/progress` (Local Agent results)
- Publish: `reply/draft` (Your email drafts)

### Option 2: Direct Python Import (For Testing/Development)

```python
import asyncio
from datetime import datetime, timedelta
from src.agent import LocalAgent

# Initialize
agent = LocalAgent(workspace_path="/path/to/workspace")

# Create task with context
task_ticket = {
    "task_id": "task_001",
    "workspace_path": "/path/to/workspace",
    "start_date": datetime.now() - timedelta(days=7),
    "end_date": datetime.now(),
    "scan_all": False,
    "reindex": True
}

context = {
    "requester": "Sarah Chen",
    "requester_email": "sarah.chen@example.com",
    "original_question": "What changed this week?",
    "channel": "Email",
    "urgency": "Normal"
}

# Process
result = asyncio.run(agent.handle_task(task_ticket, context))

# result is the comprehensive JSON
```

### Option 3: REST API (Future)

*Not yet implemented - would wrap LocalAgent in FastAPI/Flask*

## Email Drafting Recommendations

### 1. Extract Key Information

```python
def extract_email_data(result):
    """Extract data needed for email."""
    context = result.get("context", {})
    requester = context.get("requester", {})
    request = context.get("request", {})
    insights = result.get("insights", {})
    results = result.get("results", {})

    return {
        "to": requester.get("email", "unknown@example.com"),
        "name": requester.get("name", "there"),
        "subject": generate_subject(results.get("statistics", {})),
        "original_question": request.get("original_question", ""),
        "activity_summary": insights.get("activity_summary", ""),
        "highlights": insights.get("highlights", []),
        "recommendations": insights.get("recommendations", []),
        "suggested_actions": result.get("suggested_actions", []),
        "urgency": request.get("urgency", "Normal"),
        "channel": request.get("channel", "Email")
    }
```

### 2. Draft Email Template

```python
def draft_email(data):
    """Draft email from extracted data."""

    # Adjust tone based on urgency
    greeting = f"Hi {data['name']}," if data['urgency'] != 'High' else f"Hi {data['name']} (urgent),"

    # Build email
    email = f"""{greeting}

You asked: "{data['original_question']}"

Here's what I found:

{data['activity_summary']}

KEY HIGHLIGHTS:
{format_highlights(data['highlights'])}

RECOMMENDATIONS:
{format_recommendations(data['recommendations'])}

WHAT YOU CAN DO NEXT:
{format_suggestions(data['suggested_actions'])}

Let me know if you need more details!

Best,
SignalMesh Local Agent
"""

    return {
        "to": data['to'],
        "subject": data['subject'],
        "body": email,
        "thread_id": data.get('thread_id'),
        "urgency": data['urgency']
    }
```

### 3. Format Based on Channel

```python
def format_for_channel(email_data, channel):
    """Format differently for Email vs Slack."""

    if channel == "Slack":
        # Use Slack markdown
        return format_slack_message(email_data)
    elif channel == "Email":
        # Use email formatting
        return format_email_message(email_data)
    elif channel == "Teams":
        # Use Teams markdown
        return format_teams_message(email_data)
```

## Error Handling

### Check for Errors

```python
def handle_result(result):
    """Handle result with error checking."""

    # Check status
    if result.get("status") != "success":
        return draft_error_email(result)

    # Check for processing errors
    errors = result.get("errors", [])
    if errors:
        # Mention errors in email
        return draft_partial_success_email(result, errors)

    # Check metadata
    metadata = result.get("metadata", {})
    if not metadata.get("reply_agent_compatible"):
        logger.warning("Result may not be compatible with Reply Agent")

    # Normal processing
    return draft_normal_email(result)
```

## Testing Your Integration

### 1. Run the Message Bus Simulation

```bash
cd signalmesh-local-agent
python tests/test_message_bus_simulation.py
```

This shows the complete flow: Local Agent → Message Bus → Reply Agent (mocked)

### 2. Use the Test Output

The test creates a complete JSON output you can use as a reference. Save it:

```bash
python tests/test_message_bus_simulation.py > test_output.log
```

Extract the JSON from `local/progress` messages to use in your tests.

### 3. Mock the Local Agent

```python
# For testing your Reply Agent without running Local Agent
mock_result = {
    "task_id": "test_001",
    "status": "success",
    "context": {
        "requester": {"name": "Test User", "email": "test@example.com"},
        "request": {"original_question": "Test question?"}
    },
    "insights": {"highlights": ["Test highlight"]},
    # ... rest of structure
}

# Test your email drafting
draft = your_reply_agent.draft_email(mock_result)
```

## Key Fields for Reply Agent

### Must-Have Fields

1. **context.requester.email** - Where to send
2. **context.requester.name** - Who to address
3. **context.request.original_question** - What they asked
4. **insights.highlights** - What to feature
5. **results.summary** - Overview text

### Nice-to-Have Fields

6. **context.request.channel** - How to format (Email/Slack)
7. **context.request.urgency** - Response priority
8. **insights.recommendations** - Action items
9. **suggested_actions** - Follow-up options
10. **results.breakdown_by_type** - Detailed statistics

### Optional Fields

11. **performance** - For debugging
12. **errors/warnings** - To handle issues
13. **metadata** - For compatibility checking

## Example Email Output

Based on the comprehensive JSON, your Reply Agent might generate:

```
From: SignalMesh <signalmesh@example.com>
To: sarah.chen@example.com
Subject: Weekly Workspace Summary - 4 Files Analyzed

Hi Sarah,

You asked: "Can you give me a summary of what the team worked on this week?"

Here's what I found:

Active development with code changes across the workspace.

KEY HIGHLIGHTS:
• New code added (sample_analysis.py)
• Data file updated (sample_data.csv)
• Analysis notebook created (sample_notebook.ipynb)
• Documentation updated (research_notes.md)

DETAILS:
- Code: 1 file (67 lines, 5 functions, 1 class)
- Data: 1 file (10 rows analyzed)
- Notebooks: 1 file (7 cells)
- Documentation: 1 file (142 words)

RECOMMENDED ACTIONS:
✓ Review sample_analysis.py implementation
✓ Execute sample_notebook.ipynb to generate visualizations

WHAT YOU CAN DO NEXT:
→ Search for specific topics: "Find all code related to 'data processing'"
→ Get detailed analysis: "Analyze sample_analysis.py in detail"
→ Compare with last week: "Show changes from last week"

Let me know if you need more details!

Best,
SignalMesh
```

## Questions to Clarify

### Architecture Questions

1. **Message Bus Choice**
   - Are we using RabbitMQ, Redis, Kafka, or something else?
   - What's the message format standard? (JSON, Protocol Buffers, Avro?)
   - Do we need message persistence/replay?

2. **Error Handling**
   - How should Reply Agent handle Local Agent failures?
   - Should we retry? Queue for later?
   - What's the timeout policy?

3. **LLM Integration**
   - Are you using an LLM to draft emails (recommended)?
   - If yes, which one? (OpenAI, Anthropic, local model?)
   - Do you need help with prompt engineering?

### Data Questions

4. **Context Requirements**
   - Is the current context sufficient?
   - Do you need additional fields? (user preferences, history, etc.)
   - Should we track conversation threads?

5. **Email Formatting**
   - Plain text, HTML, or both?
   - Markdown support?
   - Attachments? (e.g., CSV summaries)

### Testing Questions

6. **Integration Testing**
   - Can we do end-to-end testing together?
   - Do you need mock data/fixtures?
   - What's your testing strategy?

7. **Deployment**
   - Will Reply Agent run as a separate service?
   - Docker containers? Kubernetes?
   - Same cluster as Local Agent?

## Next Steps

1. **Review this guide** and the comprehensive JSON schema
2. **Run the test scripts** to see the full message flow
3. **List any additional fields** you need in the output
4. **Discuss message bus** architecture decisions
5. **Plan integration testing** session
6. **Define error handling** strategy together

## Contact

- GitHub Repo: [signalmesh-local-agent](link-to-repo)
- Local Agent Lead: [Your name]
- Integration Questions: [How to reach you]

## Resources

- `/tests/test_message_bus_simulation.py` - See full message flow
- `/tests/test_pipeline.py` - See output examples
- `/src/agent.py` - LocalAgent implementation
- This document - Integration guide

---

**Schema Version:** 1.0
**Last Updated:** 2025-10-31
**Compatible With:** Reply Agent v0.1.0+
