# Local Agent - Quick Reference for Reply Agent

## Essential Fields (Must Use)

```python
# WHO to send to
to_email = result["context"]["requester"]["email"]
to_name = result["context"]["requester"]["name"]

# WHAT they asked
question = result["context"]["request"]["original_question"]

# WHAT to tell them
highlights = result["insights"]["highlights"]  # Bullet points
summary = result["results"]["summary"]          # Overview paragraph
activity = result["insights"]["activity_summary"]  # One-liner

# HOW to format
channel = result["context"]["request"]["channel"]  # Email, Slack, Teams
urgency = result["context"]["request"]["urgency"]  # Normal, High, Critical

# WHAT to suggest
recommendations = result["insights"]["recommendations"]  # Action items
suggestions = result["suggested_actions"]              # Follow-up queries
```

## Quick Email Template

```python
def draft_email(result):
    context = result["context"]
    insights = result["insights"]
    results_data = result["results"]

    email = f"""Hi {context["requester"]["name"]},

You asked: "{context["request"]["original_question"]}"

{insights["activity_summary"]}

KEY FINDINGS:
{format_bullets(insights["highlights"])}

RECOMMENDATIONS:
{format_bullets(insights["recommendations"])}

NEXT STEPS:
{format_suggestions(result["suggested_actions"])}

Best regards,
SignalMesh"""

    return {
        "to": context["requester"]["email"],
        "subject": f"Workspace Summary - {results_data['statistics']['files_processed']} Files",
        "body": email
    }
```

## Error Handling

```python
# Always check status first
if result["status"] != "success":
    return draft_error_email(result["message"])

# Check for processing errors
if result.get("errors"):
    # Mention in email that some files failed
    pass

# Check compatibility
if not result["metadata"]["reply_agent_compatible"]:
    logger.warning("Unexpected output format")
```

## Channel-Specific Formatting

```python
def format_for_channel(content, channel):
    if channel == "Email":
        return format_html_email(content)
    elif channel == "Slack":
        return {
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": content}}
            ]
        }
    elif channel == "Teams":
        return format_teams_adaptive_card(content)
```

## Common Patterns

### Extract Statistics
```python
stats = result["results"]["statistics"]
print(f"{stats['files_processed']} files in {stats['total_size_mb']:.2f} MB")
```

### List Files by Type
```python
breakdown = result["results"]["breakdown_by_type"]
for file_type, info in breakdown.items():
    print(f"{file_type}: {info['count']} files")
```

### Get Top Highlights
```python
top_highlights = result["insights"]["highlights"][:5]
```

### Format Suggestions
```python
for action in result["suggested_actions"]:
    print(f"→ {action['description']}")
    print(f"  Example: {action['example']}")
```

## Testing Snippet

```python
# Mock result for testing
mock_result = {
    "task_id": "test_001",
    "status": "success",
    "context": {
        "requester": {
            "name": "Test User",
            "email": "test@example.com"
        },
        "request": {
            "original_question": "What happened today?",
            "channel": "Email"
        }
    },
    "insights": {
        "highlights": ["Change 1", "Change 2"],
        "activity_summary": "Test activity",
        "recommendations": ["Action 1"]
    },
    "results": {
        "summary": "Test summary",
        "statistics": {"files_processed": 5}
    },
    "suggested_actions": [
        {
            "action": "search",
            "description": "Search workspace",
            "example": "Find code about X"
        }
    ],
    "metadata": {
        "reply_agent_compatible": True
    }
}

# Test your drafting
draft = draft_email(mock_result)
print(draft["body"])
```

## Schema Validation

```python
import jsonschema
import json

# Load schema
with open("docs/output_schema_v1.0.json") as f:
    schema = json.load(f)

# Validate result
try:
    jsonschema.validate(result, schema)
    print("✓ Valid output")
except jsonschema.ValidationError as e:
    print(f"✗ Invalid output: {e.message}")
```

## Questions? See:

- **Full Integration Guide**: `INTEGRATION_GUIDE.md`
- **JSON Schema**: `docs/output_schema_v1.0.json`
- **Example Output**: Run `python tests/test_message_bus_simulation.py`
- **Code Examples**: See `tests/test_message_bus_simulation.py` MockReplyAgent class
