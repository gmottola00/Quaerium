"""ReAct prompt templates and output parser for Quaerium Agents."""

from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

REACT_SYSTEM_TEMPLATE = """\
You are a helpful AI assistant that uses tools to answer questions.

{tool_block}

To use a tool, respond with EXACTLY this format (no extra text before Thought):

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: {{"param": "value"}}

After receiving the tool result (Observation), continue reasoning until you have a final answer.
When you have enough information, respond with:

Thought: <final reasoning>
Final Answer: <your complete answer to the user>

IMPORTANT RULES:
- Always start with "Thought:"
- Use ONLY tools from the Available Tools list
- Action Input must be valid JSON
- Never invent tool results; wait for the Observation
- Respond with "Final Answer:" when you have enough information
"""

REACT_USER_TEMPLATE = """\
Task: {task}
"""

REACT_OBSERVATION_TEMPLATE = "Observation: {observation}"


def build_react_prompt(task: str, tool_block: str, history: list[dict[str, str]]) -> str:
    """Build the full ReAct prompt including conversation history."""
    system = REACT_SYSTEM_TEMPLATE.format(tool_block=tool_block)
    user = REACT_USER_TEMPLATE.format(task=task)

    parts = [system, ""]
    # Inject prior conversation turns
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prefix = "User" if role == "user" else "Assistant"
        parts.append(f"{prefix}: {content}")

    parts.append(user)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

_THOUGHT_RE = re.compile(r"Thought\s*:\s*(.*?)(?=\nAction\s*:|\nFinal Answer\s*:)", re.DOTALL)
_ACTION_RE = re.compile(r"Action\s*:\s*(\w+)", re.DOTALL)
_ACTION_INPUT_RE = re.compile(r"Action Input\s*:\s*(\{.*?\}|\[.*?\]|\".*?\"|'.*?'|[^\n]+)", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"Final Answer\s*:\s*(.*)", re.DOTALL)


def parse_react_output(text: str) -> dict[str, Any]:
    """Parse a single LLM output in ReAct format.

    Returns a dict with keys:
    - thought: str
    - action: str | None
    - action_input: dict | None
    - final_answer: str | None
    - is_final: bool
    """
    result: dict[str, Any] = {
        "thought": "",
        "action": None,
        "action_input": None,
        "final_answer": None,
        "is_final": False,
    }

    # Extract thought
    thought_match = _THOUGHT_RE.search(text)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    # Check for final answer first
    final_match = _FINAL_ANSWER_RE.search(text)
    if final_match:
        result["final_answer"] = final_match.group(1).strip()
        result["is_final"] = True
        if not result["thought"]:
            # Grab everything before "Final Answer:"
            pre = text[: text.lower().find("final answer")].strip()
            result["thought"] = pre
        return result

    # Extract action + action input
    action_match = _ACTION_RE.search(text)
    if action_match:
        result["action"] = action_match.group(1).strip()

    input_match = _ACTION_INPUT_RE.search(text)
    if input_match:
        raw_input = input_match.group(1).strip()
        result["action_input"] = _parse_action_input(raw_input)

    # If no action found but text is non-empty, treat as final answer
    if result["action"] is None and not result["is_final"]:
        result["final_answer"] = text.strip()
        result["is_final"] = True

    return result


def _parse_action_input(raw: str) -> dict[str, Any]:
    """Try to parse the action input as JSON; fall back to {query: raw}."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        return {"query": str(parsed)}
    except (json.JSONDecodeError, ValueError):
        return {"query": raw}


__all__ = [
    "REACT_SYSTEM_TEMPLATE",
    "build_react_prompt",
    "parse_react_output",
]
