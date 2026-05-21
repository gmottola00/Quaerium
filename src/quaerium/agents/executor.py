"""ReAct execution strategy for Quaerium Agents.

Implements the ExecutionStrategy Protocol using the Reason+Act loop.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from quaerium.agents.models import AgentStep, AgentTrace, ToolCall
from quaerium.agents.prompts import (
    REACT_OBSERVATION_TEMPLATE,
    build_react_prompt,
    parse_react_output,
)

logger = logging.getLogger(__name__)

_MAX_STEP_FALLBACK = "I could not determine a final answer within the allowed steps."


class ReActExecutor:
    """Default ExecutionStrategy using the ReAct (Reason+Act) loop.

    Works with any LLMClient that satisfies `generate(prompt) -> str`.
    No structured output or function-calling required.
    """

    def execute(
        self,
        *,
        task: str,
        tools: dict[str, Any],
        memory: Any,
        llm: Any,
        observers: list[Any],
        max_steps: int,
    ) -> AgentTrace:
        """Run the ReAct loop and return a complete AgentTrace."""
        start_time = time.monotonic()
        steps: list[AgentStep] = []
        # Build tool description block once
        tool_block = _build_tool_block(tools)
        # Accumulate the in-context reasoning transcript
        transcript: list[str] = []
        final_answer: str = ""
        success = False
        error: str | None = None

        for step_num in range(1, max_steps + 1):
            step_start = time.monotonic()
            history = memory.get_history()
            prompt = build_react_prompt(task, tool_block, history) + "\n".join(transcript)

            # LLM call
            try:
                raw_output = llm.generate(prompt)
            except Exception as exc:
                error = str(exc)
                logger.error("LLM call failed at step %d: %s", step_num, exc)
                break

            parsed = parse_react_output(raw_output)
            thought = parsed.get("thought", "")
            step_latency = (time.monotonic() - step_start) * 1000

            # Build AgentStep (tool call filled in below if applicable)
            agent_step = AgentStep(
                step_number=step_num,
                thought=thought,
                tool_call=None,
                observation=None,
                timestamp=time.time(),
                latency_ms=step_latency,
            )
            _notify(observers, "on_step_start", agent_step)

            if parsed["is_final"]:
                final_answer = parsed.get("final_answer", "")
                transcript.append(raw_output)
                agent_step.observation = final_answer
                steps.append(agent_step)
                success = True
                _notify(observers, "on_step_end", agent_step)
                break

            # Tool invocation
            action = parsed.get("action")
            action_input = parsed.get("action_input") or {}

            if action and action in tools:
                tool_obj = tools[action]
                tool_call = ToolCall(
                    tool_name=action,
                    arguments=action_input,
                    raw_llm_output=raw_output,
                )
                agent_step.tool_call = tool_call

                try:
                    observation = tool_obj.run(**action_input)
                except Exception as exc:
                    observation = f"Tool error: {exc}"
                    logger.warning("Tool '%s' raised: %s", action, exc)

                agent_step.observation = observation
                obs_text = REACT_OBSERVATION_TEMPLATE.format(observation=observation)
                transcript.append(raw_output)
                transcript.append(obs_text)
                _notify(observers, "on_tool_call", action, action_input, observation)
            else:
                # Unknown tool or no action — treat as final answer
                final_answer = raw_output.strip()
                agent_step.observation = final_answer
                transcript.append(raw_output)
                success = True
                steps.append(agent_step)
                _notify(observers, "on_step_end", agent_step)
                break

            steps.append(agent_step)
            _notify(observers, "on_step_end", agent_step)
        else:
            # Exhausted max_steps
            if not final_answer:
                final_answer = _MAX_STEP_FALLBACK
                error = "max_steps_reached"

        total_latency = (time.monotonic() - start_time) * 1000
        trace = AgentTrace(
            task=task,
            steps=steps,
            final_answer=final_answer,
            success=success,
            total_latency_ms=total_latency,
            error=error,
        )
        _notify(observers, "on_agent_finish", trace)
        return trace


def _build_tool_block(tools: dict[str, Any]) -> str:
    import json
    lines: list[str] = ["Available tools:"]
    for t in tools.values():
        schema_str = json.dumps(getattr(t, "parameters_schema", {}), ensure_ascii=False)
        lines.append(f"- {t.name}: {t.description}")
        lines.append(f"  Parameters: {schema_str}")
    return "\n".join(lines)


def _notify(observers: list[Any], method: str, *args: Any) -> None:
    for obs in observers:
        if hasattr(obs, method):
            try:
                getattr(obs, method)(*args)
            except Exception as exc:
                logger.warning("Observer %s.%s failed: %s", obs.__class__.__name__, method, exc)


__all__ = ["ReActExecutor"]
