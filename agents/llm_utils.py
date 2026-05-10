"""Shared LLM streaming utilities used by all sub-agents."""

import json
from openai import OpenAI


def stream_llm(
    client: OpenAI,
    model: str,
    messages: list,
    tools: list = None,
    temperature: float = 0.3,
    silent: bool = False,
) -> tuple:
    """Stream LLM response, printing text in real-time.

    Args:
        client: OpenAI client
        model: model name
        messages: chat messages list
        tools: optional tool definitions
        temperature: sampling temperature
        silent: if True, don't print (just return)

    Returns:
        (full_text: str, tool_calls: list, reasoning: str)
    """
    kwargs = dict(model=model, messages=messages, temperature=temperature, stream=True)
    if tools:
        kwargs["tools"] = tools

    stream = client.chat.completions.create(**kwargs)

    full_text = ""
    tool_calls_data = {}
    reasoning_text = ""

    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is None:
            continue

        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning_text += delta.reasoning_content

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_data:
                    tool_calls_data[idx] = {"id": "", "name": "", "arguments": ""}
                if tc.id:
                    tool_calls_data[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls_data[idx]["name"] += tc.function.name
                    if tc.function.arguments:
                        tool_calls_data[idx]["arguments"] += tc.function.arguments

        if delta.content:
            if not silent:
                print(delta.content, end="", flush=True)
            full_text += delta.content

    # Parse tool calls
    tool_calls = []
    for idx in sorted(tool_calls_data):
        tc = tool_calls_data[idx]
        try:
            args = json.loads(tc["arguments"]) if tc["arguments"] else {}
        except json.JSONDecodeError:
            args = {}
        tool_calls.append({"id": tc["id"], "name": tc["name"], "arguments": args})

    return full_text, tool_calls, reasoning_text


def call_llm_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    silent: bool = False,
) -> dict:
    """Call LLM with system+user prompt, expecting JSON response.

    Streams the response. Parses JSON from output.

    Returns:
        dict or empty dict on failure
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text, _, _ = stream_llm(client, model, messages, temperature=0.3, silent=silent)

    if not silent:
        print()  # newline after stream

    return parse_json(text)


def parse_json(content: str) -> dict:
    """Extract JSON object from LLM response (may have markdown fences)."""
    for method in [
        lambda c: json.loads(c),
        lambda c: json.loads(c[c.index("```json") + 7:c.index("```", c.index("```json") + 7)]),
        lambda c: json.loads(c[c.index("```") + 3:c.index("```", c.index("```") + 3)]),
        lambda c: json.loads(c[c.index("{"):c.rindex("}") + 1]),
    ]:
        try:
            return method(content)
        except (ValueError, json.JSONDecodeError, AttributeError):
            continue
    return {}
