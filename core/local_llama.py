"""
Setup:
    1. Install Ollama: https://ollama.com/download
    2. Pull a model with tool-use support:
           ollama pull llama3.1        # recommended (8B, good tool use)
           ollama pull llama3.2        # lighter (3B)
           ollama pull qwen2.5         # strong at tool use
    3. Make sure Ollama is running:
           ollama serve
    4. Set LOCAL_MODEL in your .env (or it defaults to "llama3.1"):
           LOCAL_MODEL=llama3.1

Usage — in main.py replace:
    from core.claude import Claude
    ...
    claude_model = os.getenv("CLAUDE_MODEL", "")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
    assert claude_model, ...
    assert anthropic_api_key, ...
    claude_service = Claude(model=claude_model)

With:
    from core.local_llama import LocalLlama
    ...
    local_model = os.getenv("LOCAL_MODEL", "llama3.1")
    claude_service = LocalLlama(model=local_model)
"""

import json
from dataclasses import dataclass, field
from typing import Any
from openai import OpenAI   # Ollama exposes an OpenAI-compatible endpoint


# ---------------------------------------------------------------------------
# Minimal shims that mimic the anthropic.types objects the rest of the
# codebase expects (Message, ContentBlock, ToolUseBlock, etc.)
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class ToolUseBlock:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class FakeMessage:
    """
    Mimics anthropic.types.Message well enough for Claude / CliChat to work.
    Fields referenced elsewhere:
        .content  -> list[TextBlock | ToolUseBlock]
        .stop_reason -> "end_turn" | "tool_use"
        .role    -> "assistant"
    """
    content: list = field(default_factory=list)
    stop_reason: str = "end_turn"
    role: str = "assistant"


# ---------------------------------------------------------------------------
# Helper: convert Anthropic-style tool schema -> OpenAI function schema
# ---------------------------------------------------------------------------

def _anthropic_tool_to_openai(tool: dict) -> dict:
    """
    Anthropic tool format:
        {"name": "...", "description": "...", "input_schema": { JSON Schema }}
    OpenAI function format:
        {"type": "function", "function": {"name": "...", "description": "...", "parameters": { JSON Schema }}}
    """
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        },
    }


# ---------------------------------------------------------------------------
# Helper: convert Anthropic message history -> OpenAI message history
# The main difference is tool_result content blocks.
# ---------------------------------------------------------------------------

def _convert_messages(messages: list) -> list:
    out = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        # content is a list of blocks
        text_parts = []
        for block in content:
            btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)

            if btype == "text":
                text = block.get("text") if isinstance(block, dict) else block.text
                text_parts.append(text)

            elif btype == "tool_use":
                # Assistant called a tool — emit as assistant message then skip;
                # the matching tool_result will follow.
                tool_id   = block.get("id")   if isinstance(block, dict) else block.id
                tool_name = block.get("name") if isinstance(block, dict) else block.name
                tool_input = block.get("input") if isinstance(block, dict) else block.input
                out.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_input),
                        },
                    }],
                })

            elif btype == "tool_result":
                tool_use_id = block.get("tool_use_id") if isinstance(block, dict) else block.tool_use_id
                result_content = block.get("content") if isinstance(block, dict) else block.content
                if isinstance(result_content, list):
                    result_text = " ".join(
                        (b.get("text") if isinstance(b, dict) else getattr(b, "text", ""))
                        for b in result_content
                    )
                else:
                    result_text = str(result_content)
                out.append({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": result_text,
                })

        if text_parts:
            out.append({"role": role, "content": "\n".join(text_parts)})

    return out


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LocalLlama:
    """
    Drop-in replacement for core.claude.Claude.
    Talks to a locally running Ollama instance via its OpenAI-compatible API.
    """

    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434/v1"):
        self.model = model
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama",   # Ollama ignores the key but the client requires one
        )

    # ------------------------------------------------------------------
    # These three helpers are called by core/chat.py exactly as on Claude
    # ------------------------------------------------------------------

    def add_user_message(self, messages: list, message):
        content = message.content if isinstance(message, FakeMessage) else message
        messages.append({"role": "user", "content": content})

    def add_assistant_message(self, messages: list, message):
        content = message.content if isinstance(message, FakeMessage) else message
        messages.append({"role": "assistant", "content": content})

    def text_from_message(self, message: FakeMessage) -> str:
        return "\n".join(
            block.text for block in message.content if isinstance(block, TextBlock)
        )

    # ------------------------------------------------------------------
    # Main chat method
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list,
        system: str = None,
        temperature: float = 1.0,
        stop_sequences: list = [],
        tools: list = None,
        thinking: bool = False,       # ignored — local models don't support this
        thinking_budget: int = 1024,  # ignored
    ) -> FakeMessage:

        oai_messages = []

        if system:
            oai_messages.append({"role": "system", "content": system})

        oai_messages.extend(_convert_messages(messages))

        params: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": min(temperature, 2.0),  # OAI max is 2.0
            "max_tokens": 8000,
        }

        if stop_sequences:
            params["stop"] = stop_sequences

        oai_tools = None
        if tools:
            oai_tools = [_anthropic_tool_to_openai(t) for t in tools]
            params["tools"] = oai_tools
            params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**params)
        choice = response.choices[0]
        oai_msg = choice.message

        # Build a FakeMessage that looks like an Anthropic Message
        content_blocks = []

        if oai_msg.content:
            content_blocks.append(TextBlock(text=oai_msg.content))

        stop_reason = "end_turn"
        if oai_msg.tool_calls:
            stop_reason = "tool_use"
            for tc in oai_msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                content_blocks.append(ToolUseBlock(
                    id=tc.id,
                    name=tc.function.name,
                    input=args,
                ))

        return FakeMessage(content=content_blocks, stop_reason=stop_reason)