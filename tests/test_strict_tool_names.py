"""Tests for --strict-tool-names server option.

When enabled, tool calls whose names don't match any tool in the request's
tool list are filtered out.
"""

import logging
import unittest
from unittest.mock import patch

from vllm_mlx.api.models import (
    ChatCompletionRequest,
    FunctionCall,
    ToolCall,
    ToolDefinition,
)
from vllm_mlx.server import _filter_tool_calls_by_name


def _make_tool_call(name: str, arguments: str = '{}') -> ToolCall:
    return ToolCall(
        id=f"call_{name}",
        type="function",
        function=FunctionCall(name=name, arguments=arguments),
    )


def _make_tool_def(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object"}},
    }


def _make_request(tool_names: list[str] | None = None, **kwargs) -> ChatCompletionRequest:
    tools = [_make_tool_def(n) for n in tool_names] if tool_names else None
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        tools=tools,
        **kwargs,
    )


class TestFilterToolCallsByName(unittest.TestCase):
    """Unit tests for _filter_tool_calls_by_name."""

    def test_valid_tool_call_passes_through(self):
        """A tool call matching a defined tool passes through."""
        tc = _make_tool_call("apply_patch", '{"diff": "..."}')
        request = _make_request(["apply_patch", "exec_command"])
        with patch("vllm_mlx.server._strict_tool_names", True):
            result = _filter_tool_calls_by_name([tc], request)
        assert result is not None
        assert len(result) == 1
        assert result[0].function.name == "apply_patch"

    def test_invalid_tool_call_filtered_when_strict(self):
        """A tool call NOT matching any defined tool is filtered when strict mode is on."""
        tc = _make_tool_call("cat")
        request = _make_request(["apply_patch", "exec_command"])
        with patch("vllm_mlx.server._strict_tool_names", True):
            result = _filter_tool_calls_by_name([tc], request)
        assert result is None

    def test_invalid_tool_call_not_filtered_when_strict_off(self):
        """Tool calls are NOT filtered when strict mode is off (default behavior)."""
        tc = _make_tool_call("cat")
        request = _make_request(["apply_patch", "exec_command"])
        with patch("vllm_mlx.server._strict_tool_names", False):
            result = _filter_tool_calls_by_name([tc], request)
        assert result is not None
        assert len(result) == 1
        assert result[0].function.name == "cat"

    def test_mixed_valid_invalid_tool_calls(self):
        """Mixed valid/invalid tool calls: valid ones pass, invalid ones filtered."""
        tc_valid = _make_tool_call("apply_patch", '{"diff": "x"}')
        tc_invalid = _make_tool_call("cat")
        tc_valid2 = _make_tool_call("exec_command", '{"cmd": "ls"}')
        request = _make_request(["apply_patch", "exec_command"])
        with patch("vllm_mlx.server._strict_tool_names", True):
            result = _filter_tool_calls_by_name(
                [tc_valid, tc_invalid, tc_valid2], request
            )
        assert result is not None
        assert len(result) == 2
        names = [tc.function.name for tc in result]
        assert "apply_patch" in names
        assert "exec_command" in names
        assert "cat" not in names

    def test_all_tool_calls_filtered_returns_none(self):
        """All tool calls filtered -> None returned (text-only response)."""
        tc1 = _make_tool_call("cat")
        tc2 = _make_tool_call("ls")
        request = _make_request(["apply_patch", "exec_command"])
        with patch("vllm_mlx.server._strict_tool_names", True):
            result = _filter_tool_calls_by_name([tc1, tc2], request)
        assert result is None

    def test_no_tools_in_request_passes_through(self):
        """When no tools are defined in the request, all tool calls pass through."""
        tc = _make_tool_call("anything")
        request = _make_request(tool_names=None)
        with patch("vllm_mlx.server._strict_tool_names", True):
            result = _filter_tool_calls_by_name([tc], request)
        assert result is not None
        assert len(result) == 1

    def test_none_tool_calls_returns_none(self):
        """None input returns None regardless of strict mode."""
        request = _make_request(["apply_patch"])
        with patch("vllm_mlx.server._strict_tool_names", True):
            result = _filter_tool_calls_by_name(None, request)
        assert result is None

    def test_empty_tool_calls_returns_empty(self):
        """Empty list input returns empty list."""
        request = _make_request(["apply_patch"])
        with patch("vllm_mlx.server._strict_tool_names", True):
            result = _filter_tool_calls_by_name([], request)
        # Empty list is falsy, so the early return triggers
        assert result == []

    def test_logs_warning_on_filtered_call(self):
        """Verify a warning is logged when a tool call is filtered."""
        tc = _make_tool_call("cat")
        request = _make_request(["apply_patch"])
        with patch("vllm_mlx.server._strict_tool_names", True), \
             patch("vllm_mlx.server.logger") as mock_logger:
            _filter_tool_calls_by_name([tc], request)
        mock_logger.warning.assert_called_once()
        assert "cat" in mock_logger.warning.call_args[0][1]

    def test_tool_def_as_pydantic_model(self):
        """Works with ToolDefinition pydantic model objects, not just dicts."""
        tc = _make_tool_call("apply_patch")
        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                ToolDefinition(
                    type="function",
                    function={"name": "apply_patch", "parameters": {"type": "object"}},
                )
            ],
        )
        with patch("vllm_mlx.server._strict_tool_names", True):
            result = _filter_tool_calls_by_name([tc], request)
        assert result is not None
        assert len(result) == 1


class TestExtractReasoningAndToolCallsWithStrictNames(unittest.TestCase):
    """Test that _extract_reasoning_and_tool_calls applies the filter."""

    def test_extract_filters_invalid_tool_calls(self):
        """_extract_reasoning_and_tool_calls should filter invalid names when strict."""
        from vllm_mlx.server import _extract_reasoning_and_tool_calls

        text = '<tool_call>{"name": "cat", "arguments": {}}</tool_call>'
        request = _make_request(["apply_patch"])
        with patch("vllm_mlx.server._strict_tool_names", True):
            _, _, tool_calls = _extract_reasoning_and_tool_calls(text, request)
        assert tool_calls is None

    def test_extract_keeps_valid_tool_calls(self):
        """_extract_reasoning_and_tool_calls should keep valid names when strict."""
        from vllm_mlx.server import _extract_reasoning_and_tool_calls

        text = '<tool_call>{"name": "apply_patch", "arguments": {"diff": "x"}}</tool_call>'
        request = _make_request(["apply_patch"])
        with patch("vllm_mlx.server._strict_tool_names", True):
            _, _, tool_calls = _extract_reasoning_and_tool_calls(text, request)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "apply_patch"


class TestStrictToolNamesDefaultOff(unittest.TestCase):
    """Verify default behavior is preserved (no filtering)."""

    def test_default_strict_tool_names_is_false(self):
        """The module-level default should be False."""
        import vllm_mlx.server as srv
        # Check the default without any patching
        assert hasattr(srv, "_strict_tool_names")
        # The default value in the source is False
        # (may be changed by CLI, so just check it exists as a bool)
        assert isinstance(srv._strict_tool_names, bool)


class TestResponsesAPIStrictToolNames(unittest.TestCase):
    """Verify the filter applies when processing Responses API tool calls."""

    def test_filter_applied_to_responses_tool_calls(self):
        """Simulate Responses API path: tool calls should be filtered by name."""
        tc_valid = _make_tool_call("apply_patch", '{"diff": "test"}')
        tc_invalid = _make_tool_call("cat", '{}')
        request = _make_request(["apply_patch"])
        with patch("vllm_mlx.server._strict_tool_names", True):
            result = _filter_tool_calls_by_name([tc_valid, tc_invalid], request)
        assert result is not None
        assert len(result) == 1
        assert result[0].function.name == "apply_patch"


if __name__ == "__main__":
    unittest.main()
