# SPDX-License-Identifier: Apache-2.0
"""
Tests for tool definition schema validation.

Tests the validate_tool_definitions function that rejects malformed tool
definitions with clear error messages before they reach the model.
"""

import pytest

from vllm_mlx.api.tool_calling import validate_tool_definitions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chat_tool(name="get_weather", description="Get weather", parameters=None):
    """Build a Chat Completions-style tool definition dict."""
    func = {"name": name, "description": description}
    if parameters is not None:
        func["parameters"] = parameters
    return {"type": "function", "function": func}


def _responses_tool(name="get_weather", description="Get weather", parameters=None):
    """Build a Responses API-style tool definition dict (flat, no 'function' wrapper)."""
    tool = {"type": "function", "name": name, "description": description}
    if parameters is not None:
        tool["parameters"] = parameters
    return tool


# ---------------------------------------------------------------------------
# Tests — valid schemas pass without error
# ---------------------------------------------------------------------------

class TestValidToolDefinitions:
    """Valid tool definitions must pass validation without raising."""

    def test_valid_chat_tool_with_parameters(self):
        """A well-formed Chat Completions tool with parameters passes."""
        tool = _chat_tool(
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        )
        validate_tool_definitions([tool])  # should not raise

    def test_valid_chat_tool_without_parameters(self):
        """A tool with no parameters field is valid (parameters are optional)."""
        tool = _chat_tool(parameters=None)
        validate_tool_definitions([tool])  # should not raise

    def test_valid_responses_tool(self):
        """A well-formed Responses API tool passes."""
        tool = _responses_tool(
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )
        validate_tool_definitions([tool])  # should not raise

    def test_valid_responses_tool_without_parameters(self):
        """A Responses tool with no parameters is valid."""
        tool = _responses_tool(parameters=None)
        validate_tool_definitions([tool])  # should not raise

    def test_valid_multiple_tools(self):
        """Multiple valid tools all pass."""
        tools = [
            _chat_tool(name="tool_a", parameters={"type": "object", "properties": {}}),
            _chat_tool(name="tool_b"),
        ]
        validate_tool_definitions(tools)  # should not raise

    def test_valid_empty_properties(self):
        """Parameters with an empty properties dict is valid."""
        tool = _chat_tool(
            parameters={"type": "object", "properties": {}},
        )
        validate_tool_definitions([tool])  # should not raise

    def test_valid_parameters_without_properties_key(self):
        """Parameters dict with type 'object' but no 'properties' key is valid."""
        tool = _chat_tool(
            parameters={"type": "object"},
        )
        validate_tool_definitions([tool])  # should not raise


# ---------------------------------------------------------------------------
# Tests — missing / empty function.name
# ---------------------------------------------------------------------------

class TestMissingFunctionName:
    """Tools with missing or empty function.name must be rejected."""

    def test_missing_name_chat_tool(self):
        """Chat tool with no function.name raises ValueError."""
        tool = {"type": "function", "function": {"description": "No name"}}
        with pytest.raises(ValueError, match="function.name"):
            validate_tool_definitions([tool])

    def test_empty_name_chat_tool(self):
        """Chat tool with empty string function.name raises ValueError."""
        tool = _chat_tool(name="")
        with pytest.raises(ValueError, match="function.name"):
            validate_tool_definitions([tool])

    def test_whitespace_only_name(self):
        """Chat tool with whitespace-only name raises ValueError."""
        tool = _chat_tool(name="   ")
        with pytest.raises(ValueError, match="function.name"):
            validate_tool_definitions([tool])

    def test_missing_name_responses_tool(self):
        """Responses tool with no name raises ValueError."""
        tool = {"type": "function", "description": "No name"}
        with pytest.raises(ValueError, match="function.name"):
            validate_tool_definitions([tool])

    def test_empty_name_responses_tool(self):
        """Responses tool with empty name raises ValueError."""
        tool = _responses_tool(name="")
        with pytest.raises(ValueError, match="function.name"):
            validate_tool_definitions([tool])


# ---------------------------------------------------------------------------
# Tests — parameters type != "object"
# ---------------------------------------------------------------------------

class TestParametersTypeMustBeObject:
    """function.parameters.type must be 'object' when parameters are present."""

    def test_parameters_type_array(self):
        """Parameters with type 'array' raises ValueError."""
        tool = _chat_tool(
            parameters={"type": "array", "items": {"type": "string"}},
        )
        with pytest.raises(ValueError, match="function.parameters.type.*'object'"):
            validate_tool_definitions([tool])

    def test_parameters_type_string(self):
        """Parameters with type 'string' raises ValueError."""
        tool = _chat_tool(parameters={"type": "string"})
        with pytest.raises(ValueError, match="function.parameters.type.*'object'"):
            validate_tool_definitions([tool])

    def test_parameters_missing_type_key(self):
        """Parameters dict without a 'type' key raises ValueError."""
        tool = _chat_tool(parameters={"properties": {"a": {"type": "string"}}})
        with pytest.raises(ValueError, match="function.parameters.type.*'object'"):
            validate_tool_definitions([tool])

    def test_responses_parameters_type_array(self):
        """Responses tool with parameters type 'array' raises ValueError."""
        tool = _responses_tool(
            parameters={"type": "array", "items": {"type": "string"}},
        )
        with pytest.raises(ValueError, match="function.parameters.type.*'object'"):
            validate_tool_definitions([tool])


# ---------------------------------------------------------------------------
# Tests — parameters.properties must be a dict
# ---------------------------------------------------------------------------

class TestPropertiesMustBeDict:
    """function.parameters.properties must be a dict when present."""

    def test_properties_as_list(self):
        """Properties as a list raises ValueError."""
        tool = _chat_tool(
            parameters={"type": "object", "properties": ["not", "a", "dict"]},
        )
        with pytest.raises(ValueError, match="function.parameters.properties.*dict"):
            validate_tool_definitions([tool])

    def test_properties_as_string(self):
        """Properties as a string raises ValueError."""
        tool = _chat_tool(
            parameters={"type": "object", "properties": "bad"},
        )
        with pytest.raises(ValueError, match="function.parameters.properties.*dict"):
            validate_tool_definitions([tool])

    def test_responses_properties_as_list(self):
        """Responses tool with properties as list raises ValueError."""
        tool = _responses_tool(
            parameters={"type": "object", "properties": [1, 2, 3]},
        )
        with pytest.raises(ValueError, match="function.parameters.properties.*dict"):
            validate_tool_definitions([tool])


# ---------------------------------------------------------------------------
# Tests — parameters as non-dict
# ---------------------------------------------------------------------------

class TestParametersMustBeDict:
    """function.parameters itself must be a dict when present."""

    def test_parameters_as_string(self):
        """Parameters as a string raises ValueError."""
        tool = _chat_tool(name="my_tool")
        tool["function"]["parameters"] = "not a dict"
        with pytest.raises(ValueError, match="function.parameters must be a dict"):
            validate_tool_definitions([tool])

    def test_parameters_as_list(self):
        """Parameters as a list raises ValueError."""
        tool = _chat_tool(name="my_tool")
        tool["function"]["parameters"] = [1, 2, 3]
        with pytest.raises(ValueError, match="function.parameters must be a dict"):
            validate_tool_definitions([tool])

    def test_parameters_as_int(self):
        """Parameters as an integer raises ValueError."""
        tool = _chat_tool(name="my_tool")
        tool["function"]["parameters"] = 42
        with pytest.raises(ValueError, match="function.parameters must be a dict"):
            validate_tool_definitions([tool])


# ---------------------------------------------------------------------------
# Tests — error message includes tool index / name
# ---------------------------------------------------------------------------

class TestErrorMessages:
    """Error messages should identify which tool failed."""

    def test_error_includes_index(self):
        """Error message references the tool index."""
        tools = [
            _chat_tool(name="good_tool"),
            _chat_tool(name=""),  # bad
        ]
        with pytest.raises(ValueError, match="index 1"):
            validate_tool_definitions(tools)

    def test_error_includes_tool_name_for_param_error(self):
        """Error message includes the tool name when parameters are bad."""
        tool = _chat_tool(
            name="broken_tool",
            parameters={"type": "string"},
        )
        with pytest.raises(ValueError, match="broken_tool"):
            validate_tool_definitions([tool])
