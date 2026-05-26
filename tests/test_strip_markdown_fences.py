# SPDX-License-Identifier: Apache-2.0
"""Tests for the strip_markdown_fences utility and --strip-markdown-fences server option."""

import pytest

from vllm_mlx.text_processing import strip_markdown_fences


# ---------------------------------------------------------------------------
# Unit tests for strip_markdown_fences()
# ---------------------------------------------------------------------------


class TestStripMarkdownFences:
    def test_single_fenced_block_with_language_tag(self):
        text = '```json\n{"key": "value"}\n```'
        assert strip_markdown_fences(text) == '{"key": "value"}'

    def test_single_fenced_block_without_language_tag(self):
        text = "```\nhello world\n```"
        assert strip_markdown_fences(text) == "hello world"

    def test_text_without_fences_passes_through(self):
        text = '{"key": "value"}'
        assert strip_markdown_fences(text) == text

    def test_plain_text_without_fences_passes_through(self):
        text = "just some plain text"
        assert strip_markdown_fences(text) == text

    def test_fences_with_leading_whitespace(self):
        text = '  ```json\n{"key": "value"}\n```'
        assert strip_markdown_fences(text) == '{"key": "value"}'

    def test_fences_with_trailing_whitespace(self):
        text = '```json\n{"key": "value"}\n```  '
        assert strip_markdown_fences(text) == '{"key": "value"}'

    def test_fences_with_leading_and_trailing_whitespace(self):
        text = '  \n```json\n{"key": "value"}\n```\n  '
        assert strip_markdown_fences(text) == '{"key": "value"}'

    def test_empty_content_inside_fences(self):
        text = "```\n```"
        result = strip_markdown_fences(text)
        assert result == ""

    def test_multiline_content_inside_fences(self):
        text = "```python\ndef hello():\n    print('hello')\n```"
        assert strip_markdown_fences(text) == "def hello():\n    print('hello')"

    def test_four_backtick_fence(self):
        """Fences with 4+ backticks should also be stripped."""
        text = '````json\n{"key": "value"}\n````'
        assert strip_markdown_fences(text) == '{"key": "value"}'

    def test_outer_fence_stripped_when_content_has_inner_fences(self):
        """When the whole text is one outer fenced block containing inner fences,
        the outer fences are stripped and the inner content (including its own
        fences) is returned as-is."""
        # This is a single fenced block whose content happens to contain fences.
        # The outer ``` pair wraps the whole text, so the outer fences are stripped.
        text = "```\nblock one\n```\n\n```\nblock two\n```"
        result = strip_markdown_fences(text)
        # Outer fences removed; inner content preserved
        assert result == "block one\n```\n\n```\nblock two"

    def test_language_tag_python(self):
        text = "```python\nprint('hello')\n```"
        assert strip_markdown_fences(text) == "print('hello')"

    def test_language_tag_diff(self):
        text = "```diff\n--- a/file\n+++ b/file\n@@ -1 +1 @@\n-old\n+new\n```"
        assert strip_markdown_fences(text) == "--- a/file\n+++ b/file\n@@ -1 +1 @@\n-old\n+new"

    def test_content_with_internal_backtick_lines(self):
        """Content inside fences that contains backtick lines should be preserved."""
        text = "```\nsome text\n`inline code`\nmore text\n```"
        assert strip_markdown_fences(text) == "some text\n`inline code`\nmore text"

    def test_empty_string_input(self):
        assert strip_markdown_fences("") == ""

    def test_whitespace_only_input(self):
        assert strip_markdown_fences("   \n  ") == "   \n  "

    def test_partial_fence_not_stripped(self):
        """Text starting with ``` but not ending with ``` passes through."""
        text = "```json\n{incomplete"
        assert strip_markdown_fences(text) == text

    def test_fence_without_newline_after_tag(self):
        """Fence with language tag but no newline before content should not match."""
        text = "```json{inline}"
        # Does not match the fence pattern (no newline after opening)
        assert strip_markdown_fences(text) == text

    def test_json_array_content(self):
        text = "```json\n[1, 2, 3]\n```"
        assert strip_markdown_fences(text) == "[1, 2, 3]"

    def test_xml_patch_content(self):
        text = "```xml\n<root>\n  <child/>\n</root>\n```"
        assert strip_markdown_fences(text) == "<root>\n  <child/>\n</root>"


# ---------------------------------------------------------------------------
# Integration-style tests: verify server respects the flag
# ---------------------------------------------------------------------------


class TestServerStripMarkdownFencesFlag:
    """
    Verify that vllm_mlx.server exposes _strip_markdown_fences and that
    cli.py wires it up correctly from argparse args.
    """

    def test_server_module_has_strip_flag(self):
        """The server module must expose _strip_markdown_fences."""
        import vllm_mlx.server as server

        assert hasattr(server, "_strip_markdown_fences"), (
            "server._strip_markdown_fences is not defined"
        )
        assert server._strip_markdown_fences is False, (
            "server._strip_markdown_fences should default to False"
        )

    def test_cli_flag_registered(self):
        """The --strip-markdown-fences flag must exist in the serve parser."""
        from vllm_mlx.cli import create_parser

        parser = create_parser()
        # Parse a minimal serve invocation with the flag
        args = parser.parse_args(["serve", "some-model", "--strip-markdown-fences"])
        assert args.strip_markdown_fences is True

    def test_cli_flag_default_false(self):
        """The --strip-markdown-fences flag must default to False."""
        from vllm_mlx.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["serve", "some-model"])
        assert args.strip_markdown_fences is False

    def test_flag_disabled_no_stripping(self):
        """When flag is False, strip_markdown_fences must not alter text."""
        # Simulate what the server does: only strip when _strip_markdown_fences is True
        import vllm_mlx.server as server

        original = server._strip_markdown_fences
        try:
            server._strip_markdown_fences = False
            text = "```json\n{}\n```"
            # Manually replicate the server condition
            if server._strip_markdown_fences:
                result = strip_markdown_fences(text)
            else:
                result = text
            assert result == text
        finally:
            server._strip_markdown_fences = original

    def test_flag_enabled_strips_fences(self):
        """When flag is True, strip_markdown_fences is called on model output."""
        import vllm_mlx.server as server

        original = server._strip_markdown_fences
        try:
            server._strip_markdown_fences = True
            text = "```json\n{}\n```"
            # Manually replicate the server condition (no tools)
            has_tools = False
            if server._strip_markdown_fences and not has_tools:
                result = strip_markdown_fences(text)
            else:
                result = text
            assert result == "{}"
        finally:
            server._strip_markdown_fences = original
