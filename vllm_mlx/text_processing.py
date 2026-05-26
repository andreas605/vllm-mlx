# SPDX-License-Identifier: Apache-2.0
"""
Text post-processing utilities for model output.
"""

import re

# Matches a single fenced block that wraps the entire text.
# Group 1: optional language tag (e.g. "json", "python")
# Group 2: content inside the fence
_SINGLE_FENCE_RE = re.compile(
    r"^[ \t]*```+([^\n`]*)[ \t]*\n(.*?)[ \t]*```+[ \t]*$",
    re.DOTALL,
)


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from model output.

    Detects when the *entire* response is a single fenced block and removes
    the opening fence line (with optional language tag) and the closing fence,
    returning only the inner content.

    Conservative rules:
    - Only strips when the whole trimmed text is one fenced block.
    - Handles leading/trailing whitespace around the outer fences.
    - Handles any number of backticks (``` or ````).
    - An empty inner block returns an empty string.
    - If the text does not match a single fenced block, it is returned unchanged.

    Args:
        text: The raw text from the model.

    Returns:
        Inner content with fences stripped, or the original text if no
        single-fence-block pattern is detected.
    """
    stripped = text.strip()
    if not stripped:
        return text

    m = _SINGLE_FENCE_RE.match(stripped)
    if m is None:
        return text

    # content may have a trailing newline before the closing fence; strip it.
    inner = m.group(2)
    # Remove only one trailing newline if present (the one before the closing ```)
    if inner.endswith("\n"):
        inner = inner[:-1]
    return inner
