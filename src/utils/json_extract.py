"""Robust JSON extraction from an LLM response.

Local + frontier models wrap their JSON in prose, markdown fences, or reasoning
<think>...</think> tags. This stack-based extractor (originally an inner function
duplicated in reviewer.py and manager.py) strips that wrapping and returns the first
balanced top-level JSON object as a string for json.loads.

Only the duo Director imports this; the quad Manager/Reviewer keep their own inner copies
unchanged, so extracting it here is zero-risk for the existing pipeline.
"""
from __future__ import annotations

import re


def extract_json(content: str) -> str:
    """Extract the first balanced top-level JSON object/array from an LLM response.

    Strips <think>/<thinking> reasoning tags and ```json fences, then uses a brace/bracket
    stack (string- and escape-aware) to find the matching closing brace so nested objects
    and arrays are handled correctly. Returns "{}" for empty input; on failure returns the
    best-effort substring and lets the caller's json.loads raise."""
    if not content:
        return "{}"

    content = content.strip()

    # Remove thinking tags (common in reasoning models). Keep the original in case the
    # JSON actually lived inside the tags and stripping emptied everything.
    content_clean = re.sub(r'<think[^>]*>.*?</think[^>]*>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content_clean = re.sub(r'<thinking[^>]*>.*?</thinking[^>]*>', '', content_clean, flags=re.DOTALL | re.IGNORECASE)
    content_to_use = content if not content_clean.strip() else content_clean

    # Prefer a markdown code block if present.
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content_to_use, re.DOTALL)
    if json_match:
        content_to_use = json_match.group(1).strip()

    json_start = content_to_use.find('{')
    if json_start == -1:
        if content_to_use != content:
            json_start = content.find('{')
            if json_start != -1:
                content_to_use = content
            else:
                return content_to_use
        else:
            return content_to_use

    # Stack-based matching of the closing brace (handles nesting; ignores braces in strings).
    stack = []
    in_string = False
    escape_next = False
    json_end = json_start

    for i in range(json_start, len(content_to_use)):
        char = content_to_use[i]

        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue

        if char == '{':
            stack.append('{')
        elif char == '}':
            if stack and stack[-1] == '{':
                stack.pop()
                if not stack:  # matching closing brace for the top-level object
                    json_end = i + 1
                    break
        elif char == '[':
            stack.append('[')
        elif char == ']':
            if stack and stack[-1] == '[':
                stack.pop()

    if json_end > json_start:
        content_to_use = content_to_use[json_start:json_end]

    return content_to_use
