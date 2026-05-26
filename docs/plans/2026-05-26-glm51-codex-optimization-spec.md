# GLM-5.1 Codex Optimization Spec

Date: 2026-05-26
Branch: `feat/glm51-codex-optimization`
Fork: `andreas605/vllm-mlx` (upstream: `waybarrios/vllm-mlx`)

## Goal

Fix three protocol gaps that currently require a W31 normalizer bridge between vllm-mlx and Codex, and add GLM-5.1 latency optimizations. If these land, the bridge becomes a thin passthrough or can be removed entirely.

## Protocol Fixes

### Fix 1: Emit `stop_reason` in Responses API

**Root cause:** `ResponseObject` in `responses_models.py:182` has no `stop_reason` field. `_build_response_object()` in `server.py:2107` receives `finish_reason` but discards it.

**Changes:**
- `vllm_mlx/api/responses_models.py`: Add `stop_reason: str | None = None` to `ResponseObject`
- `vllm_mlx/server.py` `_build_response_object()`: Map `finish_reason` to `stop_reason`:
  - `"stop"` → `"end_turn"`
  - `"tool_calls"` → `"tool_use"`
  - `"length"` → `"max_tokens"`
  - Other/None → None
- `vllm_mlx/server.py` streaming path: Ensure `response.completed` events include `stop_reason`
- Tests: Add to `tests/test_responses_api.py`

**W31 elimination:** `normalize_completed_event()` (end_turn defaulting) becomes unnecessary.

### Fix 2: Validate tool schemas at intake

**Root cause:** `ToolDefinition` in `models.py:106` accepts `function: dict` with zero validation. `validate_json_schema()` exists in `tool_calling.py:429` but is only used for response output.

**Changes:**
- `vllm_mlx/api/tool_calling.py`: Add `validate_tool_definitions(tools)` that checks:
  - `function.name` is a non-empty string
  - `function.parameters` (if present) has `type: "object"`
  - `function.parameters.properties` (if present) is a dict
  - Rejects structurally invalid JSON Schema in parameters
- `vllm_mlx/server.py`: Call `validate_tool_definitions()` at request intake for `/v1/chat/completions` and `/v1/responses`, return 400 on failure
- `vllm_mlx/api/responses_models.py`: Validate `ResponseFunctionTool.parameters` similarly
- Tests: New `tests/test_tool_schema_validation.py`

**W31 elimination:** `validate_tools_fail_closed()` becomes unnecessary.

### Fix 3: Native custom tool support in Responses API

**Root cause:** vllm-mlx only supports `function_call` type. Codex needs `custom_tool_call` for freeform patch input (e.g., `apply_patch` with raw string).

**Changes:**
- `vllm_mlx/api/responses_models.py`: Add models:
  - `ResponseCustomTool(type="custom", name, description)`
  - `ResponseCustomToolCallItem(type="custom_tool_call", name, input: str, call_id, status)`
  - `ResponseCustomToolCallOutputItem(type="custom_tool_call_output", call_id, output: str)`
- `vllm_mlx/server.py` `_responses_tools_to_chat_tools()`: When tool type is `"custom"`, bridge internally as `function` with `parameters: {type: "object", properties: {input: {type: "string"}}, required: ["input"]}`, track in a `custom_tool_map`
- `vllm_mlx/server.py` response building: When a function_call matches a custom-bridged tool, emit as `custom_tool_call` with raw string `input` (not JSON `arguments`)
- `vllm_mlx/server.py` continuation input: Accept `custom_tool_call_output` items, convert to `function_call_output` internally
- Streaming: Emit `custom_tool_call` items in streaming events when tool is custom-bridged
- Tests: Add custom tool tests to `tests/test_responses_api.py`

**W31 elimination:** `normalize_request_for_vllm()`, `normalize_response_item()`, and `normalize_tool_output_item_for_vllm()` all become unnecessary.

## Engine Optimizations

### Opt 1: Pre-compute system KV cache gate

**Location:** `vllm_mlx/engine/simple.py:1034-1040`

**Change:** Evaluate the gate conditions (MTP enabled, max_kv_size, logits_processors) once at engine initialization. Store result as `self._system_kv_cache_eligible: bool`. Replace per-request checks with the cached boolean.

**Impact:** ~2-5ms saved per request.

### Opt 2: Widen memory pressure polling interval

**Location:** `vllm_mlx/engine_core.py:177-188`

**Change:** Add a configurable `memory_check_interval` parameter (default 64, recommend 512 for high-memory systems). The synchronous `mx.get_active_memory()` call runs every N steps instead of every 64.

**Impact:** ~0.5ms saved per 64 tokens on 512GB systems.

## Documentation

### GLM-5.1 Performance Guide

**File:** `docs/guides/glm51-optimization.md`

Contents:
- SpecPrefill configuration for long-context agentic workflows
- System KV cache reuse for multi-turn conversations
- 8-bit KV quantization for expanded cache capacity
- Prefill chunk size tuning for GLM layer dimensions
- Recommended CLI flags for Mac Studio M3 Ultra

## Test Plan

- All existing tests must pass (`pytest -v --tb=short -m "not slow and not integration"`)
- New tests for each protocol fix (unit tests, no MLX required)
- Negative tests for schema validation (malformed schemas must 400)
- Custom tool round-trip tests (request → bridge → response → continuation)
- Streaming tests for stop_reason emission
