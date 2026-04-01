# AGENTS.md - Agent Workflow Guide

This document helps AI agents work effectively in this codebase.

## Project Overview

This is an **async Python agent framework** that interacts with the NVIDIA API (Kimi K2 model) to provide a conversational AI with tool-calling capabilities. The agent can read/write files, execute bash commands, search files, and work with dates/PDFs.

## Project Structure

```
/home/clxud/Documents/github/agentzero/
├── main.py                    # Main entry point - agent logic, API calls, tools
├── test_simple.py             # Basic tool functionality tests
├── test_tools.py              # Tool calling flow debugging tests
├── test_process_response.py   # Response processing edge case tests
├── test_tool_calling_flow.py  # Full flow integration tests
└── .swarmy/                   # Swarmy framework metadata (ignore)
```

## Essential Commands

### Running the Agent
```bash
# Start interactive chat loop
python3 main.py
```

### Running Tests
```bash
# Run all test files
python3 test_simple.py
python3 test_tools.py
python3 test_process_response.py
python3 test_tool_calling_flow.py
```

### Dependencies
No requirements.txt exists. Key dependencies observed:
- `aiohttp` - HTTP client for API calls
- Standard library: `asyncio`, `json`, `subprocess`, `glob`, `re`, `logging`, `datetime`, `pathlib`
- External binary dependency: `pdftotext` (for PDF reading)

## Code Patterns & Conventions

### Async/Await Pattern
All tool functions and API calls are async:
```python
async def tool_function(param):
    result = await some_async_operation()
    return {"success": True, "data": result}
```

### Tool Registry Pattern
Tools are registered in a global `TOOLS` dictionary with aliases:
```python
TOOLS = {
    "read": read_file_tool,
    "read_file": read_file_tool,  # Alias
    "readFile": read_file_tool,   # Another alias
    # ...
}
```

### Tool Result Format
All tools return a consistent dictionary format:
```python
{"success": True, "content": "...", "...": "..."}  # Success
{"success": False, "error": "error message"}          # Failure
```

### API Payload Pattern
- `BASE_PAYLOAD` is the template for API requests
- Always copy it: `current_payload = BASE_PAYLOAD.copy()`
- Never mutate the global `BASE_PAYLOAD` directly

### Tool Call Execution Flow
1. API returns response with `tool_calls` in `message`
2. `execute_tool_calls()` extracts function name and arguments
3. Arguments are JSON-parsed from `tool_call["function"]["arguments"]`
4. Tool is looked up in `TOOLS` registry and executed
5. Results formatted as tool messages with `tool_call_id`, `role: "tool"`, JSON content
6. Results appended to conversation history
7. Follow-up API call made with updated messages

## Testing Approach

### Test Style
- Tests use `asyncio.run()` pattern
- Mock API responses using `unittest.mock.AsyncMock`
- Tests can be run standalone: `python3 test_file.py`
- No pytest configuration - tests are self-contained scripts

### Test Categories
1. **test_simple.py**: Unit tests for individual tools, payload isolation
2. **test_tools.py**: Tool execution flow, payload mutation testing
3. **test_process_response.py**: Edge cases (empty content, unknown tools, multiple tool calls)
4. **test_tool_calling_flow.py**: Full integration tests with mocked API

## Important Gotchas

### API Key Handling
- API key is hardcoded in `main.py` (line 58): `API_KEY = "nvapi-..."`
- No environment variable fallback currently implemented

### Payload Mutation Risk
- **Critical**: Always use `BASE_PAYLOAD.copy()` before modifying
- Original code had a `payload` variable that caused state persistence bugs
- Fixed by copying payload in `handle()` function

### Tool Call Loop
- `process_response()` handles multiple rounds of tool calls via `while` loop
- Each iteration: execute tools → append results → API follow-up → check for more tool calls
- Loop exits when response has no `tool_calls`

### PDF Tool Dependency
- `read_pdf` tool requires `pdftotext` binary (poppler-utils package)
- Uses subprocess to call: `pdftotext -layout <filepath> -`

### Error Handling
- API errors checked via `"error" in response_data`
- Tool errors return `{"success": False, "error": "..."}` format
- Network errors have retry logic with exponential backoff

### Rate Limiting
- `api_call_with_retry()` handles rate limits automatically
- Retries up to 3 times with exponential backoff
- Checks for `"rate limit"` in error messages

## Key Functions Reference

| Function | Purpose | Location |
|----------|---------|----------|
| `handle(request)` | Main entry for processing requests | main.py:563 |
| `process_response()` | Handles tool calling loop | main.py:515 |
| `execute_tool_calls()` | Executes tools from API response | main.py:496 |
| `api_call_with_retry()` | API call with retry logic | main.py:18 |
| `chat_loop()` | Interactive CLI loop | main.py:584 |

## Telegram Bot Commands

The Telegram bot supports the following slash commands:

| Command | Description |
|---------|-------------|
| `/start` | Initialize interaction with the bot |
| `/setprompt` | Change the system prompt. After sending this command, the bot will ask you to provide the new prompt in your next message. The prompt is stored persistently and takes effect immediately. |

## Adding New Tools

1. Implement async tool function returning `{"success": ...}` format
2. Add tool definition to `BASE_PAYLOAD["tools"]` list
3. Register in `TOOLS` dictionary (include aliases if needed)
4. Add test in `test_simple.py`

Example tool signature:
```python
async def my_tool(param: str):
    try:
        result = await do_something(param)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```
