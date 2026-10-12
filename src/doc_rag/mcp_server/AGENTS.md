# src/doc_rag/mcp_server

## Purpose

Exposes the documents to MCP clients (Claude Code, Claude Desktop) as three tools. The folder is named `mcp_server`, not `mcp`, so it never shadows the SDK package.

## Ownership

- `server.py` - the `mcp` object (`MCPServer`) and its tools `list_documents`, `search_documents`, `ask_documents`
- `schemas.py` - `AskResponse` (answer, citations, sources), the result of `ask_documents`

## Local Contracts

- The SDK is `mcp` 2.x. `FastMCP` no longer exists; the class is `from mcp.server.mcpserver import MCPServer` (and `Context`, `mcp.server.mcpserver.exceptions.ToolError`). Importing `mcp.server.fastmcp` raises on purpose.
- The server is mounted by `main.py`: `app.routes.extend(mcp.streamable_http_app(stateless_http=True, json_response=True).routes)` adds the exact route `/mcp`, and the lifespan wraps its `yield` in `mcp.session_manager.run()`. Do not use a catch-all `app.mount("/", ...)`: it would shadow every route added after it. The session manager can be run only once per `MCPServer` instance.
- Stateless JSON mode: each call stands alone, there are no sessions to lose when the API restarts, and tools do not stream or push notifications. Anything that needs progress updates or server-initiated messages needs stateful mode.
- DNS-rebinding protection is on: the SDK only accepts `Host` headers `localhost`, `127.0.0.1` and `[::1]` (any port). Reaching `/mcp` under another hostname fails with 421 until `transport_security` is configured in `streamable_http_app(...)`.
- Tools get the lifespan objects from the call's own HTTP request: `ctx.request_context.request.state.embedder / reranker / llm` (see `_state`). That request is `None` over non-HTTP transports, which the tools reject with a `ToolError`. Tools never create models.
- Tools are not route handlers, so they have no FastAPI dependencies. Each opens its own repository with `async with DBRepository.open() as db:` and builds the service from it (`RetrievalService`, `DocumentsService`, `ChatPipeline`), mirroring `dependencies.py`.
- `ask_documents` runs `ChatPipeline.run` and returns the `done` event with the `sources` event; it does not stream. A pipeline `error` event becomes a `ToolError` carrying the event's user-safe message. Any other exception reaches the client only as "Error executing tool ...", with the details in the server log.
- Tool inputs are validated by the SDK from the Python signature (`Annotated[..., Field(...)]`); `ctx` is injected and hidden from the client's schema. Descriptions come from the docstrings, which are written for the model that reads them: keep them accurate when behaviour changes.
- Return types are pydantic models from the other feature folders (`SearchResult`, `DocumentResponse`, `AskResponse`), so the SDK publishes an output schema and returns structured content.
- Like the REST API, `/mcp` has no authentication. Do not expose port 8000 beyond the local machine.

## Work Guidance

- Add a tool here by writing an `async def` with `@mcp.tool()`; put any logic in the feature's service or pipeline, not in the tool.
- Check the SDK's own source in `.venv/Lib/site-packages/mcp` before using an API; it changed completely between 1.x and 2.x.

## Verification

- `uv run pytest tests/test_mcp_server.py` - the tools with a fake context, repository, search and pipeline; tool registration and schemas; the `/mcp` route
- With the stack running, the real protocol: `uv run python <script>` using `from mcp import Client` and `Client("http://localhost:8000/mcp")` (`list_tools`, `call_tool`)
- Register it in Claude Code: `claude mcp add --transport http doc-rag http://localhost:8000/mcp`

## Child DOX Index

None.
