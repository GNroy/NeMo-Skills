# NeMo-Skills MCP servers

This directory hosts the MCP servers NeMo-Skills ships out of the box.
They fall into two groups, distinguished by whether the file itself is
runnable as an MCP server or whether it only implements the in-process
`Tool` interface (`nemo_skills.mcp.tool_manager.Tool`).

## Group A: native FastMCP entrypoints

These modules already register their own `FastMCP` server and expose a
`main()` plus `if __name__ == "__main__": main()` block.  Run them
directly via `python -m`:

| Module | Run command |
|---|---|
| `nemo_skills.mcp.servers.python_tool` | `python -m nemo_skills.mcp.servers.python_tool` |
| `nemo_skills.mcp.servers.tavily_search_tool` | `python -m nemo_skills.mcp.servers.tavily_search_tool` |
| `nemo_skills.mcp.servers.exa_tool` | `python -m nemo_skills.mcp.servers.exa_tool` |

Hermes consumes these via the standard `mcp_servers:` config:

```yaml
# ~/.hermes/config.yaml
mcp_servers:
  ns_python:
    command: "python"
    args: ["-m", "nemo_skills.mcp.servers.python_tool"]
  ns_tavily:
    command: "python"
    args: ["-m", "nemo_skills.mcp.servers.tavily_search_tool"]
    env:
      TAVILY_API_KEY: "..."
```

## Group B: in-process `Tool` classes (wrapped by `ns-mcp-serve`)

These modules expose a `Tool` subclass but do not host their own MCP
server.  Use the generic stdio wrapper `nemo_skills.mcp.stdio_serve`
(console script `ns-mcp-serve`) to spawn each as an MCP stdio server:

| Module:Class | Description |
|---|---|
| `nemo_skills.mcp.servers.chemistry.periodictable_tool:PeriodictableTool` | Element / isotope lookups (no external deps). |
| `nemo_skills.mcp.servers.physics.coolprop_tool:CoolPropTool` | Thermo-physical properties via CoolProp. |
| `nemo_skills.mcp.servers.physics.particle_tool:ParticleTool` | Particle Data Group lookups. |
| `nemo_skills.mcp.servers.physics.radioactivedecay_tool:RadioactivedecayTool` | Decay chain calculations. |
| `nemo_skills.mcp.servers.web.arxiv_tool:ArxivSearchTool` | arXiv search + paper metadata. |
| `nemo_skills.mcp.servers.web.wikipedia_tool:WikipediaSearchTool` | Wikipedia search + article fetch. |

Hermes consumes them uniformly:

```yaml
# ~/.hermes/config.yaml
mcp_servers:
  ns_periodictable:
    command: "ns-mcp-serve"
    args: ["nemo_skills.mcp.servers.chemistry.periodictable_tool:PeriodictableTool"]

  ns_arxiv:
    command: "ns-mcp-serve"
    args: ["nemo_skills.mcp.servers.web.arxiv_tool:ArxivSearchTool"]
```

Tool names emerge on the Hermes side as `<server_name>.<raw_tool>` (the
server name is the YAML key — `ns_periodictable`, `ns_arxiv`, …) so they
never collide with Hermes built-ins like `execute_code` or with each
other.

### Passing config overrides

`ns-mcp-serve` forwards a JSON object to `Tool.configure(...)`:

```yaml
mcp_servers:
  ns_arxiv:
    command: "ns-mcp-serve"
    args:
      - nemo_skills.mcp.servers.web.arxiv_tool:ArxivSearchTool
      - --overrides
      - '{"max_results": 5}'
```

### Custom server name (debug only)

By default the MCP server name advertised over stdio matches the `Tool`
class name.  Pass `--server-name foo` to override; rarely needed because
Hermes uses its own YAML key for prefixing.

### Adding a new wrapped tool

1. Implement a `Tool` subclass in `nemo_skills/mcp/servers/<category>/<your>_tool.py`.
2. Make sure `default_config()`, `configure(...)`, `list_tools()`, and
   `execute(...)` all work without needing a `ToolManager` around it.
3. Add an entry to the table above.  No code changes anywhere else;
   `ns-mcp-serve` discovers the class via dotted-path lookup.

The Phase 2 design doc and Phase 0/1 context for this wrapper live in
`NeMo-Skills/workdir-agents/hermes_integration_plan.md`.
