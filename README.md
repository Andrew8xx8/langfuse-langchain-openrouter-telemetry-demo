### Clean Telemetry Demo — OpenRouter + Langfuse cost tracking (LangChain/LangGraph)

This repo demonstrates reliable cost tracking in Langfuse when using OpenRouter through LangChain and LangGraph.

Problem: out-of-the-box, Langfuse did not record costs from LangChain/LangGraph runs when the underlying provider was OpenRouter. We fixed this by patching the Langfuse LangChain callback to parse and forward cost information that OpenRouter returns.

---

### What’s included
- **Patched handler**: `callback_handler_patch.py` — extends Langfuse’s `CallbackHandler` to extract OpenRouter usage.cost and attach it to generations as `cost_details`.
- **Direct client demo**: `runner.py::run_direct_openai_test` — uses `langfuse.openai` with `extra_body={"usage": {"include": true}}` to show cost in responses.
- **LangChain demo**: `runner.py::run_langchain_test` — uses `ChatOpenAI` + patched handler for per-call cost tracking.
- **LangGraph demo**: `runner.py::run_langgraph_test` — multi-node workflow with per-node cost tracking via the patched handler.
- **Clean structure**: `config.py`, `factories.py`, `utils.py`, `main.py` for clear separation of concerns.

---

### Requirements
- Docker
- Langfuse project with keys
- OpenRouter API key

Environment variables used by the app are documented in `env.example`. Copy it to `.env` and fill in your values.

---

### Run (Docker)
Build the image and run the demo using the included Dockerfile.

From the repo root:
```bash
docker build -t telemetry-demo .
docker run --rm --env-file .env telemetry-demo
```

---

### Setup
- Copy `env.example` to `.env` and fill in your keys:
  - `cp env.example .env`
- Run with Docker (see above).

What you’ll see:
- Environment and key checks
- Three separate test sections: Direct OpenAI, LangChain with cost tracking, and LangGraph multi-node
- Printed responses and cost info where available
- A summary of what was tracked

After running, open your Langfuse dashboard and filter by the printed session IDs (e.g., `langchain-cost-tracking-YYYYMMDD_HHMMSS`). You should see `cost_details` attached to generations for LangChain/LangGraph runs.

---

### How the cost tracking fix works
OpenRouter can return usage with cost when explicitly requested. We ensure this at two layers:
- Direct client: set `extra_body={"usage": {"include": true}}` in `chat.completions.create(...)`.
- LangChain/Graph: pass `model_kwargs={"extra_body": {"usage": {"include": true}}}` to `ChatOpenAI`.

Then, the patched handler `CostTrackingCallbackHandler`:
- In `on_llm_end`, inspects the LangChain `LLMResult` for usage/cost across multiple common shapes: `response.llm_output`, message `response_metadata`, or `generation_info`.
- When present, formats and forwards cost as `cost_details` on the generation update, mirroring the shape used by `langfuse.openai`.
- Adds zeroed `cost_details` in `on_llm_error` for consistency.

Relevant code:
- `factories.py` — constructs clients and the patched handler, ensures `extra_body.usage.include` is set when using `ChatOpenAI`.
- `runner.py` — three demos: direct, LangChain, LangGraph.
- `utils.py` — helpers for displaying cost and building metadata.
- `main.py` — entrypoint creating sessions/tags and running all tests.

---

### Verifying in Langfuse
- The demo sets `metadata.langfuse_session_id` and `langfuse_tags` on traces/generations.
- In Langfuse, open the session printed by the app and check each generation:
  - Direct OpenAI will show normal usage; cost may display in the console output.
  - LangChain/LangGraph generations should include `cost_details` with `total` (and input/output when provided).

---

### Caveats
- The patch relies on response shapes commonly returned by OpenRouter and LangChain; providers that omit `usage.cost` won’t produce cost entries.
- The handler extends internal behavior of Langfuse’s LangChain integration; future upstream changes may require adjusting `_parse_cost_from_llm_result` or the generation update payload.

---

### File map
```
src/
  └── telemetry_demo/
      ├── main.py                   # Entry point
      ├── runner.py                 # Direct, LangChain, LangGraph demos
      ├── factories.py              # Client + handler factories
      ├── callback_handler_patch.py # Patched Langfuse LangChain handler with cost tracking
      ├── config.py                 # Typed configs and telemetry context
      ├── utils.py                  # Cost extraction + metadata helpers
      └── __init__.py
```

---

### Inspiration / original issue
Langfuse’s default LangChain handler did not propagate OpenRouter cost information for LangChain/LangGraph calls. This repo patches the handler and provides a working example that records costs per generation/node while preserving the clean separation of config, factories, and runners.


---

### License
MIT — see `LICENSE`.


