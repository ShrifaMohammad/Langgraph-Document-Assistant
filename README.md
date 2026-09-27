# Langgraph-Document-Assistant (Intelligent Document Assistant)

A multi-agent AI assistant built with **LangChain** and **LangGraph** that
answers questions, summarizes, and performs calculations on financial and
healthcare documents (invoices, contracts, insurance claims).

The assistant classifies each incoming message and routes it to one of
three specialized agents, each backed by real tools (document search,
document reader, calculator) and structured, validated outputs.

## Features

- **Intent classification** — every message is classified as `qa`,
  `summarization`, `calculation`, or `unknown` before being routed.
- **Three specialist agents**
  - **Q&A Agent** — answers specific questions about document content,
    citing document IDs.
  - **Summarization Agent** — extracts key points and produces
    structured summaries.
  - **Calculation Agent** — extracts numbers from documents and performs
    math using a dedicated, sandboxed calculator tool (never "mental
    math" from the LLM).
- **Persistent conversation memory** — powered by LangGraph's
  `InMemorySaver` checkpointer, keyed by session `thread_id`, so the
  assistant remembers earlier turns (e.g. "what was the tax on **that
  one**?" correctly resolves to the invoice discussed a few turns
  earlier).
- **Structured outputs everywhere** — every agent response and the
  intent classification itself are enforced Pydantic models
  (`AnswerResponse`, `SummarizationResponse`, `CalculationResponse`,
  `UserIntent`, ...), not free-form text.
- **Tool usage logging** — every tool call (search, read, calculate) is
  logged with timestamp, input, and output for auditability.
- **Session persistence to disk** — conversation history and active
  document context are saved per user session under `sessions/`.

## Architecture

```
START
  │
  ▼
classify_intent  ──►  decides "qa" | "summarization" | "calculation" | "unknown"
  │
  ├──► qa_agent ────────────┐
  ├──► summarization_agent ─┤──► update_memory ──► END
  └──► calculation_agent ───┘
```

Each specialist agent is a LangGraph **ReAct agent**
(`create_react_agent`) with access to four tools:

| Tool | Purpose |
|---|---|
| `document_search` | Keyword, type, or amount-based search across documents |
| `document_reader` | Read a specific document's full content by ID |
| `document_statistics` | Aggregate stats across the whole document collection |
| `calculator` | Safely evaluate a mathematical expression |

State (messages, intent, active documents, tools used, etc.) flows
through every node as a single `AgentState` object, with LangGraph
**reducers** controlling how updates merge:

- `messages` uses the built-in `add_messages` reducer (append, don't
  overwrite) — this is what gives the assistant memory across turns.
- `actions_taken` uses `operator.add` to accumulate every node visited
  during a turn.

## Project structure

```
docmind/
├── src/
│   ├── schemas.py     # Pydantic models (structured outputs)
│   ├── retrieval.py   # Simulated document store + search logic
│   ├── tools.py        # Agent tools (calculator, search, reader, stats)
│   ├── prompts.py      # System prompts per intent type
│   ├── agent.py         # LangGraph workflow definition
│   └── assistant.py     # Session management + workflow invocation
├── sessions/            # Auto-generated: one JSON file per conversation
├── logs/                # Auto-generated: tool usage logs per session
├── main.py               # CLI entry point
├── requirements.txt
└── .env.example
```

## Setup

**Prerequisites:** Python 3.9+, an OpenAI-compatible API key.

```bash
git clone <this-repo-url>
cd docmind

python -m venv venv
# Windows
venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and add your OPENAI_API_KEY
```

If your key is a standard OpenAI key (starts with `sk-`), leave
`OPENAI_BASE_URL` unset in `.env`. If you're using a proxied/alternate
endpoint, set both:

```
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://your-endpoint/v1
```

## Running

```bash
python main.py
```

Commands available inside the assistant:

```
/help     Show help
/docs     List available documents
/quit     Exit
```

Example queries:

```
What's the total amount in invoice INV-001?
Summarize all contracts
Calculate the sum of all invoice totals
Find documents with amounts over $50,000
```

## Notable implementation details

- **Safety-checked calculator** — expressions are validated against a
  whitelist regex before evaluation, and `eval()` is called with an
  empty `__builtins__`/locals scope so no names or functions outside
  basic arithmetic are reachable.
- **Document-ID-aware retrieval** — the search tool matches directly
  against a document's ID (not just its title/content), and the system
  prompts instruct agents to call `document_reader` directly whenever the
  user names an explicit document ID (e.g. "INV-001"), rather than
  relying purely on keyword search.
- **Per-thread persistence** — `DocumentAssistant.process_message` always
  passes the current session's ID as `thread_id` in the LangGraph config,
  so the checkpointer transparently resumes state across turns without
  the caller needing to re-send the full history each time.
