# Customer Support Agentic RAG

A customer-support Q&A service built on **LangGraph**. Questions pass through input guardrails and a topic filter, get answered from a **FAISS** index of real support conversations, and the answer is checked by output guardrails before it's returned.

Stack: FastAPI · LangGraph / LangChain · FAISS · HuggingFace embeddings (`all-MiniLM-L6-v2`) · Ollama (default) or OpenAI · LLM Guard · Polars · ragas · Docker Compose.

## Architecture

```mermaid
flowchart LR
    U[Browser / client] -->|POST /answer| API[FastAPI<br/>src/api/main.py]
    API --> G[LangGraph workflow<br/>src/graph/graph.py]
    G --> F[(FAISS index<br/>data/indexes)]
    G --> L[LLM<br/>Ollama or OpenAI]
    G --> LG[LLM Guard scanners]
    HF[(HuggingFace dataset<br/>Bitext customer support)] -->|src/indexing/preprocess.py| F
```

## Workflow

Three input scanners run in parallel. If they pass, the question is checked for topic, matching Q&A pairs are retrieved and graded, and an answer is generated. Three output scanners then run in parallel before the answer is accepted.

```mermaid
flowchart TD
    S([START]) --> PI[scan_prompt_injection]
    S --> TX[scan_toxicity]
    S --> TL[scan_token_limit]
    PI & TX & TL --> QC{question_check_node<br/>question_valid?}
    QC -- False --> E1([END: 'Question failed checks'])
    QC -- True --> TC{topic_classifier<br/>on_topic?}
    TC -- No --> E2([END])
    TC -- Yes --> R[retrieve_docs<br/>FAISS top-k = 5]
    R --> DG[docs_grader<br/>LLM keeps relevant docs]
    DG --> GA[generate_answer]
    GA --> LS[check_language_same]
    GA --> RL[check_relevance]
    GA --> SE[check_sentiment]
    LS & RL & SE --> AC[answer_check_node]
    AC --> E3([END])
```

The graph state (`src/graph/state.py`) holds `question`, `question_status`, `question_valid`, `on_topic`, `documents`, `prompt`, `llm_output`, `answer_status` and `answer_valid`. The status lists use an `add` reducer, so results from the parallel scanners are merged.

## Request lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant A as FastAPI
    participant W as LangGraph
    participant V as FAISS
    participant M as LLM
    C->>A: POST /answer {"question": "..."}
    A->>W: graph.invoke({"question"})
    W->>W: input scanners (LLM Guard)
    W->>M: topic classification
    W->>V: similarity search (k=5)
    W->>M: grade each document
    W->>M: generate answer from graded context
    W->>W: output scanners (LLM Guard)
    W-->>A: final state
    A-->>C: JSON (llm_output, answer_valid, ...)
```

## Indexing

`src/indexing/preprocess.py` downloads the Bitext dataset, renames `instruction`/`response` to `question`/`answer` and drops nulls. It then embeds each question as a document (keeping the Q&A pair as metadata) and writes the FAISS index to `data/indexes/faiss_index.faiss`. If the index already exists, only new document IDs are added.

## Project structure

```
src/
├── config.py              # pydantic-settings; all tunables live here
├── api/                   # FastAPI app + static web UI (index.html, script.js, styles.css)
├── graph/                 # LangGraph nodes, state, and workflow wiring
├── indexing/preprocess.py # dataset download + FAISS index build
└── evaluation/evalute_rag.py  # ragas evaluation
data/                      # FAISS index (generated)
evaluation_results/        # ragas HTML reports
```

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/) (or pip with `requirements.txt`), plus [Ollama](https://ollama.com/) for local inference.

```bash
uv sync
cp .env.example .env              # add OPENAI_API_KEY if using OpenAI / evaluation
ollama pull llama3.2:3b           # must match OLLAMA_MODEL_NAME in src/config.py
```

### Configuration

Settings live in `src/config.py`, and any of them can be overridden from `.env`:

| Setting | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL_NAME` | `llama3.2:3b` | Local LLM |
| `LLM_MODEL_NAME` | `gpt-4o-mini` | OpenAI model when `local_llm=False` |
| `LLM_MAX_TOKENS` | `100` | Max answer length for Ollama |
| `EMBEDDINGS_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embeddings |
| `FAISS_TOP_K` | `5` | Retrieved documents |
| `EVALUATION_SAMPLE_SIZE` | `10` | ragas sample size |
| `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT` | — | Optional LangSmith tracing |

## Running

```bash
# 1. Build the index
uv run python -m src.indexing.preprocess

# 2. Start the API + web UI at http://localhost:8000
uv run uvicorn src.api.main:app --reload

# 3. Ask a question
curl -X POST localhost:8000/answer -H 'Content-Type: application/json' \
     -d '{"question": "I want to return a package"}'
```

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Chat web UI |
| `/answer` | POST | `{"question": str}` → final graph state as JSON |
| `/health` | GET | `{"status": "ok"}` |

### Docker Compose

```mermaid
flowchart LR
    O[ollama<br/>:11434] --> I[data-indexing<br/>builds FAISS index]
    I -- completed successfully --> B[bot-api<br/>:8000]
    O --> B
```

```bash
docker compose up --build
```

