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

