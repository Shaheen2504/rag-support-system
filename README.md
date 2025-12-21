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

