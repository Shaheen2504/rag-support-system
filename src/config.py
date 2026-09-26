"""Configuration settings for the project."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Project settings."""

    model_config = SettingsConfigDict(
        env_file="./.env", env_file_encoding="utf-8", extra="allow"
    )

    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    INDEX_DIR: Path = DATA_DIR / "indexes"

    # Data settings
    DATA_URL: str = (
        "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    )
    RAW_DATA_PATH: str = str(DATA_DIR / "customer_care_emails.csv")
    PROCESSED_DATA_PATH: str = str(DATA_DIR / "processed_data.csv")

    # Embeddings settings
    EMBEDDINGS_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM settings: "<provider>:<model>" for langchain init_chat_model,
    # e.g. "openai:gpt-4o-mini", "groq:llama-3.1-8b-instant", "ollama:llama3.2:3b"
    LLM_MODEL: str = "ollama:llama3.2:3b"
    LLM_TEMPERATURE: float = 0
    LLM_MAX_TOKENS: int = 300

    # Held-out evaluation split (never indexed)
    TEST_DATA_PATH: str = str(DATA_DIR / "test.csv")
    TEST_FRACTION: float = 0.05

    FAISS_INDEX_PATH: str = str(INDEX_DIR / "faiss_index.faiss")

    FAISS_TOP_K: int = 5

    # Retrieval: "faiss" (baseline), "hybrid" (BM25 + FAISS, RRF fusion),
    # or "hybrid_rerank" (hybrid candidates reordered by a cross-encoder)
    RETRIEVAL_MODE: str = "hybrid_rerank"
    CANDIDATE_K: int = 20  # per-retriever candidates before fusion / rerank
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"

    # Evaluation settings
    EVALUATION_SAMPLE_SIZE: int = 30
    EVALUATION_LLM_MODEL: str = "openai:gpt-4o-mini"
    EVALUATION_OUTPUT_DIR: str = str(BASE_DIR / "evaluation_results")
    EVALUATION_RANDOM_SEED: int = 123

    # Logging settings
    LOGGING_LEVEL: str = "INFO"
    LOGGING_FILE: str = str(BASE_DIR / "logs" / "preprocessing.log")

    # Ensure that the data directory exists
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.INDEX_DIR, exist_ok=True)
        os.makedirs(self.BASE_DIR / "logs", exist_ok=True)
        os.makedirs(self.EVALUATION_OUTPUT_DIR, exist_ok=True)


settings = Settings()
