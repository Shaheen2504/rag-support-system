from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from src.config import settings


@lru_cache(maxsize=None)
def get_llm():
    """Shared chat model, chosen by settings.LLM_MODEL ("provider:model")."""
    return init_chat_model(
        settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )


def load_faiss_index():
    """
    Load the FAISS index.

    Returns:
        FAISS retriever object.
    """
    try:
        logger.info("Loading FAISS index...")
        embeddings_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDINGS_MODEL_NAME
        )
        vector_store = FAISS.load_local(
            settings.FAISS_INDEX_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        logger.exception("Failed to load FAISS index.")
        raise e

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.FAISS_TOP_K},
    )

    return retriever
