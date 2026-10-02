import re
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda
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


def bm25_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def load_faiss_index(mode: str | None = None):
    """
    Load the retriever for `mode` (defaults to settings.RETRIEVAL_MODE):
    "faiss" dense only, "hybrid" BM25+FAISS, or "hybrid_rerank".
    """
    mode = mode or settings.RETRIEVAL_MODE
    try:
        logger.info(f"Loading FAISS index (retrieval mode: {mode})...")
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

    if mode == "faiss":
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.FAISS_TOP_K},
        )

    # BM25 over the same Q&A pairs, matching question + answer text: answers
    # carry words users say that the dataset's questions never use ("return").
    bm25 = BM25Retriever.from_texts(
        [
            f"{d.metadata['question']}\n{d.metadata['answer']}"
            for d in vector_store.docstore._dict.values()
        ],
        metadatas=[d.metadata for d in vector_store.docstore._dict.values()],
        preprocess_func=bm25_tokenize,
        k=settings.CANDIDATE_K,
    )
    # Return the question as content, like FAISS docs, so RRF fuses duplicates
    # and the reranker scores every candidate on the same text.
    for doc in bm25.docs:
        doc.page_content = doc.metadata["question"]
    dense = vector_store.as_retriever(search_kwargs={"k": settings.CANDIDATE_K})
    hybrid = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])

    if mode == "hybrid":
        # EnsembleRetriever returns every fused doc; keep the top k.
        return hybrid | RunnableLambda(lambda docs: docs[: settings.FAISS_TOP_K])
    if mode == "hybrid_rerank":
        reranker = CrossEncoderReranker(
            model=HuggingFaceCrossEncoder(model_name=settings.RERANKER_MODEL_NAME),
            top_n=settings.FAISS_TOP_K,
        )
        return ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=hybrid
        )
    raise ValueError(f"Unknown RETRIEVAL_MODE: {mode}")
