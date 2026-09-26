"""
Preprocess the dataset and create a FAISS index.
"""

import polars as pl
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from loguru import logger

# local imports
from src.config import settings


# Load the dataset using Polars
def download_and_preprocess_dataset() -> pl.DataFrame:
    """Download and preprocess the dataset using Polars."""
    # Load the dataset
    customer_care_df = pl.read_csv(settings.DATA_URL)
    logger.info(f"Loaded dataset with {customer_care_df.height} records.")

    # Preprocess the dataset
    customer_care_df = customer_care_df.select(["instruction", "response", "category", "intent"]).rename(
        {"instruction": "question", "response": "answer"}
    )
    customer_care_df = customer_care_df.drop_nulls()
    logger.info(f"Preprocessed dataset with {customer_care_df.height} records.")

    return customer_care_df


def split_train_test(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Hold out a test split that is never indexed. Test rows whose question text
    also appears in train are dropped, so evaluation can't hit an exact copy.
    """
    df = df.with_row_index("row_id").sample(
        fraction=1.0, shuffle=True, seed=settings.EVALUATION_RANDOM_SEED
    )
    n_test = int(df.height * settings.TEST_FRACTION)
    test, train = df.head(n_test), df.tail(df.height - n_test)
    train_questions = train["question"].str.to_lowercase().str.strip_chars()
    test = test.filter(
        ~pl.col("question").str.to_lowercase().str.strip_chars().is_in(train_questions)
    )
    logger.info(f"Split: {train.height} train (indexed), {test.height} held-out test.")
    return train.drop("row_id"), test.drop("row_id")


def generate_documents(customer_care_df: pl.DataFrame) -> list[Document]:
    """Generate documents from a Polars DataFrame."""
    documents = [
        Document(page_content=row["question"], metadata=row)
        for row in customer_care_df.to_dicts()
    ]
    logger.info(f"Generated {len(documents)} documents.")
    return documents


def create_faiss_index(documents: list[Document]) -> None:
    """Build the FAISS index from scratch (rebuilds keep it in sync with the split)."""
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDINGS_MODEL_NAME)
    logger.info("Creating FAISS index...")
    faiss_index = FAISS.from_documents(documents, embeddings)
    faiss_index.save_local(settings.FAISS_INDEX_PATH)
    logger.info(f"Index saved to {settings.FAISS_INDEX_PATH}")


def embed_and_index():
    """Download, split, and index the train portion of the dataset."""
    customer_care_df = download_and_preprocess_dataset()
    train_df, test_df = split_train_test(customer_care_df)
    test_df.write_csv(settings.TEST_DATA_PATH)
    create_faiss_index(generate_documents(train_df))


if __name__ == "__main__":
    embed_and_index()
