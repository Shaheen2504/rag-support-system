"""
Preprocess the dataset and create a FAISS index.
"""

import os

import polars as pl
from langchain_community.embeddings import HuggingFaceEmbeddings
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
    customer_care_df = customer_care_df.select(["instruction", "response"]).rename(
        {"instruction": "question", "response": "answer"}
    )
    customer_care_df = customer_care_df.drop_nulls()
    logger.info(f"Preprocessed dataset with {customer_care_df.height} records.")

    return customer_care_df


