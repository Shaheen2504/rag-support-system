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
