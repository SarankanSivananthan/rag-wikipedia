"""
Reusable components for the Wikipedia RAG pipeline: data loading,
chunking, and vector indexing. More to come (retrieval eval, generation).
"""
from __future__ import annotations

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_corpus(path: str = "data/passages.parquet") -> pd.DataFrame:
    """The Wikipedia passages that make up the knowledge base."""
    return pd.read_parquet(path)


def load_qa(path: str = "data/qa_test.parquet") -> pd.DataFrame:
    """Question/answer pairs used to evaluate retrieval and generation."""
    return pd.read_parquet(path)


def passages_to_documents(df: pd.DataFrame) -> list[Document]:
    return [
        Document(page_content=row.passage, metadata={"passage_id": row.id})
        for row in df.itertuples()
    ]


def chunk_documents(
    documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 50
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def build_vector_store(
    documents: list[Document], embedding_model: str = DEFAULT_EMBEDDING_MODEL
) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return FAISS.from_documents(documents, embeddings)
