"""
Reusable components for the Wikipedia RAG pipeline: data loading,
chunking, and vector indexing. More to come (retrieval eval, generation).

Pinned numpy<2 in requirements.txt — sentence-transformers/torch on this
machine segfaulted against numpy 2.x.
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
        Document(page_content=row.passage, metadata={"passage_id": int(row.Index)})
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


def evaluate_retrieval(
    vector_store: FAISS,
    qa_df: pd.DataFrame,
    k: int = 4,
    sample_size: int = 150,
    seed: int = 42,
) -> tuple[float, pd.DataFrame]:
    """For each sampled question, does ANY of the top-k retrieved chunks
    actually contain the gold answer text? This is what silently fails
    when chunking/embeddings are bad.

    Yes/no answers are excluded: checking whether "yes" appears in a
    retrieved chunk is close to meaningless, so they'd inflate the hit
    rate without saying anything about retrieval quality.
    """
    non_yes_no = qa_df[~qa_df["answer"].str.lower().str.strip(".").isin({"yes", "no"})]
    sample = non_yes_no.sample(n=min(sample_size, len(non_yes_no)), random_state=seed)

    rows = []
    for row in sample.itertuples():
        docs = vector_store.similarity_search(row.question, k=k)
        answer_lower = str(row.answer).lower()
        hit = any(answer_lower in doc.page_content.lower() for doc in docs)
        rows.append({"question": row.question, "answer": row.answer, "hit": hit})

    results = pd.DataFrame(rows)
    hit_rate = results["hit"].mean()
    return hit_rate, results
