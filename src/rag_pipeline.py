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


def get_llm(provider: str = "local", model_name: str | None = None):
    """Generation backend. Only a local Hugging Face model for now."""
    if provider == "local":
        from transformers import pipeline
        from langchain_huggingface import HuggingFacePipeline

        pipe = pipeline(
            "text2text-generation",
            model=model_name or "google/flan-t5-base",
            max_new_tokens=200,
        )
        return HuggingFacePipeline(pipeline=pipe)

    raise ValueError(f"Unknown provider: {provider!r} (expected local)")


RAG_PROMPT = """Answer the question using only the context below. \
If the context doesn't contain the answer, say "I don't know" — do not guess.

Context:
{context}

Question: {question}

Answer:"""


def answer_question(vector_store: FAISS, llm, question: str, k: int = 4):
    """Retrieve top-k chunks and generate an answer grounded in them.
    Returns (answer, source_documents) so the sources can be shown.
    """
    docs = vector_store.similarity_search(question, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = RAG_PROMPT.format(context=context, question=question)
    answer = llm.invoke(prompt)
    if hasattr(answer, "content"):  # chat models return a message object
        answer = answer.content
    return answer, docs
