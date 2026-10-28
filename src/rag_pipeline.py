"""
Reusable components for the Wikipedia RAG pipeline: data loading, chunking,
vector indexing, retrieval, generation, and a simple retrieval-relevance
evaluation.

Kept as plain functions (rather than a framework-y class hierarchy) so the
notebook can call each step individually and inspect intermediate results.
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
    """Re-chunk the (already passage-sized) corpus at a configurable
    granularity, so chunk size/overlap can be compared like a real corpus
    that wasn't pre-split by the dataset authors.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def build_vector_store(
    documents: list[Document], embedding_model: str = DEFAULT_EMBEDDING_MODEL
) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return FAISS.from_documents(documents, embeddings)


def get_llm(provider: str = "local", model_name: str | None = None):
    """Pluggable generation backend.

    - "local": a small Hugging Face seq2seq model running in-process — no
      account, no server, works out of the box. Used as the default so the
      pipeline is runnable without any external setup.
    - "ollama": a locally-served model (e.g. Mistral, Llama) via Ollama.
      Requires Ollama installed and the model pulled (`ollama pull mistral`).
    - "openai": a hosted model via the OpenAI API. Requires
      `OPENAI_API_KEY` set in the environment.
    """
    if provider == "local":
        from transformers import pipeline
        from langchain_huggingface import HuggingFacePipeline

        pipe = pipeline(
            "text2text-generation",
            model=model_name or "google/flan-t5-base",
            max_new_tokens=200,
        )
        return HuggingFacePipeline(pipeline=pipe)

    if provider == "ollama":
        from langchain_ollama import OllamaLLM

        return OllamaLLM(model=model_name or "mistral")

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name or "gpt-4o-mini", temperature=0)

    raise ValueError(f"Unknown provider: {provider!r} (expected local/ollama/openai)")


RAG_PROMPT = """Answer the question using only the context below. \
If the context doesn't contain the answer, say "I don't know" — do not guess.

Context:
{context}

Question: {question}

Answer:"""


CLOSED_BOOK_PROMPT = """Answer the question as briefly as possible.

Question: {question}

Answer:"""


def build_rag_prompt(docs: list[Document], question: str) -> str:
    context = "\n\n".join(doc.page_content for doc in docs)
    return RAG_PROMPT.format(context=context, question=question)


def answer_question(vector_store: FAISS, llm, question: str, k: int = 4):
    """Retrieve top-k chunks and generate an answer grounded in them.
    Returns (answer, source_documents) so the sources can be shown.
    """
    docs = vector_store.similarity_search(question, k=k)
    answer = llm.invoke(build_rag_prompt(docs, question))
    if hasattr(answer, "content"):  # chat models return a message object
        answer = answer.content
    return answer, docs


def generate_answers(llm, prompts: list[str]) -> list[str]:
    """Run many prompts through any backend from `get_llm()`.

    Text-completion LLMs (the local Hugging Face pipeline, Ollama) go
    through `generate()`, which batches internally; chat models (OpenAI)
    go through `batch()` and return message objects.
    """
    from langchain_core.language_models.llms import BaseLLM

    if isinstance(llm, BaseLLM):
        return [gen[0].text for gen in llm.generate(prompts).generations]
    return [msg.content for msg in llm.batch(prompts)]


def scorable_questions(qa_df: pd.DataFrame) -> pd.DataFrame:
    """Questions whose gold answer can be checked by substring match.

    Yes/no-style answers ("yes", "No, it contributed 1.56% to GDP in 2001.")
    are excluded: looking for "yes" in a retrieved chunk is close to
    meaningless (the dataset has no gold passage ids to check against
    instead), and the longer ones can never appear verbatim in a chunk.
    Either way they would distort the hit rate without saying anything
    about retrieval quality. On the 918 questions this leaves 498.
    """
    answers = qa_df["answer"].astype(str).str.lower().str.strip().str.strip(".")
    return qa_df[~answers.str.match(r"^(yes|no)\b")]


def evaluate_retrieval_at_k(
    vector_store: FAISS, qa_df: pd.DataFrame, max_k: int = 8
) -> pd.DataFrame:
    """Full (non-sampled) retrieval evaluation over every scorable question.

    Retrieves the top `max_k` chunks once per question and records the
    1-based rank of the first chunk containing the gold answer text
    (NaN = not in the top max_k). hit@k for any k <= max_k is then simply
    `first_hit_rank <= k`, so every k comes from the same retrieval.
    """
    scored = scorable_questions(qa_df)
    # Queries are embedded in one batch and searched by vector — identical
    # to similarity_search(), without re-loading the model per question.
    vectors = vector_store.embeddings.embed_documents(scored["question"].tolist())

    rows = []
    for row, vec in zip(scored.itertuples(), vectors):
        docs = vector_store.similarity_search_by_vector(vec, k=max_k)
        answer_lower = str(row.answer).lower()
        rank = next(
            (i + 1 for i, doc in enumerate(docs) if answer_lower in doc.page_content.lower()),
            None,
        )
        rows.append({
            "question": row.question,
            "answer": row.answer,
            "answer_len": len(str(row.answer)),
            "first_hit_rank": rank,
        })
    return pd.DataFrame(rows)


def evaluate_retrieval(
    vector_store: FAISS,
    qa_df: pd.DataFrame,
    k: int = 4,
    sample_size: int | None = 150,
    seed: int = 42,
) -> tuple[float, pd.DataFrame]:
    """A simple but real relevance check: for each (optionally sampled)
    question, does ANY of the top-k retrieved chunks actually contain the
    gold answer text? This is what silently fails when chunking/embeddings
    are bad — the LLM then generates a plausible-sounding but ungrounded
    answer. `sample_size=None` evaluates every scorable question; see
    `evaluate_retrieval_at_k` for the full, multi-k version.
    """
    non_yes_no = scorable_questions(qa_df)
    sample = (
        non_yes_no
        if sample_size is None
        else non_yes_no.sample(n=min(sample_size, len(non_yes_no)), random_state=seed)
    )

    rows = []
    for row in sample.itertuples():
        docs = vector_store.similarity_search(row.question, k=k)
        answer_lower = str(row.answer).lower()
        hit = any(answer_lower in doc.page_content.lower() for doc in docs)
        rows.append({"question": row.question, "answer": row.answer, "hit": hit})

    results = pd.DataFrame(rows)
    hit_rate = results["hit"].mean()
    return hit_rate, results
