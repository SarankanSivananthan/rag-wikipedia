import sys
import time

sys.path.insert(0, "../src")
from rag_pipeline import (
    answer_question,
    build_vector_store,
    chunk_documents,
    evaluate_retrieval,
    get_llm,
    load_corpus,
    load_qa,
    passages_to_documents,
)

print("Loading data...")
corpus_df = load_corpus("../data/passages.parquet")
qa_df = load_qa("../data/qa_test.parquet")
print(f"  {len(corpus_df)} passages, {len(qa_df)} QA pairs")

documents = passages_to_documents(corpus_df)

print("\nChunking (500/50)...")
chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)
print(f"  {len(documents)} passages -> {len(chunks)} chunks")

print("\nBuilding FAISS index (this downloads the embedding model on first run)...")
t0 = time.time()
vector_store = build_vector_store(chunks)
print(f"  done in {time.time()-t0:.0f}s")

vector_store.save_local("../data/faiss_index")
print("  saved to ../data/faiss_index")

print("\nEvaluating retrieval (k=4, 150 non-yes/no questions)...")
t0 = time.time()
hit_rate, results = evaluate_retrieval(vector_store, qa_df, k=4, sample_size=150)
print(f"  hit rate: {hit_rate:.1%} ({time.time()-t0:.0f}s)")

print("\nLoading local LLM (flan-t5-base)...")
t0 = time.time()
llm = get_llm(provider="local")
print(f"  loaded in {time.time()-t0:.0f}s")

print("\nSample Q&A:")
for q in [
    "Was Abraham Lincoln the sixteenth President of the United States?",
    "What is the capital of Uruguay?",
    "Who discovered penicillin?",
]:
    answer, docs = answer_question(vector_store, llm, q, k=4)
    print(f"\nQ: {q}")
    print(f"A: {answer}")
    print(f"   (top source: {docs[0].page_content[:100]}...)")

print("\nDone.")
