import sys
import time

sys.path.insert(0, "../src")
from rag_pipeline import (
    build_vector_store,
    chunk_documents,
    evaluate_retrieval,
    load_corpus,
    load_qa,
    passages_to_documents,
)

corpus_df = load_corpus("../data/passages.parquet")
qa_df = load_qa("../data/qa_test.parquet")
documents = passages_to_documents(corpus_df)

configs = [
    (100, 0),
    (250, 25),
    (500, 50),
    (1000, 100),
]

for chunk_size, overlap in configs:
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=overlap)
    t0 = time.time()
    vs = build_vector_store(chunks)
    build_time = time.time() - t0
    hit_rate, _ = evaluate_retrieval(vs, qa_df, k=4, sample_size=150)
    print(f"chunk_size={chunk_size:>5} overlap={overlap:>4}: {len(chunks):>5} chunks, "
          f"hit_rate={hit_rate:.1%}, build_time={build_time:.0f}s")
