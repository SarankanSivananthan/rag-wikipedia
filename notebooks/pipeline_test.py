import sys

sys.path.insert(0, "../src")
from rag_pipeline import build_vector_store, chunk_documents, load_corpus, load_qa, passages_to_documents

corpus_df = load_corpus("../data/passages.parquet")
qa_df = load_qa("../data/qa_test.parquet")
print(f"{len(corpus_df)} passages, {len(qa_df)} QA pairs")

documents = passages_to_documents(corpus_df)
chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)
print(f"{len(documents)} passages -> {len(chunks)} chunks")

vector_store = build_vector_store(chunks)
results = vector_store.similarity_search("What is the capital of Uruguay?", k=3)
for r in results:
    print(r.page_content[:120])
