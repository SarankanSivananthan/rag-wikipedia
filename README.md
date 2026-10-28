# RAG over Wikipedia

A Retrieval-Augmented Generation pipeline built end to end: embed a text corpus, index it locally, retrieve relevant chunks for a question, and ground an LLM's answer in them — treating retrieval quality as something to *measure*, not assume.

**Stack**: Hugging Face (dataset + sentence-transformers) · FAISS · LangChain · pluggable LLM backend (local Hugging Face model / Ollama / OpenAI)

## 🎯 Goal

Most RAG write-ups show the happy path. This one also shows the pipeline failing — retrieving the wrong context for a question — and the two things that catch that: an explicit "I don't know" instruction in the prompt, and a retrieval-quality metric measured independently of generation.

## 📊 Dataset

[`rag-datasets/rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia) (CC-BY) — built specifically for RAG experimentation, so the focus stays on the pipeline rather than data collection/cleaning:

| | |
|---|---|
| Passages (knowledge base) | 3,200 Wikipedia passages |
| Question/answer pairs (evaluation) | 918 |
| License | CC-BY 3.0 |

## 🏗 Architecture

```
Wikipedia passages
    │  RecursiveCharacterTextSplitter (configurable chunk size/overlap)
    ▼
Chunks
    │  sentence-transformers (all-MiniLM-L6-v2) embeddings
    ▼
FAISS index (local, saved to disk)
    │
    ├──▶ evaluate_retrieval()  — does the top-k context actually contain the gold answer?
    │
    ▼
similarity_search(question, k)
    │
    ▼
Prompt: "answer using only this context; say I don't know if it's not there"
    │
    ▼
LLM (local HF model / Ollama / OpenAI — swappable)
    │
    ▼
Grounded answer + sources shown
```

## 🧪 Methodology

1. **Load** the passages and Q&A pairs
2. **Chunk** the corpus at 4 different sizes (100/250/500/1000 chars) using LangChain's `RecursiveCharacterTextSplitter`
3. **Embed & index** each chunking variant with `sentence-transformers/all-MiniLM-L6-v2` into a FAISS store
4. **Evaluate retrieval** at each chunk size on **every scorable question — no sampling**: does *any* of the top-k retrieved chunks contain the gold answer text? Reported as hit@1/2/4/8 (see [Retrieval results](#-retrieval-results)). 420 of the 918 questions have yes/no-style answers, which can't be checked by substring match, leaving 498.
5. **Generate** answers grounded strictly in retrieved context, with an explicit instruction to say "I don't know" rather than guess
6. **Inspect a real retrieval failure** — the pipeline doesn't just describe the risk, it demonstrates one

Full walkthrough with code and real outputs: [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb).

## 📈 Retrieval results

hit@k = share of questions where at least one of the top-k retrieved chunks contains the gold answer text. All 918 questions were run; the 498 with a checkable (non yes/no) answer are scored. Embeddings: `all-MiniLM-L6-v2`. Reproduce with `python src/eval_retrieval.py` (~6 min on CPU).

**All 498 scorable questions** (of 918; yes/no-style answers excluded)

| Chunk size | Overlap | Chunks | hit@1 | hit@2 | hit@4 | hit@8 |
|---|---|---|---|---|---|---|
| 100 | 0 | 14,416 | 22.3% | 26.1% | 31.5% | 34.9% |
| 250 | 25 | 7,050 | 31.1% | 39.8% | 46.6% | 50.2% |
| 500 | 50 | 4,492 | 32.5% | 41.6% | 46.4% | 52.2% |
| 1000 | 100 | 3,405 | 33.5% | 42.4% | 47.8% | 53.0% |

**Answers of at most 100 characters only (471 questions)** — the subset that physically fits inside every chunk size, so no row is penalised for chunks too small to ever contain the gold answer

| Chunk size | Overlap | Chunks | hit@1 | hit@2 | hit@4 | hit@8 |
|---|---|---|---|---|---|---|
| 100 | 0 | 14,416 | 23.6% | 27.6% | 33.3% | 36.9% |
| 250 | 25 | 7,050 | 32.1% | 40.6% | 47.8% | 51.6% |
| 500 | 50 | 4,492 | 33.3% | 42.9% | 47.3% | 53.3% |
| 1000 | 100 | 3,405 | 34.4% | 43.7% | 48.8% | 53.9% |

With n≈500 each cell carries a 95% interval of about ±4 points (per-cell Wilson intervals are in [`results/retrieval_hit_at_k.csv`](results/retrieval_hit_at_k.csv); per-question first-hit ranks in `results/retrieval_per_question.csv`).

## 🔎 Key findings

- **100-character chunks are clearly worse; 250-1000 are indistinguishable.** hit@4 is 31.5% at 100 characters vs. 46.4-47.8% at 250/500/1000 — a gap of ~15 points against ±4-point intervals, and it holds on the subset of answers short enough to fit in every chunk size (so it isn't an artifact of chunks too small to contain the answer). Among 250/500/1000 the differences are inside the noise. An earlier version of this project reported "250 edges out the others" from a 150-question sample; the full evaluation shows that was sampling noise.
- **Retrieval failures are silent by default.** Asking *"who discovered penicillin?"* retrieves a passage about Michael Faraday (who discovered benzene) — plausible-adjacent, factually wrong. Nothing in a standard pipeline flags this; the model would confidently answer from the wrong context if not explicitly told it's allowed to refuse.
- **A one-line prompt change (allowing "I don't know") turns that silent failure into a visible non-answer** — cheap insurance against hallucination.
- **Retrieval hit-rate, measured independently of generation, is what actually catches this** — not eyeballing a few outputs.

## 📂 Project structure

```
notebooks/
  analysis.ipynb          # full pipeline, executed with real outputs and commentary
  build_notebook.py       # generates analysis.ipynb from source
src/
  rag_pipeline.py         # reusable pipeline: loading, chunking, indexing, retrieval, generation, evaluation
  eval_retrieval.py       # full hit@k evaluation across chunk sizes -> results/
results/
  retrieval_hit_at_k.md   # the hit@k tables above
  retrieval_hit_at_k.csv  # same numbers + 95% intervals
  retrieval_per_question.csv  # first-hit rank per question x chunk size
data/
  passages.parquet        # Wikipedia passages (included, <1MB)
  qa_test.parquet         # evaluation Q&A pairs (included, <100KB)
  faiss_index/            # generated by running the notebook (not versioned)
```

## 🧠 Swapping the LLM backend

`get_llm()` in `src/rag_pipeline.py` supports three interchangeable backends — the rest of the pipeline (chunking, retrieval, evaluation) is identical regardless of which one answers:

```python
get_llm(provider="local")                              # Hugging Face model, in-process — no setup, used in the notebook
get_llm(provider="ollama", model_name="mistral")        # local server via Ollama — `ollama pull mistral` first
get_llm(provider="openai", model_name="gpt-4o-mini")    # hosted — needs OPENAI_API_KEY set in the environment
```

## 🚀 Running it

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/eval_retrieval.py            # optional: regenerates results/ (~6 min)
jupyter notebook notebooks/analysis.ipynb
```

The passages/Q&A data is small enough to be included directly. The embedding model (~80MB) and local generation model (~250MB) download automatically on first run.
