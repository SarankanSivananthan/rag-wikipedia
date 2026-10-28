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
6. **Evaluate generation end to end on all 918 Q&A** — exact match and token F1 of the answers the pipeline actually produces, against a closed-book run of the same model and a trivial "always yes" baseline (see [Generation results](#-generation-results))
7. **Inspect a real retrieval failure** — the pipeline doesn't just describe the risk, it demonstrates one

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

## 🎯 Generation results

Retrieval hit@k says whether the right text reached the model, not whether the answer was right. [`src/eval_generation.py`](src/eval_generation.py) scores the pipeline's actual answers on **all 918 questions** with SQuAD-style **exact match** (EM) and token **F1** ([`src/qa_metrics.py`](src/qa_metrics.py): lowercase, punctuation and articles stripped), against two reference systems on the same questions:

- **Always "yes"** — a trivial baseline (36% of the gold answers are literally "yes")
- **Closed-book** — the same generator (`flan-t5-base`) given only the question, no retrieval

RAG here is 250-character chunks, top-4, with the "answer only from the context, otherwise say I don't know" prompt. An "I don't know" counts as wrong for EM/F1; its rate is shown separately. Intervals are 95% bootstrap intervals. Reproduce with `python src/eval_generation.py` (about 1h30 on CPU, resumable); the full per-question predictions are in [`results/generation_predictions.csv`](results/generation_predictions.csv).

**All 918 questions**

| System | Exact match (95% CI) | F1 (95% CI) | Answered "I don't know" |
|---|---|---|---|
| Always "yes" | 36.3% (33.2%–39.5%) | 36.6% (33.5%–39.8%) | — |
| Closed-book (no retrieval) | 1.3% (0.7%–2.1%) | 5.2% (4.2%–6.2%) | 0.0% |
| RAG (chunks 250, top-4) | 35.9% (32.7%–39.1%) | 44.6% (41.6%–47.8%) | 21.4% |

**yes/no-style 420 questions**

| System | Exact match (95% CI) | F1 (95% CI) | Answered "I don't know" |
|---|---|---|---|
| Always "yes" | 79.3% (75.5%–83.3%) | 79.9% (76.2%–83.8%) | — |
| Closed-book (no retrieval) | 0.7% (0.0%–1.7%) | 1.7% (0.9%–2.7%) | 0.0% |
| RAG (chunks 250, top-4) | 41.2% (36.2%–45.7%) | 41.6% (36.7%–46.2%) | 24.5% |

**open 498 questions**

| System | Exact match (95% CI) | F1 (95% CI) | Answered "I don't know" |
|---|---|---|---|
| Always "yes" | 0.0% (0.0%–0.0%) | 0.0% (0.0%–0.0%) | — |
| Closed-book (no retrieval) | 1.8% (0.8%–3.0%) | 8.1% (6.5%–9.9%) | 0.0% |
| RAG (chunks 250, top-4) | 31.5% (27.5%–35.7%) | 47.2% (43.4%–51.1%) | 18.7% |

**RAG on open questions, by retrieval outcome**

| Retrieval | Questions | Exact match | F1 (95% CI) | Answered "I don't know" |
|---|---|---|---|---|
| gold answer in top-4 | 232 | 56.9% | 71.2% (66.4%–76.3%) | 8.2% |
| gold answer NOT in top-4 | 266 | 9.4% | 26.3% (22.1%–30.6%) | 27.8% |

**How to read these numbers**

- **The overall row is not a quality measure.** RAG's 35.9% EM over all 918 questions is statistically the same as the 36.3% of answering "yes" to everything: 46% of the dataset is yes/no questions and 81% of those are "yes". (On F1, RAG is ahead: 44.6% vs. 36.6%.) The two slices below are the informative ones.
- **On open questions retrieval is what makes it work.** Same model, same 498 questions: 31.5% EM / 47.2% F1 with retrieval vs. 1.8% / 8.1% closed-book.
- **Generation is bounded by retrieval.** When the gold answer is in the top-4 chunks, F1 is 71.2% (EM 56.9%); when it isn't, 26.3% (EM 9.4%). The "I don't know" instruction does react to bad context (27.8% of answers on retrieval misses vs. 8.2% on hits) but is no guarantee: on 192 of the 266 misses the model answered anyway, and 65 of those answers share no word with the gold answer.
- **Yes/no questions are the weak spot — 41.2% EM vs. 79.3% for always saying "yes".** When RAG does say "yes" or "no" it is right 184 times out of 194 (94.8%), but it does so on only 46% of these questions. 103 answers are "I don't know", and 123 are something else: sentences copied or restated from the context (median 98 characters, against 3 for a real yes/no). EM/F1 score those as wrong for their format, so the 41.2% mixes wrong answers with right answers in the wrong shape; the split isn't measured here. Reproduced in [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) §6.

**Scope.** One configuration (250/25 chunks, top-4), one generator, one run, scored against the dataset's gold strings — so these numbers inherit its quirks: the yes/no skew, typos in gold answers (e.g. `MassachuS08_setts`), and exact-string matching that can't credit a correct answer phrased differently. A stronger backend (`get_llm("ollama")` / `"openai"`) or a prompt asking explicitly for "Yes or No" are the obvious next experiments; they have not been run.

## 🔎 Key findings

- **100-character chunks are clearly worse; 250-1000 are indistinguishable.** hit@4 is 31.5% at 100 characters vs. 46.4-47.8% at 250/500/1000 — a gap of ~15 points against ±4-point intervals, and it holds on the subset of answers short enough to fit in every chunk size (so it isn't an artifact of chunks too small to contain the answer). Among 250/500/1000 the differences are inside the noise. An earlier version of this project reported "250 edges out the others" from a 150-question sample; the full evaluation shows that was sampling noise.
- **Retrieval failures are silent by default.** Asking *"who discovered penicillin?"* retrieves a passage about Michael Faraday (who discovered benzene) — plausible-adjacent, factually wrong. Nothing in a standard pipeline flags this; the model would confidently answer from the wrong context if not explicitly told it's allowed to refuse.
- **A one-line prompt change (allowing "I don't know") turns that silent failure into a visible non-answer** — cheap insurance against hallucination.
- **Retrieval hit-rate, measured independently of generation, is what actually catches this** — not eyeballing a few outputs.
- **A single end-to-end score can hide more than it shows.** Over all 918 questions RAG's exact match ties the trivial "always yes" baseline; split by question type it is far above it on open questions (31.5% vs. 0%) and far below it on yes/no questions (41.2% vs. 79.3%).

## 📂 Project structure

```
notebooks/
  analysis.ipynb          # full pipeline, executed with real outputs and commentary
  build_notebook.py       # generates analysis.ipynb from source
src/
  rag_pipeline.py         # reusable pipeline: loading, chunking, indexing, retrieval, generation, evaluation
  eval_retrieval.py       # full hit@k evaluation across chunk sizes -> results/
  eval_generation.py      # exact match / F1 of the generated answers on all 918 Q&A -> results/
  qa_metrics.py           # SQuAD-style normalisation, exact match, F1, bootstrap CIs
results/
  retrieval_hit_at_k.md   # the hit@k tables above
  retrieval_hit_at_k.csv  # same numbers + 95% intervals
  retrieval_per_question.csv  # first-hit rank per question x chunk size
  generation_metrics.md   # the EM / F1 tables above
  generation_metrics.csv  # same numbers + 95% bootstrap intervals
  generation_predictions.csv  # every question: gold, each system's answer, scores
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
python src/eval_retrieval.py            # optional: regenerates the hit@k results/ (~6 min)
python src/eval_generation.py           # optional: regenerates the EM/F1 results/ (~1h30 on CPU)
jupyter notebook notebooks/analysis.ipynb
```

The passages/Q&A data is small enough to be included directly. The embedding model (~80MB) and local generation model (~250MB) download automatically on first run.
