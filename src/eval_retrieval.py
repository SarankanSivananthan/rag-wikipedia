"""
Full retrieval evaluation across chunk sizes: every scorable question in
the 918-question set (no sampling), hit@k for k in KS, published as a
table.

Run from the project root: python src/eval_retrieval.py
Outputs (in results/):
  - retrieval_hit_at_k.csv       one row per chunk size, hit@k + 95% CIs
  - retrieval_hit_at_k.md        the same, as markdown tables for the README
  - retrieval_per_question.csv   first-hit rank for every question x chunk size
"""
from __future__ import annotations

import math
import os
import time

import pandas as pd

from rag_pipeline import (
    build_vector_store,
    chunk_documents,
    evaluate_retrieval_at_k,
    load_corpus,
    load_qa,
    passages_to_documents,
    scorable_questions,
)

CONFIGS = [(100, 0), (250, 25), (500, 50), (1000, 100)]  # (chunk_size, overlap)
KS = (1, 2, 4, 8)
FAIR_MAX_LEN = 100  # answers that fit inside even the smallest chunk
OUT_DIR = "results"


def wilson_half_width(hits: int, n: int, z: float = 1.96) -> float:
    """Half-width of the Wilson score interval for a proportion."""
    if n == 0:
        return float("nan")
    p = hits / n
    denom = 1 + z**2 / n
    return z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom


def summarize(per_question: pd.DataFrame, chunk_size: int, overlap: int, n_chunks: int, subset: str) -> dict:
    n = len(per_question)
    row = {"chunk_size": chunk_size, "overlap": overlap, "n_chunks": n_chunks, "subset": subset, "n_questions": n}
    for k in KS:
        hits = int((per_question["first_hit_rank"] <= k).sum())
        row[f"hit@{k}"] = hits / n
        row[f"ci95@{k}"] = wilson_half_width(hits, n)
    return row


def to_markdown(summary: pd.DataFrame, subset: str) -> str:
    sub = summary[summary["subset"] == subset]
    header = "| Chunk size | Overlap | Chunks | " + " | ".join(f"hit@{k}" for k in KS) + " |"
    sep = "|---|---|---|" + "---|" * len(KS)
    lines = [header, sep]
    for d in sub.to_dict("records"):  # not itertuples: it mangles "hit@k" column names
        cells = " | ".join(f"{d[f'hit@{k}']:.1%}" for k in KS)
        lines.append(f"| {d['chunk_size']} | {d['overlap']} | {d['n_chunks']:,} | {cells} |")
    return "\n".join(lines)


def main():
    corpus_df = load_corpus("data/passages.parquet")
    qa_df = load_qa("data/qa_test.parquet")
    documents = passages_to_documents(corpus_df)

    n_scored = len(scorable_questions(qa_df))
    print(f"{len(qa_df)} questions total, {n_scored} scorable (yes/no-style answers excluded)", flush=True)

    summaries, per_question_all = [], []
    for chunk_size, overlap in CONFIGS:
        chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=overlap)
        t0 = time.time()
        vs = build_vector_store(chunks)
        print(f"chunk_size={chunk_size}: {len(chunks)} chunks indexed in {time.time() - t0:.0f}s", flush=True)

        per_question = evaluate_retrieval_at_k(vs, qa_df, max_k=max(KS))
        per_question.insert(0, "chunk_size", chunk_size)
        per_question_all.append(per_question)

        summaries.append(summarize(per_question, chunk_size, overlap, len(chunks), "all"))
        fair = per_question[per_question["answer_len"] <= FAIR_MAX_LEN]
        summaries.append(summarize(fair, chunk_size, overlap, len(chunks), f"answers<={FAIR_MAX_LEN}chars"))
        print(f"  hit@k (all {len(per_question)} scorable): "
              + ", ".join(f"@{k}={summaries[-2][f'hit@{k}']:.1%}" for k in KS), flush=True)

    summary = pd.DataFrame(summaries)
    os.makedirs(OUT_DIR, exist_ok=True)
    summary.to_csv(f"{OUT_DIR}/retrieval_hit_at_k.csv", index=False)
    pd.concat(per_question_all).to_csv(f"{OUT_DIR}/retrieval_per_question.csv", index=False)

    write_markdown(summary, n_scored, len(qa_df))


def write_markdown(summary: pd.DataFrame, n_scored: int, n_total: int) -> None:
    fair_subset = f"answers<={FAIR_MAX_LEN}chars"
    n_fair = int(summary[summary["subset"] == fair_subset]["n_questions"].iloc[0])
    md = (
        f"**All {n_scored} scorable questions** (of {n_total}; yes/no-style answers excluded)\n\n"
        + to_markdown(summary, "all")
        + f"\n\n**Answers of at most {FAIR_MAX_LEN} characters only ({n_fair} questions)** — "
        "the subset that physically fits inside every chunk size, so no row is penalised "
        "for chunks too small to ever contain the gold answer\n\n"
        + to_markdown(summary, fair_subset)
        + "\n"
    )
    with open(f"{OUT_DIR}/retrieval_hit_at_k.md", "w") as f:
        f.write(md)
    print("\n" + md, flush=True)


if __name__ == "__main__":
    main()
