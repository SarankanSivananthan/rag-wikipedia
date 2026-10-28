"""
End-to-end generation evaluation on all 918 question/answer pairs:
exact match and token F1 of the answers the RAG pipeline actually
produces, against the gold answers.

Three systems are scored on identical questions, so the RAG number has
something to be compared to:
  - Always "yes"      : a trivial baseline (a third of the gold answers are "yes")
  - Closed-book       : the same LLM given only the question, no retrieval
  - RAG               : retrieve top-k chunks, then generate (the pipeline under test)

Run from the project root: python src/eval_generation.py
Options: --limit N   score only the first N questions (smoke test; writes
                     to --out-dir, so point that somewhere disposable)
Outputs (in results/):
  - generation_metrics.csv       EM / F1 (+95% bootstrap CIs) per system x question subset
  - generation_metrics.md        the same, as markdown tables for the README
  - generation_predictions.csv   every question: gold, each system's answer and scores
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import pandas as pd

from qa_metrics import bootstrap_ci, exact_match, f1_score, is_abstention
from rag_pipeline import (
    CLOSED_BOOK_PROMPT,
    build_rag_prompt,
    build_vector_store,
    chunk_documents,
    generate_answers,
    get_llm,
    load_corpus,
    load_qa,
    passages_to_documents,
    scorable_questions,
)

CHUNK_SIZE, CHUNK_OVERLAP, K = 250, 25, 4  # the configuration used in the notebook
SLICE = 64  # prompts generated between progress lines


def generate_in_slices(llm, prompts: list[str], label: str, cache_path: str) -> list[str]:
    """Generate in slices, checkpointing to `cache_path` after each one so a
    crash partway through a ~1h CPU run resumes instead of restarting. The
    cache is keyed on a hash of the prompts, so a stale one is ignored."""
    key = hashlib.sha256("\n\x00\n".join(prompts).encode()).hexdigest()
    outputs = []
    if os.path.exists(cache_path):
        cached = json.load(open(cache_path))
        if cached["key"] == key:
            outputs = cached["outputs"]
            print(f"  [{label}] resuming from {len(outputs)}/{len(prompts)} cached", flush=True)

    t0 = time.time()
    for i in range(len(outputs), len(prompts), SLICE):
        outputs += generate_answers(llm, prompts[i : i + SLICE])
        json.dump({"key": key, "outputs": outputs}, open(cache_path, "w"))
        print(f"  [{label}] {len(outputs)}/{len(prompts)} generated ({time.time() - t0:.0f}s)", flush=True)
    return outputs


def score(df: pd.DataFrame, system: str, pred_col: str) -> pd.DataFrame:
    df[f"{system}_em"] = [exact_match(p, g) for p, g in zip(df[pred_col], df["gold"])]
    df[f"{system}_f1"] = [f1_score(p, g) for p, g in zip(df[pred_col], df["gold"])]
    return df


def summarize(df: pd.DataFrame, systems: dict[str, str]) -> pd.DataFrame:
    subsets = {
        f"all {len(df)}": df,
        f"yes/no-style {int((df['question_type'] == 'yes/no').sum())}": df[df["question_type"] == "yes/no"],
        f"open {int((df['question_type'] == 'open').sum())}": df[df["question_type"] == "open"],
    }
    rows = []
    for subset_name, sub in subsets.items():
        for key, label in systems.items():
            em_lo, em_hi = bootstrap_ci(sub[f"{key}_em"])
            f1_lo, f1_hi = bootstrap_ci(sub[f"{key}_f1"])
            rows.append({
                "system": label, "subset": subset_name, "n": len(sub),
                "em": sub[f"{key}_em"].mean(), "em_lo": em_lo, "em_hi": em_hi,
                "f1": sub[f"{key}_f1"].mean(), "f1_lo": f1_lo, "f1_hi": f1_hi,
                "abstain_rate": sub[f"{key}_abstained"].mean() if f"{key}_abstained" in sub else float("nan"),
            })
    return pd.DataFrame(rows)


def by_retrieval_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """RAG answer quality on open questions, split by whether the top-k
    chunks actually contained the gold answer text."""
    open_q = df[df["question_type"] == "open"]
    rows = []
    for hit, sub in open_q.groupby("retrieval_hit"):
        em_lo, em_hi = bootstrap_ci(sub["rag_em"])
        f1_lo, f1_hi = bootstrap_ci(sub["rag_f1"])
        rows.append({
            "retrieval": f"gold answer in top-{K}" if hit else f"gold answer NOT in top-{K}",
            "n": len(sub), "em": sub["rag_em"].mean(), "f1": sub["rag_f1"].mean(),
            "f1_lo": f1_lo, "f1_hi": f1_hi, "abstain_rate": sub["rag_abstained"].mean(),
        })
    return pd.DataFrame(rows).sort_values("retrieval", ascending=False)


def ci_cell(value: float, lo: float, hi: float) -> str:
    return f"{value:.1%} ({lo:.1%}–{hi:.1%})"


def to_markdown(summary: pd.DataFrame, retrieval: pd.DataFrame, n_total: int) -> str:
    parts = []
    for subset in summary["subset"].unique():
        sub = summary[summary["subset"] == subset]
        lines = [f"**{subset} questions**" if not subset.startswith("all") else f"**All {n_total} questions**", "",
                 "| System | Exact match (95% CI) | F1 (95% CI) | Answered \"I don't know\" |", "|---|---|---|---|"]
        for r in sub.to_dict("records"):
            abstain = "—" if pd.isna(r["abstain_rate"]) else f"{r['abstain_rate']:.1%}"
            lines.append(f"| {r['system']} | {ci_cell(r['em'], r['em_lo'], r['em_hi'])} | "
                         f"{ci_cell(r['f1'], r['f1_lo'], r['f1_hi'])} | {abstain} |")
        parts.append("\n".join(lines))

    lines = [f"**RAG on open questions, by retrieval outcome**", "",
             "| Retrieval | Questions | Exact match | F1 (95% CI) | Answered \"I don't know\" |", "|---|---|---|---|---|"]
    for r in retrieval.to_dict("records"):
        lines.append(f"| {r['retrieval']} | {r['n']} | {r['em']:.1%} | {ci_cell(r['f1'], r['f1_lo'], r['f1_hi'])} | {r['abstain_rate']:.1%} |")
    parts.append("\n".join(lines))
    return "\n\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    corpus_df = load_corpus("data/passages.parquet")
    qa_df = load_qa("data/qa_test.parquet")
    if args.limit:
        qa_df = qa_df.head(args.limit)
    documents = passages_to_documents(corpus_df)
    print(f"{len(qa_df)} questions to evaluate", flush=True)

    t0 = time.time()
    vs = build_vector_store(chunk_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP))
    print(f"Index built in {time.time() - t0:.0f}s", flush=True)

    questions = qa_df["question"].tolist()
    golds = qa_df["answer"].astype(str).tolist()
    vectors = vs.embeddings.embed_documents(questions)
    retrieved = [vs.similarity_search_by_vector(v, k=K) for v in vectors]

    os.makedirs(args.out_dir, exist_ok=True)
    llm = get_llm(provider="local")
    llm.batch_size = 16
    rag_cache = f"{args.out_dir}/.cache_rag.json"
    closed_cache = f"{args.out_dir}/.cache_closed.json"
    rag_preds = generate_in_slices(llm, [build_rag_prompt(d, q) for d, q in zip(retrieved, questions)], "RAG", rag_cache)
    closed_preds = generate_in_slices(llm, [CLOSED_BOOK_PROMPT.format(question=q) for q in questions], "closed-book", closed_cache)

    open_idx = set(scorable_questions(qa_df).index)
    df = pd.DataFrame({
        "question": questions,
        "gold": golds,
        "question_type": ["open" if i in open_idx else "yes/no" for i in qa_df.index],
        "rag_pred": [p.strip() for p in rag_preds],
        "closed_pred": [p.strip() for p in closed_preds],
        "yes_pred": "yes",
        "retrieval_hit": pd.array(
            [any(g.lower() in d.page_content.lower() for d in docs) for g, docs in zip(golds, retrieved)],
            dtype="boolean",
        ),
    })
    # Retrieval outcome only means something where the gold answer is a checkable string.
    df.loc[df["question_type"] == "yes/no", "retrieval_hit"] = pd.NA

    for key, col in [("rag", "rag_pred"), ("closed", "closed_pred"), ("yes", "yes_pred")]:
        score(df, key, col)
    df["rag_abstained"] = df["rag_pred"].map(is_abstention)
    df["closed_abstained"] = df["closed_pred"].map(is_abstention)

    open_q = df[df["question_type"] == "open"]
    print(f"Sanity check — retrieval hit@{K} on the {len(open_q)} open questions: "
          f"{open_q['retrieval_hit'].mean():.1%} (retrieval evaluation reported 46.6% at 250/25)", flush=True)

    systems = {"yes": 'Always "yes"', "closed": "Closed-book (no retrieval)", "rag": f"RAG (chunks {CHUNK_SIZE}, top-{K})"}
    summary = summarize(df, systems)
    summary.loc[summary["system"] == systems["yes"], "abstain_rate"] = float("nan")
    retrieval = by_retrieval_outcome(df)

    summary.to_csv(f"{args.out_dir}/generation_metrics.csv", index=False)
    df.to_csv(f"{args.out_dir}/generation_predictions.csv", index=False)
    md = to_markdown(summary, retrieval, len(df))
    with open(f"{args.out_dir}/generation_metrics.md", "w") as f:
        f.write(md)
    os.remove(rag_cache)
    os.remove(closed_cache)
    print("\n" + md, flush=True)


if __name__ == "__main__":
    main()
