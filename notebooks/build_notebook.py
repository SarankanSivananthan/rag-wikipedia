"""Builds analysis.ipynb from scratch as a sequence of markdown/code cells."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

md = lambda src: cells.append(nbf.v4.new_markdown_cell(src))
code = lambda src: cells.append(nbf.v4.new_code_cell(src))

md("""\
# RAG over Wikipedia — retrieval-augmented question answering

**Goal**: build a working Retrieval-Augmented Generation pipeline end to
end — embed a text corpus, index it in a local vector store, retrieve
relevant chunks for a question, and ground an LLM's answer in them —
while treating retrieval quality as something to *measure*, not assume.

**Dataset**: [`rag-datasets/rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia)
(CC-BY) — 3,200 Wikipedia passages + 918 question/answer pairs, purpose-built
for RAG experimentation so the focus stays on the pipeline rather than data
collection.

**Stack**: Hugging Face (dataset + sentence-transformers) · FAISS ·
LangChain · a pluggable LLM backend (local Hugging Face model by default;
Ollama or OpenAI as drop-in alternatives).
""")

code("""\
import sys
import time

import pandas as pd
import plotly.express as px

sys.path.insert(0, "../src")
from rag_pipeline import (
    answer_question,
    build_vector_store,
    chunk_documents,
    evaluate_retrieval_at_k,
    get_llm,
    load_corpus,
    load_qa,
    passages_to_documents,
    scorable_questions,
)
from qa_metrics import is_abstention
""")

md("## 1. Load the data")

code("""\
corpus_df = load_corpus("../data/passages.parquet")
qa_df = load_qa("../data/qa_test.parquet")
print(f"{len(corpus_df)} passages, {len(qa_df)} question/answer pairs")
corpus_df.head(3)
""")

code("""\
qa_df.head(5)
""")

md("""\
Nearly half the gold answers are just "yes"/"no" — fine for the dataset's
original QA-accuracy purpose, but useless for evaluating *retrieval*
(checking whether "yes" appears in a retrieved chunk says nothing, and the
dataset has no gold passage ids to check against instead). Answers that
*start* with yes/no ("No, it contributed 1.56% to GDP in 2001.") are
excluded too: the full sentence can never appear verbatim in a chunk. The
retrieval evaluation below runs on **every remaining question — no
sampling**.
""")

code("""\
print(qa_df["answer"].str.lower().str.strip(".").value_counts().head(5))
scored = scorable_questions(qa_df)
print(f"\\n{len(qa_df)} questions -> {len(qa_df) - len(scored)} yes/no-style excluded, {len(scored)} scorable")
""")

md("""\
## 2. Chunking strategy

The dataset's passages are already split at a reasonable size, but a real
corpus wouldn't be — so we re-chunk them with LangChain's
`RecursiveCharacterTextSplitter` at a few different sizes and *measure*
which one retrieves best, rather than picking one by feel.
""")

code("""\
documents = passages_to_documents(corpus_df)
print(f"{len(documents)} source documents")
""")

md("""\
The sweep itself (4 chunk sizes x 498 questions, one FAISS index per chunk
size, ~6 minutes on CPU) lives in `src/eval_retrieval.py` and writes
`results/retrieval_hit_at_k.csv`; this notebook loads its output instead
of re-running it. For each question the top 8 chunks are retrieved once,
so hit@1/2/4/8 all come from the same retrieval. **hit@k = the share of
questions where at least one of the top-k chunks contains the gold answer
text.**
""")

code("""\
hit_df = pd.read_csv("../results/retrieval_hit_at_k.csv")
ks = [1, 2, 4, 8]
all_q = hit_df[hit_df["subset"] == "all"]
all_q[["chunk_size", "overlap", "n_chunks"] + [f"hit@{k}" for k in ks]].style.format(
    {"n_chunks": "{:,}", **{f"hit@{k}": "{:.1%}" for k in ks}}
).hide(axis="index")
""")

code("""\
long = pd.concat([
    all_q.assign(k=k, hit=all_q[f"hit@{k}"], ci=all_q[f"ci95@{k}"])[["chunk_size", "k", "hit", "ci"]]
    for k in ks
])
long["chunk_size"] = long["chunk_size"].astype(str)
fig = px.line(
    long, x="k", y="hit", color="chunk_size", error_y="ci", markers=True,
    labels={"k": "k (chunks retrieved)", "hit": "hit@k", "chunk_size": "Chunk size (chars)"},
    title="Retrieval hit@k by chunk size — all 498 scorable questions (bars = 95% Wilson CI)",
)
fig.update_layout(yaxis_tickformat=".0%")
fig.update_xaxes(tickvals=ks)
fig.show()
""")

md("""\
Three things the full evaluation shows:

- **100-character chunks are clearly worse**: 31.5% hit@4 vs. 46-48% for
  larger sizes, a ~15-16 point gap against 95% intervals of about ±4 points.
  That holds on the subset of answers short enough to fit inside every
  chunk size (33% vs. 47-49%, `results/retrieval_hit_at_k.md`), so it is
  not an artifact of chunks too small to ever contain a long answer.
- **250, 500 and 1000 characters are statistically indistinguishable**
  (hit@4: 46.6%, 46.4%, 47.8% — all well inside each other's intervals).
  An earlier 150-question sample suggested 250 "edged out" the others;
  that was sampling noise, and on the full set 1000 is nominally ahead at
  every k. The honest conclusion is "somewhere between 250 and 1000 is
  fine on this corpus," not a winner.
- **More chunks help, with diminishing returns**: going from k=4 to k=8
  adds 3-6 points, while k=1 to k=4 adds 14-16 points for 250-1000
  characters (9 for 100) — at the cost of a longer prompt for the generator.

**250/25 is used for the rest of this notebook** — not because it
retrieves best (it doesn't, measurably), but because it retrieves about
as well as the others while keeping the 4-chunk prompt short for the small
local model.
""")

code("""\
CHUNK_SIZE, CHUNK_OVERLAP = 250, 25
chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
print(f"{len(chunks)} chunks")
""")

md("""\
## 3. Embeddings + FAISS index

Each chunk is embedded with a Hugging Face sentence-transformer
(`all-MiniLM-L6-v2`, 384 dimensions) and indexed in a local FAISS store —
no external database, no network calls at query time.
""")

code("""\
t0 = time.time()
vector_store = build_vector_store(chunks)
print(f"Indexed {len(chunks)} chunks in {time.time() - t0:.0f}s")

vector_store.save_local("../data/faiss_index")
print("Saved to ../data/faiss_index")
""")

md("""\
## 4. Retrieval evaluation on the chosen index

Before trusting any generated answer, we check something more basic:
**does retrieval even surface the right information?** Re-running the
full evaluation on the 250-character index built above (all 498 scorable
questions, no sampling) should reproduce the 250 row of the table in
section 2:
""")

code("""\
eval_results = evaluate_retrieval_at_k(vector_store, qa_df, max_k=8)
print(f"{len(eval_results)} questions evaluated")
for k in ks:
    print(f"hit@{k}: {(eval_results['first_hit_rank'] <= k).mean():.1%}")
""")

md("""\
~47% hit@4 is an honest, unglamorous number — and a deliberately strict
one. The check is exact substring matching of the gold answer, so a
correct passage that phrases the answer differently counts as a miss, and
this metric cannot tell those apart from genuine retrieval failures. That
makes it a lower bound, not a final verdict: good enough to flag *when
retrieval quality changes* (across chunk sizes, embedding models, k),
which is its actual purpose here.
""")

code("""\
print("Examples where none of the top-8 chunks contained the gold answer:")
eval_results[eval_results["first_hit_rank"].isna()][["question", "answer"]].head(5)
""")

md("""\
## 5. Generation, grounded in retrieved context

The LLM only sees the retrieved chunks — it's instructed to say
*"I don't know"* rather than guess when the context doesn't contain the
answer. `get_llm()` supports three interchangeable backends; the local
one needs no setup so it's what runs here:

```python
get_llm(provider="local")                       # Hugging Face model, in-process (used below)
get_llm(provider="ollama", model_name="mistral") # local server via Ollama
get_llm(provider="openai", model_name="gpt-4o-mini")  # hosted, needs OPENAI_API_KEY
```
""")

code("""\
llm = get_llm(provider="local")
""")

code("""\
questions = [
    "Was Abraham Lincoln the sixteenth President of the United States?",
    "What is the capital of Uruguay?",
    "Who discovered penicillin?",
]

for q in questions:
    answer, docs = answer_question(vector_store, llm, q, k=4)
    print(f"Q: {q}")
    print(f"A: {answer}")
    for rank, d in enumerate(docs):
        print(f"   [{rank}] {d.page_content[:140]}...")
    print()
""")

md("""\
The third question is the interesting one: **none of the top-4 retrieved
chunks are about Alexander Fleming** — the actual discoverer of
penicillin. Instead retrieval surfaces a chunk about **Michael Faraday**
(who discovered *benzene*, not penicillin — clearly close enough in
embedding space, as a chemist's biography, to be retrieved, and far
enough in reality to be wrong). This is retrieval failing exactly the
way the intro describes: silently, with no error, no low-confidence
flag — just the wrong context handed to the model. Because the prompt
explicitly permits "I don't know," the model doesn't paper over the gap
with a plausible-sounding guess — but a less carefully prompted
pipeline, or a less careful eyeballing of outputs, would ship that
hallucination straight to a user. It's exactly this kind of failure that
section 4's evaluation step is designed to catch in aggregate rather
than one anecdote at a time.
""")

md("""\
## 6. Generation evaluation on all 918 Q&A

Retrieval hit@k says whether the right text reached the model, not whether
the model *answered correctly*. `src/eval_generation.py` (~1h30 on CPU) runs
three systems on **all 918 questions** and scores their answers against the
gold answers with SQuAD-style **exact match** (EM) and **token F1**
(`src/qa_metrics.py`; lowercase, punctuation and articles stripped). Its
output is committed in `results/` and loaded here:

- **Always "yes"** — a trivial baseline, because 36% of the gold answers are literally "yes"
- **Closed-book** — the same `flan-t5-base` with the question only, no retrieval
- **RAG** — 250-character chunks, top-4, the pipeline from section 5

"I don't know" counts as a wrong answer for EM and F1; its rate is reported
separately. Intervals are 95% bootstrap intervals.
""")

code("""\
gen_metrics = pd.read_csv("../results/generation_metrics.csv")
gen_preds = pd.read_csv("../results/generation_predictions.csv")

def with_ci(value, lo, hi):
    return value.map("{:.1%}".format) + " (" + lo.map("{:.1%}".format) + "–" + hi.map("{:.1%}".format) + ")"

gen_metrics.assign(
    EM=with_ci(gen_metrics["em"], gen_metrics["em_lo"], gen_metrics["em_hi"]),
    F1=with_ci(gen_metrics["f1"], gen_metrics["f1_lo"], gen_metrics["f1_hi"]),
    idk=gen_metrics["abstain_rate"].map(lambda v: "—" if pd.isna(v) else f"{v:.1%}"),
)[["subset", "system", "EM", "F1", "idk"]].rename(columns={"idk": "answered I don't know"})
""")

md("""\
**Read the overall row with care.** Over all 918 questions RAG scores 35.9%
EM — statistically the same as the 36.3% of answering "yes" to everything,
because 46% of the dataset is yes/no questions and 81% of those are "yes".
(On F1 RAG is ahead: 44.6% vs. 36.6%, with non-overlapping intervals.)
The overall number mostly measures the dataset's answer mix, so the two
slices below are the ones that say something about the system.

**Open questions (498): this is where retrieval earns its keep.** Same
model, same questions: RAG reaches 31.5% EM / 47.2% F1, closed-book 1.8% /
8.1%. Without retrieved context `flan-t5-base` almost never knows the
answer.
""")

code("""\
open_q = gen_preds[gen_preds["question_type"] == "open"]
by_retrieval = open_q.groupby("retrieval_hit").agg(
    questions=("question", "size"), exact_match=("rag_em", "mean"),
    f1=("rag_f1", "mean"), answered_idk=("rag_abstained", "mean"),
)
by_retrieval.index = by_retrieval.index.map({True: "gold answer in top-4", False: "gold answer NOT in top-4"})
by_retrieval.style.format({"exact_match": "{:.1%}", "f1": "{:.1%}", "answered_idk": "{:.1%}"})
""")

code("""\
missed = open_q[open_q["retrieval_hit"] == False]
answered_anyway = missed[~missed["rag_abstained"]]
no_overlap = answered_anyway[answered_anyway["rag_f1"] == 0]
print(f"Retrieval missed on {len(missed)} open questions; RAG still answered {len(answered_anyway)} of them,")
print(f"and {len(no_overlap)} of those answers share no word with the gold answer. Examples:\\n")
no_overlap[["question", "gold", "rag_pred"]].head(6)
""")

md("""\
**Generation is bounded by retrieval.** When the gold answer is in the
top-4, RAG gets 71.2% F1 (56.9% EM); when it is not, 26.3% F1 (9.4% EM).
The "I don't know" instruction does react to bad context — it fires 27.8%
of the time on retrieval misses vs. 8.2% on hits — but it is far from a
guarantee: on 192 of the 266 misses the model answered anyway, and 65
of those answers (e.g. "When was Coolidge born?" → "1906") have no word in
common with the gold answer. A few of the zero-overlap cases are the metric
being strict rather than the model being wrong (the gold answer for
Coolidge's state is the typo `MassachuS08_setts`; "23-year" does not match
"23 years"), so EM and F1 here are conservative.

**Yes/no questions (420): the weak spot.** RAG scores 41.2% EM against
79.3% for always saying "yes". Where the answers went:
""")

code("""\
yn = gen_preds[gen_preds["question_type"] == "yes/no"].copy()

def answer_kind(prediction):
    if is_abstention(prediction):
        return "I don't know"
    words = str(prediction).lower().strip().lstrip("\\"' ").split()
    first = words[0].strip(".,!?") if words else ""
    return first if first in ("yes", "no") else "something else"

yn["rag_answer"] = yn["rag_pred"].map(answer_kind)
yn["gold_answer"] = yn["gold"].map(lambda g: "yes" if str(g).lower().strip().startswith("yes") else "no")
print(pd.crosstab(yn["gold_answer"], yn["rag_answer"], margins=True, rownames=["gold"], colnames=["RAG answered"]))

committed = yn[yn["rag_answer"].isin(["yes", "no"])]
right = (committed["rag_answer"] == committed["gold_answer"]).sum()
print(f"\\nRAG committed to yes/no on {len(committed)} of {len(yn)} questions ({len(committed) / len(yn):.1%});")
print(f"on those it picked the right one {right}/{len(committed)} times ({right / len(committed):.1%}).")
""")

code("""\
other = yn[yn["rag_answer"] == "something else"]
length = other["rag_pred"].str.len()
committed_length = committed["rag_pred"].str.len()
print(f"'Something else' answers: median {length.median():.0f} characters ({(length >= 60).sum()} of {len(other)} are 60+),")
print(f"vs. a median of {committed_length.median():.0f} characters when RAG actually says yes or no.\\n")
other.sort_values("rag_pred", key=lambda col: col.str.len())[["question", "gold", "rag_pred"]].iloc[[0, 1, 30, 31, -1]].style.set_properties(
    subset=["rag_pred"], **{"white-space": "pre-wrap"}
)
""")

md("""\
When it says "yes" or "no" it is right 94.8% of the time (184/194) — but it
does that on only 46% of the questions. The rest are "I don't know" (103)
or "something else" (123). Those 123 answers have a median length of 98
characters against 3 for a real yes/no: `flan-t5-base` mostly **copies or
restates a sentence from the context instead of answering yes/no** —
often one that affirms the question ("Canada's two official languages
are English and French"), sometimes one that does not answer it at all
("unopposed"). EM and F1 score all of these as wrong because the format
is wrong, so the 41.2% mixes two things — wrong answers and right
answers in the wrong shape — and I have not separated them. What it does
show is that a small local model with a prompt tuned for refusal handles
open questions far better than yes/no ones. A stronger backend
(`get_llm("ollama")` or `"openai"`) or a prompt asking for "Yes or No"
explicitly are the obvious next experiments.

**Scope of these numbers.** One configuration (250/25 chunks, top-4), one
generator (`flan-t5-base`), one run. They are measured on the dataset's
gold strings, so they carry its quirks: yes/no skew, typos in gold answers,
and exact-string matching that cannot credit a correct answer phrased
differently.
""")

md("""\
## 7. Takeaways

- **Chunk size has a real, measurable effect on retrieval — up to a
  point.** 100-character chunks hit the right context 31.5% of the time
  (hit@4, all 498 scorable questions) vs. 46-48% for 250-1000 characters;
  within 250-1000 the differences are inside the noise. Measured on the
  full question set, not a sample — the earlier 150-question sample had
  produced a spurious "250 wins."
- **Retrieval quality should be measured, not assumed.** A generation
  step has no way to know its context is wrong — it will produce a
  fluent, confident, incorrect answer just as readily as a correct one.
  The hit@k check (`evaluate_retrieval_at_k`) exists specifically to catch
  this before it reaches a user.
- **Grounding the prompt with an explicit "I don't know" escape hatch**
  turns a retrieval failure into a visible non-answer instead of a
  silent hallucination — cheap to add, and demonstrated directly in
  section 5.
- **Answer-level evaluation on all 918 Q&A gives a more sober picture than
  retrieval alone**: 31.5% EM / 47.2% F1 on open questions (vs. 1.8% / 8.1%
  without retrieval), but only 41.2% EM on yes/no questions, where a
  trivial "always yes" scores 79.3%. A single overall EM (35.9%) would have
  hidden both facts — it ties that same trivial baseline.
- **The LLM backend is a swappable detail, not the core of the system**
  — `get_llm()` runs the exact same pipeline against a local
  Hugging Face model, a local Ollama/Mistral server, or the OpenAI API
  with a one-line change.
""")

nb["cells"] = cells
nbf.write(nb, "analysis.ipynb")
print("Wrote analysis.ipynb with", len(cells), "cells")
