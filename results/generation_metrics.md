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
