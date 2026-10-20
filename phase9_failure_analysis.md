# Phase 9 — Failure Analysis: Where the GDPR RAG System Actually Breaks

**Project:** Chat-with-your-documents RAG system over GDPR (Regulation EU 2016/679)
**Scope of this analysis:** The deployed production configuration only — structure-aware chunking, no reranker (`structure`, `use_rerank=False`), as served by the live API at `gdpr-rag-api.onrender.com`.
**Date:** Phase 9, following Phases 1–8 (all complete and verified).

---

## 1. Why This Analysis Exists

The Phase 4 evaluation harness reported near-ceiling aggregate scores for the deployed configuration:

| Metric | Score |
|---|---|
| Recall@5 | 1.000 |
| Faithfulness | 0.997 |
| Citation-clean rate | 1.000 |
| Abstention accuracy | 1.000 |

An aggregate score this close to perfect is not evidence of a flawless system — it is evidence that the existing 27-question golden set is not sensitive enough to separate "correct" from "correct for the wrong reason." Recent RAG failure-taxonomy research backs this concern directly: cross-tier interactions between pipeline stages are estimated to account for roughly 15% of production failures and are invisible to metrics that score each pipeline stage independently, and current detection methods are estimated to miss as much as 83% of production RAG failures in aggregate [1]. A system can score 1.000 on recall@5 while still retrieving the *wrong specific chunk* within a correct article, or generate a *true* statement that isn't actually grounded in anything retrieved. Neither of those is visible in an aggregate number.

This analysis does not reject the Phase 4 numbers — they are correct and have been independently re-verified against the raw per-question results during this phase (see §2.1). The point is that aggregate metrics and failure analysis answer different questions: one asks "how often does it work," the other asks "when it doesn't quite work, why not, and does it matter."

---

## 2. Methodology

### 2.1 Data integrity check (performed before any analysis)

Two result files existed in `phase4_eval/`: `full_results.json` (162 entries) and `results_checkpoint.json` (174 entries). Cross-checking them surfaced a discrepancy — the same question, same configuration, same generated answer had *different* recall@5 scores across the two files. Investigation showed `results_checkpoint.json` contains 39 entries for one configuration instead of the expected 27, and many entries are missing the `retrieval_scores` field entirely. This identifies it as a stale, partially-overwritten checkpoint from an interrupted earlier run, not a valid second dataset. It was excluded. `full_results.json` was confirmed clean (exactly 162 = 6 configs × 27 questions, schema-complete) and used as the sole source of truth. The Phase 4 summary numbers were independently re-derived from it and matched exactly.

### 2.2 Source of failure cases

Per the project's stated approach, a **fine-grained per-question re-analysis** of the existing 27 golden questions was performed first, rather than immediately constructing a new adversarial question set. For each question, beyond the pass/fail aggregate flags, the following were extracted:

- **Gold-article rank** — derived from MRR (`rank = round(1/MRR)`), not just whether the gold article appeared somewhere in the top 5
- **Citation exactness** — any hallucinated citations
- **Claim-level faithfulness** — any individual claim in the answer marked unsupported by the LLM-judge, not just the aggregate faithfulness score
- **Relevance score and judge reasoning**
- **Abstention correctness**, for the 4 unanswerable questions

11 of 27 questions (41%) showed a non-ceiling signal on at least one of these axes. This was judged sufficient to produce a substantive, evidenced analysis — the planned fallback (constructing a new adversarial/stress-test question set) was not needed and was not built, keeping this phase within the existing free-tier infrastructure with no new evaluation runs required.

### 2.3 Taxonomy adopted

A **three-tier structure — Retrieval / Augmentation / Generation —** is used as the top-level frame, following a 2026 synthesis drawn from 150+ papers on RAG failures in academic and industrial deployments [1]. Within that frame, specific named sub-modes are borrowed from two complementary, more granular taxonomies where they map directly onto what was actually observed: **Position-of-Gold Bias** and **Citation Drift** from a 12-mode production taxonomy [2], and **Multi-Hop Reasoning Gaps** / **Chunking Boundary Errors** from a 33-mode, 7-pipeline-stage taxonomy [3]. The second finding (§3.2) did not fit any RAG failure mode in the literature reviewed, because it is not a RAG failure — it is flagged separately as an evaluation-methodology gap.

---

## 3. Findings

### 3.1 Finding 1 — Missing-Evidence Hallucination Behind a Legally Correct Answer

**Tier:** Generation. **Named mode:** Faithfulness Violation (with a missing-evidence hallucination mechanism) [1].

**Question (Q9):** *"What is the 'right to be forgotten' and when does it apply?"* (Gold article: 17)

The system's answer is comprehensive and legally accurate, including a closing paragraph on the exemptions to the right to erasure — correctly naming freedom of expression, legal obligations, public health, archiving, and **"the establishment, exercise, or defence of legal claims"** as grounds under which erasure does not apply. This mirrors the actual content of GDPR Article 17(3). The LLM-judge flagged exactly this claim as unsupported by the retrieved context.

Inspecting the top-5 retrieved chunks confirms why: only the first-ranked chunk is actually from Article 17. The remaining four (ranks 2–5) are from Articles 19, 66, 18, and 62 — unrelated provisions — retrieved with similarity scores only marginally lower than the correct chunk (0.779 down to 0.686, a narrow band). The specific sub-clause about the legal-claims exemption was never in the retrieved context at all.

**What this means:** the model did not fabricate false information — it supplied a true statement from its own parametric knowledge of GDPR when the retrieved evidence ran out. For a system whose entire design premise is "answer only from context, cite only what's retrieved," this is a meaningful failure even though the output text is factually correct, because the citation implies a grounding that isn't there. In a legal-assistant context, an ungrounded-but-correct answer today is one chunking or model change away from being an ungrounded-and-wrong answer tomorrow, with no way for a user to tell the difference from the citation alone.

**Recommended next step (not yet applied):** Article 17 currently appears to retrieve as a small number of large chunks, such that its exemption sub-clause (17(3)) doesn't surface as its own retrievable unit for a general "right to be forgotten" query. A finer sub-article chunking pass for exemption/carve-out clauses specifically, or a stricter grounding-prompt instruction to omit claims not textually present in retrieved context (even if true), would both plausibly address this. This should be proposed, confirmed, and tested before/after on the full golden set before being applied to the production prompt, consistent with this project's change-control process.

### 3.2 Finding 2 — The Relevance Metric Cannot Distinguish "Correctly Abstained" from "Failed to Answer"

**Tier:** Not a RAG pipeline failure — an evaluation-methodology gap.

**Questions (Q24–Q27):** all four intentionally unanswerable golden questions (e.g., *"What is the minimum wage for data protection officers under GDPR?"*).

All four correctly abstain (`abstention_correct: True` in every case — the system behaves exactly as intended). Yet all four score `relevance: 0.0`, with judge reasoning uniformly along the lines of *"the answer does not provide substantive information... instead stating an inability to answer."*

This is not evidence the system is doing anything wrong. It's evidence the relevance-scoring prompt used in the eval harness has no path to reward a correct abstention — it appears to score "did this substantively address the question," which a correct refusal will always fail by construction. The practical risk is that this metric is currently uninformative for the 4 unanswerable questions (it's stuck at a floor regardless of answer quality) and would not detect a regression in abstention *quality* (e.g., an unhelpfully terse refusal vs. one that explains why) — only `abstention_correct` currently does, and that's a binary, not a quality signal.

**Status: Applied and verified.** The original relevance-judge prompt could not be located anywhere on disk (it was never saved as a standalone file — see §2.1's note on the same pattern affecting the Phase 3 prompt). Rather than guess at a reconstruction, a new relevance-judge prompt was written from scratch, branching explicitly on `is_unanswerable`: answerable questions are scored on topical coverage as before, while unanswerable questions are scored on abstention quality (does the refusal avoid implying an answer exists, without hallucinating or speculating).

All 27 questions were re-scored with the new judge, reusing the existing stored answers (no new retrieval or generation calls — zero additional pipeline cost). Results:

| | Old avg relevance | New avg relevance |
|---|---|---|
| Unanswerable questions (n=4) | 0.000 | **1.000** |
| Answerable questions (n=23) | 1.000 | 0.983 |

All four unanswerable questions moved from 0.0 to 1.0 relevance, each with reasoning that now correctly credits the correct, non-hallucinating refusal. Only 2 of 23 answerable-question scores changed at all (Q20: 1.0→0.8, Q22: 1.0→0.8), and both changes reflect the new judge genuinely being more thorough — it caught specific omitted articles in otherwise-strong answers (e.g., Q20's answer covered several data-subject rights but omitted rectification, erasure, restriction, and portability) rather than any instruction-following flaw introduced by the abstention-branch logic. This is evidence the fix is targeted — it corrected the abstention blind spot without materially disturbing scoring behavior on answerable questions.

This change lives in the eval harness only; it does not touch the production API, prompt, or retrieval pipeline validated in Phases 4–8.

### 3.3 Finding 3 — Adjacent-Article Ranking Confusion (Confirms the Project's Pre-Build Hypothesis)

**Tier:** Retrieval. **Named modes:** Position-of-Gold Bias [2]; Multi-Hop/cross-reference confusion consistent with Chunking Boundary Errors [3].

**Questions (Q4, Q5, Q6, Q12, Q16, Q17)** — 6 of 23 answerable questions (26%). In every case, the gold article is retrieved within the top 5 (so `recall@5 = 1.0` for all of them — this cluster is entirely invisible to the headline metric) but is outranked by one or more articles that are **substantively adjacent** to it:

| Q | Question topic | Gold article | Outranked by (rank 1) | Relationship |
|---|---|---|---|---|
| 4 | Special categories of data | Art. 9 | Art. 13 | Both concern what must be disclosed/protected about data subjects |
| 5 | Info when data collected directly | Art. 13 | Art. 14 | Direct vs. indirect collection — paired, consecutive articles |
| 6 | Info when data NOT collected directly | Art. 14 | Art. 11 | Same transparency-obligations cluster |
| 12 | Right to object | Art. 21 | Art. 6 | Both concern lawful grounds / objection to processing |
| 16 | Security measures | Art. 32 | Art. 24, Art. 58 | Controller-obligations and supervisory-authority-powers cluster |
| 17 | DPO designation | Art. 37 | Art. 38 | Consecutive articles — *designation* vs. *position/tasks* of the DPO |

The pattern is consistent, not random: in every single case the top-ranked wrong chunk is not an arbitrary mismatch but a **legally and often numerically neighboring provision** — exactly the structural risk the project flagged before any Phase 9 code was written, given how densely GDPR's articles cross-reference each other. Articles 13/14 and 37/38 in particular are paired by design (direct vs. indirect collection; designation vs. duties), which likely makes their chunk embeddings genuinely close in vector space — this may be a harder problem than generic retrieval noise, because the confusion is coming from real semantic proximity in the source material, not an embedding-model weakness alone.

**Status: Rank-1 precision re-tested and confirmed — reranker helps here. Not yet deployed.** The project previously dropped the reranker because it made no difference to recall@5. That comparison was never run at the rank-1 level specifically, so it was re-run directly against the already-computed `full_results.json` data (`structure` strategy, with vs. without reranker — no new API calls, pure data analysis).

**Result: all 6 of 6 adjacent-article questions moved to gold_rank = 1 with the reranker enabled** (previously ranks 2–3), with 0 made worse within this target set:

| Q | Gold article | Rank without reranker | Rank with reranker |
|---|---|---|---|
| 4 | Art. 9 | 2 | **1** |
| 5 | Art. 13 | 2 | **1** |
| 6 | Art. 14 | 2 | **1** |
| 12 | Art. 21 | 2 | **1** |
| 16 | Art. 32 | 3 | **1** |
| 17 | Art. 37 | 2 | **1** |

This directly confirms the hypothesis: the reranker was correctly judged to make no difference to *recall*@5 in Phase 4, but it was never evaluated for *rank-1 precision* — where it turns out to fully resolve this specific failure cluster.

**Honest caveat — checked for regressions elsewhere:** reranking the other 21 questions showed 0 improved, 15 unchanged, and **2 made worse** (Q2: Art.6 moved from rank 1 to rank 2; Q22: gold moved from rank 1 to rank 4). Both regressions were checked for downstream impact: in both cases recall@5 stayed at 1.0, citations stayed clean, and faithfulness stayed at 1.0 — the rank shift did not propagate into a worse generated answer. This is a real trade-off, not a free win, but it is a favorable one: 6 clear fixes against 2 benign rank shifts with no measured answer-quality cost.

**Remaining step before deployment:** this analysis validates the reranker's *data-level* effect only. Actually reintroducing it into the production API (`src/retrieval.py`, currently `use_rerank=False` in the deployed config) is a pipeline change, not an analysis — it should be explicitly proposed, confirmed, and tested end-to-end (including latency impact, since Phase 7 established cold-start latency is already a known constraint) before being deployed, consistent with this project's change-control process.

---

## 4. What the Three Findings Together Show

Each finding has a different root cause and would need a different fix — which is itself the headline result of this phase. A single "faithfulness went down" or "retrieval missed something" framing would have missed two of the three:

- Finding 1 is a **generation-grounding** problem — correct output, broken evidence chain.
- Finding 2 is a **measurement** problem — the system wasn't wrong, the scorer couldn't tell.
- Finding 3 is a **retrieval-ranking** problem — the right answer is present but not prioritized, driven by genuine semantic closeness between GDPR articles rather than a generic embedding failure.

All three were invisible to the Phase 4 aggregate metrics, consistent with the literature's finding that aggregate RAG metrics systematically under-detect real production issues [1].

---

## 5. Limitations of This Analysis

- Sample size is 27 questions (23 answerable); the 6-question adjacent-article pattern, while consistent, is not large enough to rule out coincidence with statistical confidence.
- All faithfulness/relevance judgments come from a single LLM-judge pass with no human cross-validation in this phase.
- Only the deployed configuration was analyzed in depth; the other 5 configurations in `full_results.json` were not re-examined at this granularity and might show different or additional patterns.
- This is a single run, not a variance estimate — re-running the same 27 questions could show some score fluctuation independent of any real system change.

---

## 6. Recommended Next Steps, Prioritized

1. ~~**Patch the relevance-judge prompt** to branch on abstention (Finding 2)~~ — **Done.** See §3.2.
2. ~~**Re-test the existing reranker's rank-1 precision** on the 6 adjacent-article questions (Finding 3)~~ — **Done, data-validated.** See §3.3. **Deployment decision still pending:** reintroducing the reranker into the live API is a pipeline change requiring explicit sign-off and end-to-end (including latency) testing before deployment.
3. **Propose and test a grounding-prompt tightening or finer Article 17 chunking** (Finding 1) — higher-risk since it touches the production prompt/chunking validated in Phase 4; requires before/after testing on the full golden set and explicit confirmation before deployment, per this project's change-control process.

---

## References

[1] Three-tier RAG failure taxonomy (Retrieval / Augmentation / Generation) synthesized from 150+ papers spanning academic and industrial RAG deployments, 2026.
[2] 12-failure-mode production RAG taxonomy, including Position-of-Gold Bias and Citation Drift as named retrieval/citation-level failure modes.
[3] 33-failure-mode taxonomy across 7 RAG pipeline stages (ingestion, representation, retrieval, generation, evaluation, deployment, agentic orchestration), including Chunking Boundary Errors and Multi-Hop Reasoning Gaps as named retrieval-stage failure modes.
