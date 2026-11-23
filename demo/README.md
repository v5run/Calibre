# Calibre demo

A chat-style demo that shows Calibre's effect on real test questions. Pick a recall target (80%, 90% or 95%), ask a question (or pick a suggested one), and the reply shows:

- the answer, and whether it was in the passages Calibre sent;
- how much context Calibre sent compared with a fixed top-5 (passages and tokens);
- the passages Calibre sent, each with Calibre's estimate that it's relevant and the dataset's own label, plus the extra passages only top-5 would have sent;
- a flag when no passage looks relevant, where an agent could search again or abstain.

Switching the target re-cuts every reply, so you can watch Calibre send more context for a stricter target.

Knowledge bases: Wikipedia (SQuAD v1.1), science papers (BEIR SciFact) and nutrition research (BEIR NFCorpus).

It's a static site: HTML, CSS and JavaScript, no build step and no server.

## Run it locally

Open `demo/index.html` in a browser, or serve the folder:

```bash
cd demo
python -m http.server 8000
# then open http://localhost:8000
```

## Deploy on Vercel

1. In Vercel, choose **Add New → Project** and import the GitHub repo.
2. Set **Root Directory** to `demo`.
3. Framework preset: **Other**. Leave the build command and output directory empty.
4. Deploy. Every push to `main` redeploys.

## How it works

- Retrieval is precomputed. For each test question, `data/demo_data.js` holds the ranked passages from `calibre_rag` (BM25 + BGE-small-en-v1.5, tuned convex blend) with each passage's P(relevant) (v1), how many passages Calibre sends for each recall target (v2) and top-5 sends, token counts, relevance labels and, for SQuAD, the answer key.
- Half of each dataset's test questions are held out for the demo. The rest, pooled with the dataset's train questions, were used to tune the blend, train the calibrator (60%) and fit the recall guarantees (40%).
- The recall target is a guarantee on average across questions, so some individual questions still miss (try "When did Mark Woods leave ABC?" at 90% and then 95%).
- Typed questions are matched to the closest question in the test set. Questions outside it get a list of the closest ones available.
- Answers come from the dataset's answer key. No LLM is called.
- `scripts/export_from_pilot.py` generated the data file from the pilot study's retrieval runs. It's kept for provenance and needs those pilot artifacts, which aren't in this repo.

## Data sources

- SQuAD v1.1 (Rajpurkar et al., 2016), CC BY-SA 4.0.
- SciFact and NFCorpus via the BEIR benchmark (Thakur et al., 2021).
