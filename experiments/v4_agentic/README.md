# v4: Agentic

**Hypothesis.** Calibre in a RAG pipeline and in a tool-using agent lowers tokens and end-to-end latency at equal answer accuracy vs hybrid top-k.

**Plan**
- RAG: NQ, HotpotQA, SQuAD, MultiHop-RAG with an 8B open model (4-bit, Kaggle T4): EM/F1 vs tokens.
- Agent: smolagents search tool backed by Calibre (passages + `nothing_relevant`) vs hybrid top-k, on NQ or BrowseComp-Plus; OpenTelemetry timing per step.
- Metrics: end-to-end latency p50/p95, search steps, tokens read, accuracy.

**Done when:** end-to-end latency and accuracy chart, Calibre vs hybrid top-k.

**Latency model reference:** with measured search timings, the cutoff alone was estimated at −2% (1 search step) to −11% (6 steps) end to end; saving one search step would be far larger.

**Results:** _not run yet._
