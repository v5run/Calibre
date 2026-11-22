# v2: Guaranteed

**Hypothesis.** Expected-recall and conformal cutoffs on calibrated probabilities hold their recall target in-domain, and send fewer passages than conformal filtering on raw dense scores.

**Plan**
- Policies: expected recall (ρ), split conformal (α), compared with conformal context filtering (Chakraborty et al. 2025).
- 100+ random calibration/test splits; realised coverage vs target; passages at equal guarantee.
- Also decide where calibration sits: before or after a cross-encoder reranker.

**Done when:** guarantee-validity chart (realised vs target coverage across splits).

**Results:** _not run yet._
