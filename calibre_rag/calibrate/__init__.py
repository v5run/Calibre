"""v1 ("calibrated"): turn fused hybrid scores into P(relevant).

Planned (from feasibility check D): a logistic regression over 16 list
features (per retriever: min-max score, z-score, log-rank, missing flag; raw
cosine; per-list max / std / top-gap; query length), cross-fitted on training
queries, plus isotonic and Platt variants and a `recalibrate()` few-shot re-fit.
"""


class Calibrator:
    def fit(self, *args, **kwargs):
        raise NotImplementedError("Calibrator arrives in v1; see experiments/v1_calibrated")

    def predict_proba(self, *args, **kwargs):
        raise NotImplementedError("Calibrator arrives in v1; see experiments/v1_calibrated")


__all__ = ["Calibrator"]
