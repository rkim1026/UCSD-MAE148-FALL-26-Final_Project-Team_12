"""Logistic-regression go/wait classifier — veto only.

final_go = rule_go AND model_go
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from roundabout_pkg.ml.hindsight_labels import LabelledSample
from roundabout_pkg.ring_gap import GapParams, RingObservation, scale_free_features

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:  # pragma: no cover
    LogisticRegression = None


class GoWaitClassifier:
    def __init__(self, threshold: float = 0.5):
        if LogisticRegression is None:
            raise RuntimeError("scikit-learn is required for GoWaitClassifier")
        self.model = LogisticRegression(max_iter=500)
        self.threshold = threshold
        self._fitted = False

    def fit(self, samples: Sequence[LabelledSample]) -> "GoWaitClassifier":
        x = np.vstack([s.features for s in samples])
        y = np.array([s.label_safe for s in samples], dtype=np.int32)
        self.model.fit(x, y)
        self._fitted = True
        return self

    def predict_proba_go(self, obs: RingObservation, params: GapParams) -> float:
        if not self._fitted:
            return 1.0
        feats = scale_free_features(obs, params).reshape(1, -1)
        return float(self.model.predict_proba(feats)[0, 1])

    def allow_go(self, obs: RingObservation, params: GapParams) -> bool:
        """Veto interface: True means the model does not block going."""
        return self.predict_proba_go(obs, params) >= self.threshold

    def save(self, path: Path) -> None:
        path = Path(path)
        payload = {
            "coef": self.model.coef_.tolist(),
            "intercept": self.model.intercept_.tolist(),
            "threshold": self.threshold,
            "classes": self.model.classes_.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "GoWaitClassifier":
        if LogisticRegression is None:
            raise RuntimeError("scikit-learn is required for GoWaitClassifier")
        payload = json.loads(Path(path).read_text())
        obj = cls(threshold=float(payload["threshold"]))
        # Rebuild a fitted estimator with stored coefficients.
        obj.model.classes_ = np.array(payload["classes"])
        obj.model.coef_ = np.array(payload["coef"])
        obj.model.intercept_ = np.array(payload["intercept"])
        obj._fitted = True
        return obj


def train_from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
    out_path: Optional[Path] = None,
) -> GoWaitClassifier:
    samples = [
        LabelledSample(features=x[i], label_safe=int(y[i]), stamp=0.0)
        for i in range(len(y))
    ]
    clf = GoWaitClassifier(threshold=threshold).fit(samples)
    if out_path is not None:
        clf.save(Path(out_path))
    return clf
