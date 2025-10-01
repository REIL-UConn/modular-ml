# preprocessing/feature_importance.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Sequence, Tuple, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.inspection import permutation_importance as _perm
import lightgbm as lgb

class FeatureImportance:
    def __init__(self, estimator: Union[str, object] = "rf", random_state: int = 42, n_jobs: int = -1, **est_kwargs):
        self.estimator_spec = estimator
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.est_kwargs = est_kwargs
        self.X = self.y = self.groups = None
        self.feature_names: Sequence[str] = ()

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Sequence[str], groups: Optional[np.ndarray] = None):
        self.X = np.asarray(X, float)
        self.y = np.asarray(y, float)
        self.feature_names = list(feature_names)
        self.groups = (np.asarray(groups) if groups is not None else None)
        return self

    def model_importance(self, plot: bool = False):
        est = self._make_estimator()
        est.fit(self.X, self.y)

        # RF / most trees:
        imp = getattr(est, "feature_importances_", None)
        if imp is None and est.__class__.__name__.lower().startswith("lgbm") and hasattr(est, "booster_"):
        # LightGBM (gain)
            imp = est.booster_.feature_importance(importance_type="gain")
        imp = np.asarray(imp if imp is not None else np.zeros(len(self.feature_names)), float)

        s = pd.Series(imp, index=self.feature_names, name="model_importance").sort_values(ascending=False)
        if plot:
            import matplotlib.pyplot as plt
            x = np.arange(len(s))
            plt.figure(figsize=(6,4)); plt.bar(x, s.values); plt.xticks(x, s.index, rotation=45, ha="right")
            plt.ylabel("Importance"); plt.title("Model-based Importance"); plt.tight_layout(); plt.show()
        return s

    def permutation_importance(self, val_size: float = 0.2, n_repeats: int = 20, plot: bool = False) -> Tuple[pd.DataFrame, float]:
        # train→val split inside training (group-aware if groups given)
        if self.groups is not None:
            (tr, va), = GroupShuffleSplit(1, test_size=val_size, random_state=self.random_state).split(self.X, groups=self.groups)
        else:
            tr, va = next(ShuffleSplit(1, test_size=val_size, random_state=self.random_state).split(self.X))
        Xtr, Ytr, Xva, Yva = self.X[tr], self.y[tr], self.X[va], self.y[va]

        est = self._make_estimator(); est.fit(Xtr, Ytr)
        val_r2 = float(est.score(Xva, Yva))

        perm = _perm(est, Xva, Yva, n_repeats=n_repeats, random_state=self.random_state, n_jobs=self.n_jobs)
        df = (pd.DataFrame({"feature": self.feature_names, "perm_mean": perm.importances_mean, "perm_std": perm.importances_std})
                .set_index("feature").sort_values("perm_mean", ascending=False))

        if plot:
            import matplotlib.pyplot as plt
            x = np.arange(len(df)); y = df["perm_mean"].values; e = df["perm_std"].values
            plt.figure(figsize=(6,4)); plt.bar(x, y); plt.errorbar(x, y, yerr=e, fmt="none", capsize=4, ecolor="tab:orange")
            plt.xticks(x, df.index, rotation=45, ha="right"); plt.ylabel("Importance (mean)")
            plt.title(f"Permutation Importance (val) — R²={val_r2:.3f}"); plt.tight_layout(); plt.show()
        return df, val_r2

    def _make_estimator(self):
        if not isinstance(self.estimator_spec, str):
            return self.estimator_spec
        name = self.estimator_spec.lower()
        if name in ("rf", "random_forest", "randomforest"):
            return RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs, **self.est_kwargs)
        if name in ("lgbm", "lightgbm"):
            return lgb.LGBMRegressor(random_state=self.random_state, n_jobs=self.n_jobs, **self.est_kwargs)
        raise ValueError("estimator must be 'rf', 'lgbm', or a ready estimator")
