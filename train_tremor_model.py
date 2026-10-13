"""Train the tremor ensemble. Run once:  python train_tremor_model.py

Honest validation: the CSV is ONE continuous recording, so neighbouring windows are
near-duplicates. We split it into 5 contiguous blocks; the model used to test a
block is trained only on the OTHER blocks (with a purge gap). That is the model
the tremor-test menu uses for the dataset you pick, so results are never in-sample."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from tremor_features import SENSOR_COLUMNS, WINDOW, STRIDE, SEGMENT_LEN, window_features

BASE = Path(__file__).resolve().parent
CSV = BASE / "tremor_data" / "Dataset.csv"
OUT = BASE / "tremor_model"
OUT.mkdir(exist_ok=True)
K = 5


def make_models():
    return {
        "extra_trees": ExtraTreesClassifier(400, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=0),
        "random_forest": RandomForestClassifier(400, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=0),
        "hist_gb": HistGradientBoostingClassifier(learning_rate=0.05, max_depth=4, max_iter=250,
                                                  l2_regularization=1.0, class_weight="balanced", random_state=0),
    }


def ens_proba(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models.values()], axis=0)


def main():
    df = pd.read_csv(CSV)
    X = df[SENSOR_COLUMNS].values.astype(float)
    y = df["Result"].values
    n_seg = len(X) // SEGMENT_LEN
    print(f"{len(X)} samples -> {n_seg} test datasets of {SEGMENT_LEN} samples")

    starts = np.arange(0, n_seg * SEGMENT_LEN - WINDOW + 1, STRIDE)
    F = np.array([window_features(X[s:s + WINDOW]) for s in starts])
    Y = np.array([int(y[s:s + WINDOW].mean() >= 0.5) for s in starts])

    seg_edges = np.linspace(0, n_seg, K + 1).astype(int)          # fold boundaries in segment units
    sample_edges = seg_edges * SEGMENT_LEN
    folds, oof = [], np.full(len(Y), np.nan)
    for k in range(K):
        lo, hi = sample_edges[k], sample_edges[k + 1]
        te = (starts >= lo) & (starts + WINDOW <= hi)
        tr = (starts + WINDOW <= lo - WINDOW) | (starts >= hi + WINDOW)
        models = make_models()
        for m in models.values():
            m.fit(F[tr], Y[tr])
        oof[te] = ens_proba(models, F[te])
        folds.append(models)
        print(f"fold {k + 1}/{K}: train windows={tr.sum()}, test windows={te.sum()}")

    ev = ~np.isnan(oof)
    pred = (oof[ev] >= 0.5).astype(int)
    metrics = {"window_accuracy": float(accuracy_score(Y[ev], pred)),
               "window_balanced_accuracy": float(balanced_accuracy_score(Y[ev], pred)),
               "window_f1": float(f1_score(Y[ev], pred)),
               "window_roc_auc": float(roc_auc_score(Y[ev], oof[ev]))}

    # segment-level accuracy (what the menu reports): mean prob over windows vs majority label
    seg_true, seg_pred = [], []
    for i in range(n_seg):
        lo, hi = i * SEGMENT_LEN, (i + 1) * SEGMENT_LEN
        idx = np.flatnonzero((starts >= lo) & (starts + WINDOW <= hi) & ev)
        if len(idx):
            seg_pred.append(int(oof[idx].mean() >= 0.5))
            seg_true.append(int(y[lo:hi].mean() >= 0.5))
    metrics["segment_accuracy"] = float(accuracy_score(seg_true, seg_pred))
    metrics["segment_balanced_accuracy"] = float(balanced_accuracy_score(seg_true, seg_pred))
    print("\nBlocked cross-validation (never tested on neighbouring training data):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    joblib.dump({"fold_models": folds, "seg_edges": seg_edges.tolist(), "n_segments": n_seg,
                 "segment_len": SEGMENT_LEN, "window": WINDOW, "stride": STRIDE,
                 "sensor_columns": SENSOR_COLUMNS, "threshold": 0.5, "metrics": metrics},
                OUT / "tremor_ensemble.pkl", compress=3)
    (OUT / "tremor_model_config.json").write_text(json.dumps(
        {"segment_len": SEGMENT_LEN, "window": WINDOW, "stride": STRIDE, "n_segments": n_seg,
         "sensor_columns": SENSOR_COLUMNS, "threshold": 0.5, "metrics": metrics}, indent=2))
    print("\nSaved tremor_model/tremor_ensemble.pkl")


if __name__ == "__main__":
    main()
