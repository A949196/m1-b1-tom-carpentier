"""Run hyperparameter search and generate experiments.md automatically.

Usage:
    python src/run_experiments.py
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import joblib

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from preprocess import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TARGET_MAPPING,
    build_preprocessor,
    load_dataset,
)

# Espace de recherche
PARAM_GRID = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 10, 15, 20],
    "classifier__min_samples_leaf": [1, 5, 10],
    "classifier__class_weight": [None, "balanced"],
}

N_ITER = 8
CV_FOLDS = 5
RANDOM_STATE = 42


def run_search(data_path: Path = Path("data/lending_club_train.csv"),) -> dict:
    """Run RandomizedSearchCV and return results."""
    X, y = load_dataset(data_path)

    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    scorer = {
        "f1_macro": make_scorer(f1_score, average="macro"),
        "roc_auc": "roc_auc",
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_GRID,
        n_iter=N_ITER,
        scoring=scorer,
        refit="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
    )

    start = time.time()
    search.fit(X, y)
    elapsed = time.time() - start

    # Affichage de TOUS les résultats
    print(f"\n{'='*70}")
    print(f"RÉSULTATS — {N_ITER} combinaisons, {CV_FOLDS}-fold CV ({elapsed:.1f}s)")
    print(f"{'='*70}")
    print(
        f"{'#':<4} {'n_est':<6} {'depth':<6} {'min_leaf':<9} "
        f"{'class_weight':<13} {'F1 macro':<18} {'ROC-AUC':<18} {'Rang'}"
    )
    print(f"{'-'*70}")

    for i in range(len(search.cv_results_["params"])):
        params = search.cv_results_["params"][i]
        f1 = search.cv_results_["mean_test_f1_macro"][i]
        f1_std = search.cv_results_["std_test_f1_macro"][i]
        auc = search.cv_results_["mean_test_roc_auc"][i]
        auc_std = search.cv_results_["std_test_roc_auc"][i]
        rank = search.cv_results_["rank_test_f1_macro"][i]

        marker = " ★ BEST" if rank == 1 else ""
        print(
            f"{i+1:<4} "
            f"{params.get('classifier__n_estimators', '-'):<6} "
            f"{str(params.get('classifier__max_depth', 'None')):<6} "
            f"{params.get('classifier__min_samples_leaf', '-'):<9} "
            f"{str(params.get('classifier__class_weight', 'None')):<13} "
            f"{f1:.4f} ± {f1_std:.4f}   "
            f"{auc:.4f} ± {auc_std:.4f}   "
            f"{rank}{marker}"
        )

    print(f"{'-'*70}")
    print(f"\nMeilleur F1 macro (CV) : {search.best_score_:.4f}")
    print(f"Hyperparams retenus    : {search.best_params_}")

    return {
        "search": search,
        "elapsed": elapsed,
        "data_path": data_path,
    }


def save_best_model(
    search: RandomizedSearchCV, data_path: Path, output_dir: Path = Path("../models/")
) -> None:
    """Persist the best pipeline + metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "pyrenex_risk_v2.joblib"
    joblib.dump(search.best_estimator_, model_path, compress=3)

    meta = {
        "model_name": "pyrenex_risk_v2",
        "model_version": "v2.0.0",
        "config_name": "randomized_search_best",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "dataset_sha256": sha256(data_path.read_bytes()).hexdigest(),
        "hyperparameters": search.best_params_,
        "search_method": f"RandomizedSearchCV(n_iter={N_ITER}, cv={CV_FOLDS})",
        "metrics_test_internal": {
            "f1_macro_cv_mean": round(search.best_score_, 4),
        },
        "feature_columns": {
            "numeric": list(NUMERIC_FEATURES),
            "categorical": list(CATEGORICAL_FEATURES),
        },
        "target": {"column": TARGET_COLUMN, "mapping": TARGET_MAPPING},
    }

    meta_path = output_dir / "pyrenex_risk_v2.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Modèle retenu sauvegardé → {model_path}")
    print(f"Métadonnées → {meta_path}")
    print(f"\nProchaine étape :")
    print(f"  python src/evaluate.py --update-meta")


def main() -> None:
    data_path = Path("../data/lending_club_train.csv")
    result = run_search(data_path)

    save_best_model(result["search"], data_path)


if __name__ == "__main__":
    main()
