# M1-B1 — Pyrenex Crédit scoring (Tom Carpentier)

Réentraînement du modèle de scoring risque de défaut pour Pyrenex Crédit.
Baseline : `pyrenex-risk-v1` (RandomForest 2017). Modèle livré : `pyrenex_risk_v2.joblib`.

→ Verdict complet : [verdict.md](./verdict.md)

---

## Reproduire l'entraînement (3 commandes)

```bash
pip install -r requirements.txt
python src/train.py --config balanced
python src/evaluate.py --update-meta
```

---

## Métriques retenues (holdout — 6 000 lignes)

| Métrique | Baseline v1 (2017) | **v2 (retenu)** |
|---|---|---|
| F1 macro | 0.5018 | **0.6179** |
| ROC-AUC | 0.7296 | **0.7368** |
| Recall défaut | 0.05 | **0.554** |
| F1 défaut | n/d | 0.4208 |

**Configuration retenue** : `n_estimators=300`, `max_depth=15`, `min_samples_leaf=10`, `class_weight='balanced'`

---

## Structure du repo

```
M1-B1-scoring-tom/
├── data/
│   ├── lending_club_train.csv
│   └── lending_club_holdout.csv        # holdout — ne pas modifier
├── notebooks/
│   └── M1-B1_tom_scoring.ipynb
├── src/
│   ├── preprocess.py                   # transformations reproductibles
│   ├── train.py                        # script d'entraînement
│   ├── evaluate.py                     # métriques sur holdout
│   └── run_experiments.py              # RandomizedSearchCV → experiments.md
├── models/
│   ├── pyrenex_risk_v2.joblib          # Pipeline complet (compress=3)
│   └── pyrenex_risk_v2.json            # métadonnées (5 clés obligatoires)
├── contract_test.py                    # validation shapes/classes/probas/stabilité
├── experiments.md                      # 8 runs tracés, aucun score holdout
├── verdict.md                          # comparaison chiffrée baseline vs v2
├── requirements.txt
└── README.md
```

---

## Charger le modèle

```python
import joblib
pipeline = joblib.load("models/pyrenex_risk_v2.joblib")
y_pred  = pipeline.predict(X)
y_proba = pipeline.predict_proba(X)[:, 1]
```

---

## Valider le contrat

```bash
python contract_test.py
# → Contract test OK — shapes valides, probas dans [0,1], stabilité confirmée.
```

---

## Ressources

| Tâche | Mini-cours |
|---|---|
| EDA + split stratifié | `ressources/01_Pandas_Sklearn_split_essentiel.md` |
| Métriques classification déséquilibrée | `ressources/02_Metrics_classif_desequilibree_essentiel.md` |
| Hyperparamètres RandomForest | `ressources/03_RandomForest_hyperparams_essentiel.md` |
| Traçage des runs | `ressources/04_Tracage_experiments_md_essentiel.md` |
| Persistance modèle | `ressources/05_Persistance_modele_joblib_essentiel.md` |
