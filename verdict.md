# Verdict — Modèle de scoring Pyrenex Crédit v2

> Document destiné à Sophie Léger (Lead Data, Pyrenex Crédit).
> 1 page max.

## Contexte

Pyrenex Crédit souhaitait réentraîner son modèle de scoring du risque de défaut (RandomForest 2017, `pyrenex_risk_v1`) sur un dataset enrichi de données récentes, afin de corriger la baisse de performance prédictive observée sur les profils post-Covid et travailleurs indépendants.

## Démarche

Le nouveau dataset Lending Club (~30 k lignes, 18 % de défauts) a été splitté en train/test interne (80/20, stratifié, `random_state=42`) et un holdout totalement isolé (6 000 lignes). Huit configurations d'hyperparamètres ont été évaluées via `RandomizedSearchCV` (5-fold CV, scorer F1 macro) ; la configuration retenue est celle avec le meilleur F1 macro CV. Le holdout n'a été consulté qu'une seule fois, pour produire le verdict final.

## Verdict chiffré

| Métrique | Baseline 2017 (Pyrenex-risk-v1) | Modèle retenu (v2) | Variation |
|---|---|---|---|
| F1 macro (holdout) | 0.5018 | 0.6179 | +0.116 |
| F1 défaut | n/d | 0.4208 | — |
| ROC-AUC | 0.7296 | 0.7368 | +0.007 |
| Recall défaut | 0.05 | 0.554 | **+0.504** |

**Configuration retenue** : `n_estimators=300`, `max_depth=15`, `min_samples_leaf=10`, `class_weight='balanced'`

## Trade-off explicité au métier

Le recall défaut passe de 5 % à 55 % — soit 11 fois plus de mauvais payeurs détectés — au prix d'une précision défaut qui chute de 0.61 à 0.34 et d'une accuracy qui passe de 0.85 à 0.72. 
Pour trouver plus de défauts, le modèle déclenche davantage de fausses alertes. 
Ce trade-off est un choix métier assumé : le coût d'un défaut non détecté est structurellement supérieur au coût d'un refus injustifié.

## Précautions avant mise en production

- Vérifier que le **schéma d'entrée** en production correspond exactement au schéma d'entraînement (cf. `pyrenex_risk_v2.json` → `feature_columns`)
- Re-évaluer le **seuil de décision** (0.5 par défaut) avec l'équipe métier — un seuil abaissé à ~0.3 peut être plus adapté
- Mettre en place un **monitoring** dès le déploiement : dérive de distribution détectée (proportion de défauts +3 pts vs 2017, grades plus risqués)
- Surveiller les **variables sensibles** identifiées (`fico_range_low`, `annual_inc`, `emp_length`)

## Recommandation

✅ **Remplacer Pyrenex-risk-v1** par v2 — un modèle qui détecte 5 % des défauts n'est pas opérationnel pour un organisme de crédit ; le v2 multiplie ce recall par 11 avec une traçabilité complète et un pipeline reproductible.

---

*Signé : Tom Carpentier, FastIA, le 2025-06-03*
