# Expériences — M1-B1 Pyrenex Crédit (Lending Club)

> Méthode : **RandomizedSearchCV** (8 combinaisons, 5-fold stratifié, scorer F1 macro)
> Temps total : 441.5s | random_state : 42
>
>  **Comparabilité.** Le holdout **n'apparaît jamais** dans les
> résultats ci-dessous. Il sort **une seule fois**, pour le modèle retenu, dans
> la section finale en bas de fichier. Cf. mini-cours 04.

---

## Résultats par combinaison

| # | n_estimators | max_depth | min_samples_leaf | class_weight | F1 macro (CV) | ROC-AUC (CV) | Rang |
|---|---|---|---|---|---|---|---|
| 1 | 200 | None | 5 | None | 0.4974 ± 0.0048 | 0.7327 ± 0.0083 | 6 |
| 2 | 300 | 15 | 10 | balanced | 0.6179 ± 0.0077 | 0.7364 ± 0.0083 | 1 |
| 3 | 100 | 15 | 1 | None | 0.4982 ± 0.0036 | 0.7294 ± 0.0102 | 5 |
| 4 | 100 | None | 1 | None | 0.5113 ± 0.0061 | 0.7218 ± 0.0065 | 3 |
| 5 | 200 | 20 | 1 | None | 0.5052 ± 0.0081 | 0.7262 ± 0.0082 | 4 |
| 6 | 300 | 10 | 5 | balanced | 0.6081 ± 0.0079 | 0.7355 ± 0.0092 | 2 |
| 7 | 200 | 10 | 1 | None | 0.4817 ± 0.0064 | 0.7373 ± 0.0075 | 8 |
| 8 | 200 | 20 | 10 | None | 0.4878 ± 0.0034 | 0.7360 ± 0.0082 | 7 |

---

## Analyse

- **`class_weight='balanced'` est déterminant** : les deux seules configs avec F1 macro > 0.60 utilisent `balanced`. Sans ce paramètre, le modèle ignore largement la classe minoritaire (défaut), d'où un F1 macro atteignant au maximum ~0.50.
- **ROC-AUC varie peu** (0.72–0.74) entre les configs → le modèle discrimine de façon comparable, mais le seuil de décision par défaut pénalise la classe minoritaire sans `balanced`.
- **Rang 1 vs Rang 2** : la config retenue (n_estimators=300, max_depth=15, min_samples_leaf=10) surpasse la seconde (n_estimators=300, max_depth=10, min_samples_leaf=5) de ~1 point de F1 macro.

---

## Config retenue (rang 1)

```json
{
  "classifier__n_estimators": 300,
  "classifier__min_samples_leaf": 10,
  "classifier__max_depth": 15,
  "classifier__class_weight": "balanced"
}
```

- **F1 macro moyen (CV)** : 0.6179
- **ROC-AUC moyen (CV)** : 0.7364
- **Modèle persisté** : `models/pyrenex_risk_v2.joblib`

---

## Évaluation finale sur holdout (modèle retenu)

> **À remplir une seule fois**, à la tâche 5 du brief, **après** avoir choisi
> le modèle retenu. Le holdout n'est consulté qu'ici.

- **Date** :
- **Modèle persisté** : `models/pyrenex_risk_v2.joblib`
- **Données holdout** : `data/lending_club_holdout.csv`
- **Métriques** :
  - F1 macro :
  - F1 défaut :
  - ROC-AUC :
  - Recall défaut :
- **Matrice de confusion** :

|  | Pred Fully Paid | Pred Charged Off |
|---|---|---|
| **Vrai Fully Paid** | … | … |
| **Vrai Charged Off** | … | … |

- **Comparaison baseline 2017** : (cf. `verdict.md`)
