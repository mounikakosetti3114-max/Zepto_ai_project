# Module 2 — Analytics Pipeline (`/analytics`)



## Structure
## setup

```
analytics/
├── 01_eda.py          # Part A — profiling, cleaning, EDA, data story (Tasks 1-6)
├── 02_modeling.py      # Part B — modeling pipeline (Tasks 7-15)
├── titanic.csv          # committed offline fallback (produced by 01_eda.py, Task 1)
├── full_pipeline.joblib # saved fitted pipeline (produced by 02_modeling.py, Task 15)
├── charts/               # all saved chart images
└── README.md             # this file
```

**How to run:** `python 01_eda.py` first (needs internet the first time, to
call `sns.load_dataset('titanic')` — it caches locally after that and also
writes `titanic.csv`), then `python 02_modeling.py` (reads only
`titanic.csv`, no network needed). Install requirements first:

```
pip install pandas numpy matplotlib seaborn scikit-learn joblib imbalanced-learn
```

Both scripts use `# %%` cell markers so they also open directly as Jupyter
notebooks in VS Code / Spyder, or can be pasted into `01_eda.ipynb` /
`02_modeling.ipynb` cell-by-cell if you'd rather submit notebooks.

---

## Part A — Profiling, cleaning, and the data story

**Task 1.** Loaded once via `sns.load_dataset('titanic')`, profiled with
`df.info()`, `df.describe()`, `df.shape`. Missing-value percentages are
printed for every column that has any (age, embarked, embark_town, deck).
Immediately saved as `titanic.csv` — this is the single committed offline
fallback; nothing after this point re-loads from the network.

**Task 2.** Threshold rule applied per column:
- `embarked` / `embark_town` — **0.22% missing** (< 5%) → rows dropped.
- `age` — **19.87% missing** (5–30%) → imputed with the column median
  (robust to the right-skew confirmed in Task 3).
- `deck` — **77.22% missing** (> 30%) → column dropped entirely. At this
  missing rate, any imputation strategy (mode, "missing" category) would
  be manufacturing a value for the large majority of rows rather than
  recovering real signal, and `deck` is largely redundant with
  `pclass`/`fare` as a proxy for cabin location — so dropping it is more
  defensible than encoding a mostly-fabricated category.

> **Actual result (from a real run):** age: 19.87% missing, embarked:
> 0.22% missing, deck: 77.22% missing, embark_town: 0.22% missing.

**Task 3.** Histograms + box plots for `age` and `fare`; IQR-rule outlier
counts reported for both. `fare` mean > median > mode → right-skewed
(a small number of very high fares pull the mean above the median/mode).

> **Actual result:** age had **65 IQR outliers** (normal range 2.5 to
> 54.5), fare had **114 IQR outliers** (normal range −26.76 to 65.66).
> fare mean = **32.10**, median = **14.45**, mode = **8.05** →
> mean > median > mode confirms fare is **right-skewed** (a small
> number of very expensive tickets pull the average up).

**Task 4.** Survival rate computed via boolean masking for (a) `sex`,
(b) `pclass`, (c) `sex` + `pclass` combined. Correlation matrix restricted
to exactly `survived, pclass, age, sibsp, parch, fare` (excluding the
derived flags `adult_male`, `alone`), rendered as a 6×6 heatmap. The two
strongest off-diagonal correlations (by `abs(correlation)`) are printed
and interpreted in the script output.

> **Actual result:** the two strongest correlations were
> **fare ↔ pclass = −0.548** (higher-class/lower-numbered tickets cost
> more) and **parch ↔ sibsp = 0.415** (passengers traveling with
> parents/children also tended to travel with siblings/a spouse).

**Task 5.** Four charts building one coherent survival story: (1) bar —
survival rate by class × sex, (2) box — fare distribution by class ×
survival, (3) scatter — age vs fare colored by survival, (4) heatmap —
survival rate by class × embarkation port. Each has a 2–4 sentence
interpretation printed alongside it (see script output / paste into this
README after running).

**Task 6.** `age` and `fare` standardized with the z-score formula on the
full cleaned frame, with a printed before/after mean/std comparison
confirming ~0 mean / ~1 std. This is explicitly an EDA-stage sanity check
only — it does not feed `02_modeling.py`, which performs its own
train-only `StandardScaler` fit.

---

## Part B — Predictive modeling

**Task 7.** Stratified 80/20 split on `survived` (`stratify=y`), justified
by the class imbalance measured in Task 1/11 (~59% not-survived / ~41%
survived) — a plain random split risks skewing that ratio between train
and test by chance.

> **Actual result:** Train shape (711, 12), Test shape (178, 12).
> Train class balance {0: 0.617, 1: 0.383}, Test class balance
> {0: 0.618, 1: 0.382} — nearly identical, confirming the stratified
> split preserved the original ~62%/38% ratio in both sets.

**Task 8.** `ColumnTransformer` + `Pipeline`: numeric features
(`pclass, age, sibsp, parch, fare`) get median imputation + `StandardScaler`;
categorical features (`sex, embarked`) get most-frequent imputation +
one-hot encoding. Every step is fit only via `pipe.fit(X_train, y_train)`;
`X_test` only ever passes through `.transform()` inside the already-fitted
pipeline — enforced structurally by scikit-learn's `Pipeline`, not by
convention.

**Task 9.** Logistic Regression, Decision Tree, and Random Forest trained
on the identical split. Decision Tree rendered with `plot_tree`, labeled
feature and class names (`charts/task9_decision_tree.png`).

**Task 10.** Confusion matrix, accuracy, precision, recall, F1, and
ROC-AUC for all three, side by side in one comparison table (see script
output / `charts/task10_confusion_matrices.png`,
`charts/task10_roc_curves.png`).

**Task 11.** Random Forest re-trained three ways: (a) baseline, (b)
`class_weight='balanced'`, (c) SMOTE applied only inside the training fold
via an `imblearn.pipeline.Pipeline` (so oversampling never touches the
test set). Precision/recall/F1 compared, with a written conclusion in the
script output.

> **Actual result:**
>
> | Strategy | Precision | Recall | F1 |
> |---|---|---|---|
> | baseline | 0.781 | 0.735 | 0.758 |
> | class_weight='balanced' | 0.739 | 0.750 | 0.745 |
> | SMOTE | 0.746 | 0.691 | 0.718 |
>
> On this run, the plain baseline actually had the best F1 (0.758); the
> class-weighted version traded some precision for a small recall gain,
> and SMOTE underperformed the baseline on both metrics here — a
> reminder that imbalance-handling techniques don't always help on
> every dataset/split and should be checked empirically, not assumed.

**Task 12.** `GridSearchCV` over `n_estimators`, `max_depth`,
`max_features` for the Random Forest (`cv=5`, scored on F1). Because
`oob_score` is only populated when `oob_score=True` is passed at
construction time, the winning hyperparameters are used to rebuild
`RandomForestClassifier(oob_score=True, **best_params)`, refit, and the
resulting `.oob_score_` is reported alongside `best_params_`.

**Task 13.** Linear Regression predicting `fare` from
`pclass, age, sibsp, parch, sex, embarked`. Reports MAE, RMSE, R²,
Adjusted R², plus a residual plot and an explicit heteroscedasticity
conclusion based on comparing residual spread across the low/high half of
predicted values.

> **Actual result:** MAE = **21.14**, RMSE = **41.75**, R² = **0.347**,
> Adjusted R² = **0.324**. The model explains about a third of the
> variation in fare from these features alone — expected, since cabin
> location/specific ticket deals aren't captured by `deck` (dropped in
> Task 2) or any finer-grained booking details.

**Task 14.** Final table presents classification metrics (accuracy,
precision, recall, F1, AUC) for the three classifiers, and regression
metrics (MAE, RMSE, R², Adjusted R²) for the fare model, as two clearly
separate metric groups (not one shared scale). Final written
recommendation names the classifier with the best F1/AUC.

> **Actual result:** Final recommendation — **deploy the Random Forest
> classifier** for the survival-prediction task; it gave the best
> balance of precision and recall (highest F1/ROC-AUC) among the three
> classifiers evaluated in Task 10. The fare regression model (Task 13)
> is kept only as a separate, non-comparable exploratory result on its
> own currency-unit scale.

**Task 15.** The complete fitted pipeline (preprocessing + tuned Random
Forest, as one object) is saved with `joblib.dump(full_pipeline,
"full_pipeline.joblib")`. The script reloads it with `joblib.load(...)`
and confirms it predicts correctly on one raw, unprocessed test row with
no manual preprocessing required.

> **Actual result:** Reloaded pipeline prediction on one raw test row =
> **[0]**, actual label for that row = **0** — match confirmed. The
> saved `full_pipeline.joblib` reproduces correct predictions end-to-end
> on raw input with no manual preprocessing step.

---

## Actual run — verified

Both scripts have been run successfully end-to-end on the real Titanic
dataset (via `sns.load_dataset('titanic')`), with no errors, and the
real numbers from that run are filled in above under each task
("Actual result"). `titanic.csv`, `full_pipeline.joblib`, and all chart
PNGs under `charts/` were produced and confirmed present. Tasks 5, 6,
9, 10, and 12 still print their full detail (chart interpretations,
Decision Tree image, confusion matrices, GridSearchCV best params +
OOB score) directly in the terminal — paste those numbers here too if
you'd like every single figure captured in this file rather than only
in the console output.