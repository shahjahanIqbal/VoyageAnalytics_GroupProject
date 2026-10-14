# Voyage Analytics — Notebook vs. Merged Script Comparison

## 1. Original Notebook Methodology (The Flawed Approach)

The original project (`flight price prediction.ipynb`) follows this workflow:

```python
# 1. Load data
df = pd.read_csv('flights.csv')

# 2. Drop identifiers
df.drop(["travelCode", "userCode", "date"], axis=1, inplace=True)

# 3. LabelEncode ALL categorical columns on the FULL dataset (LEAKAGE)
le_from = LabelEncoder()
le_to = LabelEncoder()
le_flightType = LabelEncoder()
le_agency = LabelEncoder()

df["from"] = le_from.fit_transform(df["from"])
df["to"] = le_to.fit_transform(df["to"])
df["flightType"] = le_flightType.fit_transform(df["flightType"])
df["agency"] = le_agency.fit_transform(df["agency"])

# 4. Random 70/30 split (no grouping)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# 5. StandardScaler fit_transform called TWICE on train (bug in notebook)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = scaler.fit_transform(X_train)  # duplicate line in notebook
X_test_scaled = scaler.transform(X_test)

# 6. Train models
# LinearRegression on scaled, DecisionTree/RandomForest on unscaled
```

**Result reported in notebook:**
- LinearRegression: MAE ~215, R² ~0.49
- DecisionTree: MAE ~0.08, R² ~0.9999
- RandomForest: MAE ~0.08, R² ~0.9999

---

## 2. Why the Notebook Results Are Invalid: Data Leakage

The notebook's near-perfect tree model scores are **entirely an artifact of data leakage**:

| Leakage Source | Impact |
|----------------|--------|
| **LabelEncoder fitted on full dataset** | Test categories inform the encoding; ordinal relationships imposed on cities |
| **Random 70/30 row split** | 100% of 6-column configurations in test also appear in train |
| **Target is deterministic** | 490 unique configurations → 490 unique prices, zero within-config variance |

**The target `price` is a deterministic function of 6 columns:**
```
from + to + flightType + time + distance + agency → price
```

Every configuration maps to exactly one price. The task is a **table lookup**, not regression. A random split places every test configuration in the training set, so tree models simply memorize the lookup table. The reported R² 0.9999 measures **memorization**, not generalization.

---

## 3. Merged Script (`merged_eda.py` / `eda_updated.py`): Two-Regime Design

The merged script reproduces the notebook's flawed approach **as Regime A** (for comparison) and introduces **Regime B** for honest evaluation.

### Regime A: Random Split (Notebook Reproduction)
- LabelEncoder on full dataset before split
- StandardScaler (with duplicate fit_transform) for LinearRegression
- Single random 70/30 train/test split (`random_state=42`)
- Same 3 models: LinearRegression, DecisionTree, RandomForest (100 trees)

### Regime B: GroupKFold on Origin (Leakage-Free)
- **GroupKFold on `from`** (9 origin cities → 5 folds, whole cities held out)
- **OneHotEncoder** (`handle_unknown='ignore'`) fitted **inside each fold**
- **StandardScaler** inside pipeline, fitted per-fold
- No test configuration ever seen during training (0% config overlap verified)
- Out-of-fold predictions over entire dataset (271,888 rows)

---

## 4. Critical Differences: Notebook vs. Merged Script

| Aspect | Original Notebook | Merged Script (Regime A) | Merged Script (Regime B) |
|--------|-------------------|--------------------------|--------------------------|
| **Encoding** | LabelEncoder on full data | Same (reproduces flaw) | OneHotEncoder per-fold |
| **Split** | Random 70/30 | Same | GroupKFold on `from` |
| **Scaling** | StandardScaler (double fit) | Same | Pipeline per-fold |
| **Config Overlap** | 100% (490/490) | 100% | **0%** (all folds) |
| **LinearRegression MAE** | ~215 | 215.2 | **167.3** |
| **DecisionTree MAE** | ~0.08 | 0.085 | **170.4** |
| **RandomForest MAE** | ~0.08 | 0.085 | **162.1** |
| **What It Measures** | Memorization | Memorization | **Generalization** |

**Key insight**: LinearRegression *improves* in Regime B because LabelEncoder's false ordinal relationships on cities harm the linear model. Tree models *collapse* because they can no longer memorize.

---

## 5. Leakage Analysis Findings (Core Discovery)

| Finding | Value | Evidence |
|---------|-------|----------|
| **Total rows** | 271,888 | `flights.csv` |
| **6-col configurations** | 490 | `GROUP_COLS_NO_DATE` |
| **Distinct prices** | 490 | 1:1 mapping |
| **Within-config price variance** | 0.0 | Zero — deterministic |
| **Minimal deterministic key** | `distance + flightType + from + agency` (4 cols) | Greedy search, step 4 |
| **Redundancy factor** | 554.9× | Rows per config (mean) |
| **Random split config overlap** | 100% (490/490) | `split_configuration_overlap.csv` |
| **GroupKFold config overlap** | 0% (all 5 folds) | Verified per fold |

**Implication**: The prediction task is a table lookup. Honest evaluation requires holding out **entire origin cities** (GroupKFold on `from`).

---

## 6. Two-Regime Results (Full Dataset: 271,888 rows, 5-fold GroupKFold)

| Model | Regime A: Random Split | Regime B: GroupKFold | MAE Inflation |
|-------|------------------------|----------------------|---------------|
| **LinearRegression** | MAE 215.2, R² 0.492 | MAE 167.3, R² 0.667 | **0.8×** (Regime B better!) |
| **DecisionTree** | MAE 0.085, R² 0.9999 | MAE 170.4, R² 0.602 | **1,998×** |
| **RandomForest** | MAE 0.085, R² 0.9999 | MAE 162.1, R² 0.623 | **1,914×** |

**Per-fold spread (Regime B):**
- LinearRegression: MAE 97.98–254.43 (R² 0.407–0.872)
- DecisionTree: MAE 129.99–218.42 (R² 0.352–0.771)
- RandomForest: MAE 120.65–211.67 (R² 0.370–0.794)

**A single mean hides 3× MAE range** — always report per-fold spread.

---

## 7. Permutation Importance (Regime B, Leakage-Free)

| Feature | MAE Increase | Interpretation |
|---------|--------------|----------------|
| `flightType` | 142.6 | Primary driver |
| `to` | 40.5 | Destination matters |
| `distance` | 37.8 | Distance matters |
| `time` | 17.7 | Time matters |
| `agency` | 0.76 | Minor |
| `from` | 0.0 | Held out by GroupKFold |
| `year` | 0.0 | **No signal** |
| `month` | 0.0 | **No signal** |
| `day` | 0.0 | **No signal** |

**Date features (`year`, `month`, `day`) have zero importance** — Streamlit date widgets cannot affect predictions.

---

## 8. Artifact Changes

### Before (Notebook Output — 5 files)
```
flight_price_model.joblib      # 22 MB RandomForest
from_encoder.joblib            # LabelEncoder for 'from'
to_encoder.joblib              # LabelEncoder for 'to'
flightType_encoder.joblib      # LabelEncoder for 'flightType'
agency_encoder.joblib          # LabelEncoder for 'agency'
```
**Problems**: 5 separate files, manual column ordering in `app.py`, `LabelEncoder` fails on unseen categories, no versioning.

### After (Merged Script Output — 2 files)
```
ml_artifacts/
├── flight_price_pipeline.joblib   # 20 MB Pipeline(ColumnTransformer + RandomForest)
└── manifest.json                  # Full metadata: features, categories, ranges, warning
```
**Benefits**: Self-contained, carries its own column contract, `handle_unknown='ignore'`, single versioned artifact, round-trip prediction verified.

---

## 9. Output Files (Unnumbered, 48 files in `eda_outputs/`)

| Category | Files |
|----------|-------|
| **EDA Plots** | `flight_price_distribution.png`, `flight_destination_frequency.png`, `hotel_place_frequency.png`, `hotel_expenditure_distribution.png`, `distance_vs_price.png`, `correlation_flights.png`, `correlation_hotels.png`, `monthly_travel_demand.png`, `monthly_expenditure.png`, `monthly_price_demand.png`, `seasonality_index.png`, `geographical_frequency.png`, `age_distribution.png`, `gender_distribution.png`, `user_total_expenditure.png`, `user_expenditure_by_flight_class.png`, `trip_expenditure_distribution.png`, `flight_price_by_class.png`, `flight_price_by_agency.png` |
| **EDA CSVs** | `data_quality.csv`, `monthly_price_demand.csv`, `seasonality_index.csv`, `adaptive_pricing_exploration.csv`, `annual_activity_price_summary.csv`, `geographical_frequency.csv`, `user_expenditure.csv`, `trip_level_data.csv` |
| **Leakage Analysis** | `configuration_cardinality.png`, `unique_price_levels.png`, `duplication_structure.png`, `within_configuration_price_variance.png`, `configuration_cardinality_summary.csv`, `with_date_configuration_stats.csv`, `no_date_configuration_stats.csv`, `unique_price_values.csv`, `minimal_deterministic_key.csv`, `split_configuration_overlap.csv`, `leakage_analysis.json` |
| **ML Metrics** | `ml_random_split_metrics.csv`, `ml_groupkfold_metrics.csv`, `ml_groupkfold_per_fold.csv`, `ml_regime_comparison.csv` |
| **ML Plots** | `ml_metric_bar_comparison.png`, `ml_groupkfold_fold_spread.png`, `ml_actual_vs_predicted_random_split.png`, `ml_actual_vs_predicted_groupkfold.png` |
| **Permutation Importance** | `ml_permutation_importance.csv`, `ml_permutation_importance.png` |
| **Model Artifact** | `ml_artifacts/flight_price_pipeline.joblib`, `ml_artifacts/manifest.json` |

---

## 10. Running the Scripts

### Original Notebook
```bash
cd /home/shahjahan/Projects/VoyageAnalytics_GroupProject
jupyter notebook "flight price prediction.ipynb"
# Produces 5 joblib files in project root
```

### Original EDA Script
```bash
cd /home/shahjahan/Projects/VoyageAnalytics_GroupProject
python eda.py
# Produces 20 files in eda_outputs/
```

### Enhanced Merged Script (`merged_eda.py` — with TXT reports)
```bash
cd /home/shahjahan/Projects/voyage/updated_eda
python merged_eda.py --max-threads 4              # Full run (~3-5 min)
python merged_eda.py --max-rows 40000 --cv-folds 3 --no-save-model  # Smoke test
```

### Updated Script (`eda_updated.py` — **no TXT reports, unnumbered outputs**)
```bash
cd /home/shahjahan/Projects/voyage/updated_eda
python eda_updated.py --max-threads 4              # Full run
python eda_updated.py --max-rows 5000 --skip-geo --cv-folds 2 --no-save-model  # Smoke test (48 files)
```

### Streamlit App (Uses Original 5 Joblib Files)
```bash
cd /home/shahjahan/Projects/VoyageAnalytics_GroupProject
streamlit run app.py
# http://localhost:8501
```
> **Note**: Current `app.py` loads the 5 separate joblib files from the notebook. It is **not compatible** with the new single `flight_price_pipeline.joblib`.

---

## Appendix A: Leakage Analysis Report

*Full content from `write_leakage_report()` — previously saved as `32_leakage_report.txt`*

```
LEAKAGE ANALYSIS - FLIGHT PRICE DATASET
==============================================================================

A. CONFIGURATION CARDINALITY
------------------------------------------------------------------------------
metric                                       with date             no date
n_unique_configurations                         149484                 490
n_unique_prices                                    490                 490
duplicated_configurations                        60594                 490
duplicated_rows                                 182998              271888
rows_per_configuration_mean                       1.82              554.87
redundancy_factor                                  1.8               554.9
configs_with_single_price                       149484                 490
deterministic_config_pct                         100.0               100.0
max_within_config_price_std                        0.0                 0.0
within_config_price_variance                       0.0                 0.0
unexplained_variance_ratio                         0.0                 0.0
price_is_deterministic                            True                True

B. THE DISTINCT PRICE VALUES
------------------------------------------------------------------------------
  distinct price values : 490
  rows per price value  : mean 554.9, min 108, max 1,261
  fully duplicated rows : 122,404
  duplicated 6-column configurations : 490
  redundancy factor     : 554.9x
  see unique_price_levels.png and duplication_structure.png

C. MINIMAL DETERMINISTIC KEY
------------------------------------------------------------------------------
Greedy subset search. Price becomes an exact function of target as soon as:
    distance + flightType + from + agency

D. SPLIT OVERLAP (notebook split: 70/30, random_state=42)
------------------------------------------------------------------------------
  from + to + flightType + time + distance + agency + year + month + day
    test configurations seen in train : 34,074 / 64,053
    test rows on unseen configurations : 33,794
  from + to + flightType + time + distance + agency
    test configurations seen in train : 490 / 490
    test rows on unseen configurations : 0

E. VERDICT
------------------------------------------------------------------------------
  price deterministic by configuration : True
  configurations needed               : 490
  distinct prices                     : 490
  rows in dataset                     : 271,888

  A random train/test split places every test
  configuration in train as well, so a tree ensemble
  can reproduce the target by table lookup.
  The reported R2 therefore measures memorisation,
  not generalisation to unseen routes.
```

---

## Appendix B: ML Training Report

*Full content from `write_ml_report()` — previously saved as `41_ml_report.txt`*

```
ML TRAINING REPORT - TWO REGIMES, SAME MODELS
==============================================================================

1. WHY THE DATASET REWARDS MEMORISATION
------------------------------------------------------------------------------
   rows in dataset                                : 271,888
   distinct price values                         : 490
   configurations on the 6-column key            : 490
   rows per configuration                        : mean 554.87, max 1,261
   configurations with exactly one price         : 490 (100.0%)
   price variance within a configuration         : 0.0
   minimal key that fixes price                  : distance + flightType + from + agency

   Every configuration maps to exactly one price, so the
   target is a table, not a smooth function. Any metric
   measured on rows the model has already seen
   measures recall of that table.

2. REGIME A - RANDOMIZED SPLIT (ORIGINAL METHOD)
------------------------------------------------------------------------------
   model                     MAE          RMSE              R2
   LinearRegression        215.2238      258.5118         0.491713
   DecisionTree              0.0853        2.6243         0.999948
   RandomForest              0.0847        0.8393         0.999995

   Leakage severity: 100.00% of held-out rows share a
   configuration with the training set, so the split
   cannot distinguish memorisation from generalisation.

3. REGIME B - GROUPKFOLD ON ORIGIN (LEAKAGE PREVENTED)
------------------------------------------------------------------------------
   model                     MAE          RMSE              R2          fold MAE
   LinearRegression        167.2871      209.0661         0.667031      166.14 +/- 65.09
   DecisionTree            170.3808      228.6652         0.601676      169.45 +/- 38.55
   RandomForest            162.0880      222.5324         0.622755      161.13 +/- 41.42

   0.0000% of validation rows share a configuration with
   their training folds. Scores are out-of-fold over all 271,888 rows.

4. THE GAP
------------------------------------------------------------------------------
   model                     MAE x        R2 drop
   LinearRegression           0.8x       -0.1753
   DecisionTree            1997.8x        0.3983
   RandomForest            1914.3x        0.3772

   The two regimes rank the models differently, which is
   the practical cost of reporting regime A alone: it
   selects on memorisation rather than on skill.

5. PER-FOLD SPREAD (why one number is misleading)
------------------------------------------------------------------------------
   LinearRegression          fold MAE min    97.98 | max   254.43 | R2 min 0.4065 | max 0.8718
   DecisionTree              fold MAE min   129.99 | max   218.42 | R2 min 0.3520 | max 0.7706
   RandomForest              fold MAE min   120.65 | max   211.67 | R2 min 0.3698 | max 0.7936

6. PERMUTATION IMPORTANCE (LEAKAGE-FREE)
------------------------------------------------------------------------------
   flightType                      142.59
   to                               40.50
   distance                         37.80
   time                             17.73
   agency                            0.76
   from                             0.00
   month                            0.00
   day                              0.00
   year                             0.00

   year / month / day / from sit at zero, confirming the
   Streamlit date widgets cannot change a prediction.

7. RECOMMENDATION
------------------------------------------------------------------------------
        - Quote regime B. Its numbers are what a new origin city would see.
        - Keep regime A only as a documented counter-example of leakage.
        - Drop year/month/day from the feature set and from app.py.
        - Never fit encoders or scalers outside the fold.
        - Report per-fold spread alongside any mean, since a single
          holdout figure on this dataset hides a 3x MAE range.
        - Add GroupKFold on origin to CI so the leak cannot return.
```

---

## Appendix C: EDA Summary Notes

*Previously saved as `20_eda_summary.txt`*

```
EDA NOTES (Voyage Analytics)
==========================

1. All observed cities are domestic Brazilian cities, so no
   international/domestic split can be derived from this data.

2. There is no business/pleasure purpose field. `flightType` holds
   economic / premium / firstClass, which are flight classes and
   must not be read as trip purpose.

3. Adaptive pricing is exploratory only. Inventory, booking lead
   time, competitor fares and cancellation data are all absent, so
   no causal dynamic-pricing claim is supportable.

4. Raw demand declines across the period while average price stays
   broadly flat, so falling expenditure is a volume effect.

5. Seasonal indices normalise complete years (2020-2022) so calendar
   effects are separated from the long-term decline.

6. Price is a deterministic function of configuration: 490
   configurations, 490 distinct prices, zero within-group
   variance. See unique_price_levels.png,
   duplication_structure.png, and the leakage/ML reports before
   quoting any model accuracy figure.

7. Every model is trained twice: once on a randomized 70/30 split
   as the original notebook did, and once under GroupKFold on the
   origin city with all preprocessing fitted inside each fold.
   Compare ml_regime_comparison.csv; only the second regime
   estimates performance on unseen routes.
```

---

*Generated from comparison of `flight price prediction.ipynb` (40 code cells) and `merged_eda.py` / `eda_updated.py` (2,100+ lines).*