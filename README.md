# PimaUdongo — predicting soil organic carbon from sensor-type readings

Research prototype for a BSc proposal. The claim under test is **not** "we can
predict SOC well"; it is:

> reported accuracy depends on the validation design, so a soil model must be
> tested for geographic generalisation (spatially blocked CV) and for input
> noise before its score means anything.

The prototype is built to be **valid, not impressive**. Where a number is
unflattering it is printed anyway, and every number in the UI, the API and this
README is read out of `export/results.json`, which one command regenerates.

---

## STATUS — read this before quoting anything

The only data in this repository right now is a **labelled synthetic stand-in**
(`synthetic_afsis.py`). No real AfSIS files have been downloaded, no low-cost
sensor has been measured, and no Ugandan soil has been sampled by this project.
Every result below therefore describes a **simulation that was designed to
contain a random-vs-spatial gap**, and it is evidence about the *pipeline*, not
about African soils. `export/results.json` says so in its own record:

```
"data": { "kind": "synthetic_stand_in", "is_real_data": false, "stand_in_marker": true,
          "licence": "No third-party data in this run: ... that licence is NOT inherited
          by these numbers." }
```

The stand-in directory carries a `SYNTHETIC_STAND_IN.txt` marker, and the
experiment checks for it before it is allowed to label anything `afsis_real`.
Pointing `--root` at a folder is not treated as proof that the folder holds real
AfSIS data; `--synthetic` is the human override for other fake data.

---

## Install

```powershell
cd "c:\Users\HP\pimaUdongo version 2\PimaUdongo"
python -m pip install -r requirements.txt      # pandas numpy scikit-learn matplotlib
                                               # joblib fastapi uvicorn pydantic pytest httpx
```

Verified on Python 3.13.14 with scikit-learn 1.8.0, numpy 2.3.3, pandas 3.0.0
(the exact versions are recorded in `results.json -> package_versions`, and the
API refuses to load a model saved by a different sklearn major/minor).

## Run the experiment (one command, all artefacts)

```powershell
python afsis_experiment.py
```

Takes a few minutes on an 8-core machine. It builds the stand-in if needed,
cleans it, runs every sub-experiment, and writes:

| file | contents |
|---|---|
| `export/results.json` | every number, with provenance (data card, cleaning ledger, hashes, seeds, scope) |
| `export/model.joblib` | the deployed pipeline (imputation, scaling, target transform, learner) |
| `export/model_meta.json` | feature names/order, units, sklearn version, `results_sha256` |
| `export/results_summary.csv` | one row per (experiment, model, CV, noise, arm), mean over seeds |
| `export/results_runs.csv` | the same plus one row per seed |
| `export/results_plot.png` | noise curve, gap bars, block/partition sensitivity, coverage |

Useful flags: `--root <AfSIS dir>` (real data), `--synthetic`, `--countries`,
`--target` / `--features` / `--lat-col` / `--lon-col` (column overrides),
`--list-columns` (inspect a download before trusting the auto-detection),
`--blocks`, `--seeds`, `--seed0`, `--depth-cm`, `--lab-dupes`, `--target-units`,
`--export`, `--out`. `--quick` shrinks the scope for tests and is recorded as
`experiment_scope.quick = true` in the output, so a quick artefact can never be
mistaken for a reported one.

## Start the API and the demo

```powershell
$env:MODEL_DIR = "export"
python -m uvicorn soil_api:app --port 8000      # then open http://127.0.0.1:8000/
```

Endpoints: `GET /` (web demo), `GET /health`, `GET /model-info`, `GET /results`
(the raw `results.json`), `POST /predict`. The service **refuses to start** if
`model.joblib`, `model_meta.json` or `results.json` is missing, unreadable,
incomplete, edited after the run (`results_sha256` mismatch) or saved by another
sklearn version — it raises `BundleError` at import time rather than serving
predictions whose provenance it cannot vouch for.

## Run the tests

```powershell
python -m pytest -q
```

The session fixture runs the real experiment (quick scope) on a small stand-in
in a temp directory, so the API tests are checking a genuinely produced bundle,
not a mock.

---

## Data

**Intended data.** AfSIS Phase I (`s3://afsis`, AWS Open Data registry). The
pipeline expects `<root>/Georeferences/*` and `<root>/Wet_Chemistry/<LAB>/*`:

```powershell
aws s3 sync s3://afsis/2009-2013/Georeferences   afsis/2009-2013/Georeferences  --no-sign-request
aws s3 sync s3://afsis/2009-2013/Wet_Chemistry   afsis/2009-2013/Wet_Chemistry  --no-sign-request
python afsis_experiment.py --root afsis/2009-2013 --list-columns     # check the guess
python afsis_experiment.py --root afsis/2009-2013 --target <col> --features <c1> <c2> ...
```

**Licence.** The AWS registry describes AfSIS Phase I as ODbL, the dataset
README as ODC-By; read the `LICENSE` in the download and credit *Africa Soil
Information Service (AfSIS) and partners*. That licence applies to the real
download only — `results.json -> data.licence` states which of the two applies
to the run you are looking at, and for a stand-in run it explicitly disclaims
inheritance.

**What this run used** (`results.json -> data`, `dataset_version
afsis-3f-296718c140e3`, sha256 of each input file recorded in `data.files`):

- Target `SOC`, units **percent (g/100 g)** — inferred, and the inference is
  printed (`"median 2.16 < 15, read as a percentage"`). Features: `pH_H2O`
  (dimensionless), `EC` (dS/m), `Clay`, `Sand`, `Silt` percent. Any
  carbon-derived column (SOC sum, TOC, C:N) is *blocked* from the features and
  listed in `blocked_as_features` — a target-derived feature is the classic
  accidental leak.
- Depth rule: one **0–5 cm** record per site; deeper increments dropped.
- Cleaning ledger with a count and a reason for every removal (2152 merged rows
  → 407 usable):

| step | before | removed | after | reason |
|---|---|---|---|---|
| loaded | 2152 | 0 | 2152 | Georeferences × Wet_Chemistry merge |
| exact_duplicate_rows | 2152 | 12 | 2140 | rows identical in every column |
| missing_identifier | 2140 | 0 | 2140 | no profile/site/SSN id → cannot group folds |
| missing_coordinates | 2140 | 40 | 2100 | latitude or longitude absent |
| impossible_coordinates | 2100 | 21 | 2079 | out of range, non-numeric, or a 0/0 placeholder |
| same_record_same_lab | 2079 | 0 | 2079 | repeats within one lab after cleaning |
| depth_selection | 2079 | 1632 | 447 | kept one 0–5 cm record per site and lab |
| multi_lab_replicates | 447 | 39 | 408 | same site/depth in >1 lab; kept the alphabetically first |
| missing_target | 408 | 0 | 408 | no SOC label → cannot train on it |
| impossible_target | 408 | 1 | 407 | target outside (0, 100] for percent |
| impossible_value (pH, clay) | 407 | 0 | 407 | 2 readings set to NaN, imputed **inside** training folds |
| impossible_texture_sum | 407 | 2 | 407 | clay+sand > 100.5 → both set to NaN |
| final | 407 | 0 | **407** | usable rows (NaN may remain; imputation is per fold) |

- Rows and sites after cleaning: **407 / 407**. Countries: Nigeria 124,
  **Uganda 119**, Tanzania 103, Kenya 61. The proposal's country is reported
  separately, not blended: **Uganda subset 119 rows = 29.2 % of the African set**
  (subset SOC mean 2.184 %, SD 1.400; whole-set median 2.164 %).

---

## Results — random CV and spatial CV side by side

From `export/results.json` of this run (`model_version sha256:062d36c53a562f80`,
5 seeds `20261104`–`20261108`, 5 outer folds, 3 inner folds, 10 spatial blocks,
nested tuning on). **Synthetic stand-in — do not cite these as soil results.**
Spatial CV is listed first because it is the number the proposal cares about.
`RMSE`/`MAE` are in percent (g/100 g).

| model | R² spatial | R² random | gap (random − spatial) | RMSE spat | RMSE rnd | MAE spat | RPIQ spat | beats field mean (spatial) |
|---|---|---|---|---|---|---|---|---|
| random forest | **+0.611** | +0.761 | **+0.151 ± 0.006** | 0.729 | 0.571 | 0.573 | 2.29 | yes |
| gradient boosting | +0.590 | +0.750 | +0.159 ± 0.009 | 0.748 | 0.585 | 0.583 | 2.23 | yes |
| ridge | +0.428 | +0.648 | +0.220 ± 0.009 | 0.883 | 0.693 | 0.624 | 1.89 | yes |
| PLS | +0.393 | +0.625 | +0.232 ± 0.008 | 0.910 | 0.715 | 0.646 | 1.84 | yes |
| mean baseline | −0.361 | −0.005 | +0.356 ± 0.003 | 1.363 | 1.171 | 1.124 | 1.23 | **no** |

Field-mean baseline RMSE: 1.363 (spatial), 1.171 (random). Modes of the
nested-tuned hyperparameters: random forest `max_features=0.5,
min_samples_leaf=1`; gradient boosting `learning_rate=0.05, max_depth=2`;
ridge `alpha=10.0`; PLS `n_components=3`.

**Reading it honestly.** The gap is the point of the exercise: the same model,
same data, same folds for every model, scored 0.15–0.23 R² lower when whole
regions are held out instead of random rows. In this stand-in that gap is
*largely a designed property of the simulation* (the generator's relationship is
non-stationary by construction), which is exactly why the real AfSIS run has to
be done before any of this is claimed about soils.

### Controls

- **Shuffled-target R²** (must be ≈ 0): mean baseline −0.016, ridge −0.086,
  PLS −0.093, random forest −0.179, gradient boosting −0.140. No skill found in
  noise, so the pipeline is not reading signal out of the split design.
- **Block-size sensitivity** (spatial R², 5 → 10 → 20 → 50 blocks): random
  forest 0.638 → 0.610 → 0.680 → 0.771; ridge 0.472 → 0.457 → 0.495 → 0.588.
  More, smaller blocks make the task easier and drift toward random CV — with
  10 blocks the estimate sits near its most conservative value, and the reported
  spatial number should be read together with this curve.
- **Partition sensitivity** (blocks redrawn from 5 KMeans seeds at 10 blocks,
  one learner seed): …
- **Prediction interval**: split-conformal, calibrated on held-out spatial
  blocks and scored on test blocks never seen. Nominal **90 %** →
  **measured empirical coverage 0.824** (SD 0.076) under spatial CV, mean width
  **2.29** percentage points (±1.145). The interval is **under-covering in new
  areas** and is reported that way; the API returns it together with the
  achieved coverage, not with the nominal level.
- **Noise** (random forest, spatial CV; noise = Gaussian, scaled by each
  feature's SD *inside the training fold*): R² 0.610 → 0.603 → 0.590 → 0.518 →
  0.365 for clean-train/noisy-test at 0 / 0.1 / 0.25 / 0.5 / 1.0 SD, and 0.610 →
  0.615 → 0.586 → 0.524 → 0.310 for noisy-train/noisy-test. RMSE at 1.0 SD is
  0.931 / 0.970, still below the field mean (1.363), so **no break-even noise
  level was reached within the swept range** — reported as `null` rather than
  extrapolated.

---

## How the numbers are produced

- **Folds.** Random 5-fold KFold vs spatially blocked CV: KMeans on one point
  per site (its coordinate centroid), then GroupKFold over those clusters, so
  whole regions are held out and **records of one site/profile can never appear
  in two folds**. The same folds are used for every model and every scheme
  comparison.
- **Nothing is fitted before splitting.** Imputation, scaling, the log1p target
  transform and hyperparameter tuning all happen inside each training fold.
  `tests/test_validity.py` instruments `Pipeline.fit` and fails if any fit ever
  sees all rows. Tuning is a `GridSearchCV` with an inner GroupKFold *inside*
  the outer training fold (nested CV); no test-fold row is used for tuning.
- **Metrics.** Pooled out-of-fold R², RMSE and MAE on the **original target
  scale** (the log transform is inverted before scoring, so R² is not computed
  on a rescaled target). RPIQ = IQR(y)/RMSE (Bellon-Maurel et al. 2010).
- **Why some SDs are 0.000.** The main comparison deliberately fixes one spatial
  partition across seeds, which makes `R²_random − R²_spatial` a paired
  comparison; the cost is that a spatial cell's SD across seeds measures learner
  randomness only and is exactly 0.000 for deterministic models (ridge, PLS).
  The uncertainty that hides is measured instead in
  `results.partition_sensitivity` (where the boundaries fall) and
  `results.block_sensitivity` (how many blocks). `results.json` records this
  explanation in `validation.spatial_partition_note`.
- **No hard-coded numbers.** The API and the web page read `results.json`; the
  noise series plotted in the demo is discovered from whatever the run actually
  swept. If a cell fails, the run prints `!! FAILED cell …` and leaves the
  number absent (`null` / `NaN`) rather than substituting anything.

## Provenance and reproducibility

`results.json` stores the command line, UTC timestamp, runtime, git commit (if
any), package versions, per-file sha256 manifest and `dataset_version`, the
seed list, the experiment scope, and a `model_version` that is the sha256 of the
exported model bytes. `model_meta.json` stores a `results_sha256` binding the
model to the metrics reported beside it, so editing `results.json` after the run
makes the API refuse to start. Re-running
`python afsis_experiment.py` with the same inputs reproduces every metric
exactly (checked: identical `model_version` and R² values across runs; a test
asserts same-seed → same metrics). Parallel grid search affects wall-clock only.

---

## Limitations — what this prototype does **not** show

- **No real AfSIS data has been used.** All numbers above come from a synthetic
  stand-in whose spatial structure and non-stationarity were written into the
  generator so the pipeline could be tested end to end. The observed
  random-vs-spatial gap is therefore *not* a measurement of African soils, and
  the earlier version of this stand-in produced a gap of ≈ 0 — a bug in the
  generator, not a finding, fixed before these numbers were reported.
- **No low-cost sensor data.** The features are laboratory wet-chemistry values
  that a sensor could in principle proxy. Nothing has been measured with the
  planned hardware, so the noise sweep is a *model of* sensor error, not a
  measurement of it.
- **No Ugandan field validation.** No soil has been sampled, digested, or dried
  by this project. The Uganda subset of the stand-in (119 rows) is a count of
  simulated records, not evidence about Ugandan farms.
- **Not validated for agronomic decisions.** No rates, timing, carbon-credit or
  certification claim may be made from this prototype. The output is
  **predicted SOC** — an estimate of carbon content in percent — and is not a
  soil-quality rating, not a "soil health" index, and not a carbon stock.
- **Lab-to-lab bias is under-represented.** Replicate records from a second lab
  are collapsed rather than modelled, so measurement error is understated
  relative to a real field sensor.
- **The interval is marginal.** One width for all inputs, not conditional on the
  feature values, and its measured coverage (0.824) is below its nominal level
  (0.90) under spatial CV.
- **Physical bounds are heuristic.** The impossible-value ranges the API
  enforces (pH 0–14, EC 0–50 dS/m, texture 0–100 %) are physical limits, not
  AfSIS QC rules.
- **Depth scope.** Only the 0–5 cm band is modelled; nothing here extrapolates
  to subsoil.
- **Unit inference is a guess until checked.** Units are inferred from the
  distribution and printed; a real run must confirm them (g/kg vs % changes RMSE
  by 100×).

## What the first real-data run must check

1. `--list-columns` output: confirm target, features, coordinate and id columns
   before trusting anything.
2. The cleaning ledger: the Uganda row count, and that removals are the count
   you expect from the AfSIS QC notes.
3. Whether the spatial gap survives contact with real data. If `R²_spatial` is
   *not* below `R²_random`, that is a result, not a failure — report it as is
   and check whether the blocking is actually holding out distinct regions.
4. The empirical coverage of the interval, and whether the break-even noise
   level falls inside 0–100 % of feature SD once real sensor error is measured.
