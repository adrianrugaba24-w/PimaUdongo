"""
Phase 2 - validation design.

Four ideas drive everything in this file:

1.  Folds depend only on (scheme, seed, block size) - never on the model, so
    every model is scored on exactly the same data.
2.  Everything learnable (imputation, scaling, hyperparameters) is fitted
    *inside* the training folds.  Nothing is fitted on a test fold.
3.  Metrics are computed on the original (untransformed) target scale, pooled
    over out-of-fold predictions, so a log-transform cannot flatter the score.
4.  The spatial-CV number is the headline; the random-CV number exists only to
    measure how much optimism the random split buys.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FOLDS = 5
INNER_FOLDS = 3
# Parallelism is a RUNTIME concern, not a modelling choice: it cannot change any
# number, only how long the run takes.  Learners whose fits are cheap (ridge,
# PLS, mean) stay in-process; the two tree learners, whose grid is the bulk of
# the wall clock, fork into at most GRID_JOBS worker processes.  Nested pools
# (grid n_jobs AND learner n_jobs both -1) oversubscribe badly on Windows and
# made a 'quick' 405-row run take 8 minutes, so each learner runs single-threaded.
N_JOBS = 1
GRID_JOBS = max(1, int(os.environ.get("PIMA_GRID_JOBS", "4")))
NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0]      # fraction of each feature's SD
BLOCK_SIZES = [5, 10, 20, 50]
ARMS = ("train_clean", "train_noisy")           # clean-train/noisy-test vs noisy/noisy
NOMINAL_COVERAGE = 0.90
# Deliberately no hard-coded grid grain: grid_resolution() picks one from the
# data.  A fixed 3x3 gave only 3-4 non-empty cells on clustered coordinates, and
# GroupKFold then failed with "number of splits greater than number of groups".


# ---------------------------------------------------------------------------
# Spatial blocking
# ---------------------------------------------------------------------------

def spatial_groups(coords: np.ndarray, ids: np.ndarray, blocks: int, seed: int):
    """
    Assign a fold group to every row, clustered on coordinates.

    Clustering is done on ONE POINT PER SITE (the site's coordinate centroid)
    and then mapped back onto the rows, so two rows of the same site or profile
    can never land in different folds.  Returns (row_groups, diagnostics).
    """
    coords = np.asarray(coords, float)
    ids = np.asarray(ids)
    uniq, first_idx = np.unique(ids, return_index=True)
    if len(uniq) < blocks:
        blocks = max(2, len(uniq) // 2)
    km = KMeans(n_clusters=blocks, n_init=10, random_state=seed)
    site_cluster = km.fit_predict(coords[first_idx])
    lookup = dict(zip(uniq.tolist(), site_cluster.tolist()))
    row_groups = np.array([lookup[i] for i in ids.tolist()])
    diag = dict(blocks=int(blocks), n_sites=int(len(uniq)),
                cluster_sizes={int(k): int(v) for k, v in
                               zip(*np.unique(row_groups, return_counts=True))},
                cluster_label=site_cluster.tolist())
    return row_groups, diag


def partition_signature(row_groups) -> frozenset:
    """
    A partition as a set of row-index sets.

    Cluster labels are arbitrary: two runs that number the same blocks
    differently have different label arrays and the same partition.  Comparing
    raw arrays therefore makes a stable partition look like it changed.
    """
    g = np.asarray(row_groups)
    return frozenset(frozenset(np.flatnonzero(g == v).tolist()) for v in np.unique(g))


def grid_groups(coords: np.ndarray, ids: np.ndarray, cells_per_side: int, seed: int):
    """
    Rectangular-block CV whose boundaries really move with the seed.

    The plane is rotated by a random angle and the grid origin is shifted by a
    random fraction of a cell, then the bounding box is cut into
    cells_per_side^2 cells.  KMeans blocking proved insensitive to its own seed
    (it converges to the same clusters), so this is the partition family that
    can answer 'how much of the spatial score depends on WHERE the hold-out
    boundaries fall?'.  One site gets one cell, so no site straddles a border.
    Returns (row_groups, diagnostics).
    """
    coords = np.asarray(coords, float)
    ids = np.asarray(ids)
    rng = np.random.default_rng(seed)
    uniq, first_idx = np.unique(ids, return_index=True)
    pt = coords[first_idx]
    theta = float(rng.uniform(0.0, np.pi / 2.0))
    c, s = np.cos(theta), np.sin(theta)
    rot = pt @ np.array([[c, -s], [s, c]])
    lo = rot.min(0)
    span = rot.max(0) - lo
    span = np.where(span > 0, span, 1.0)
    cell = span / cells_per_side
    idx = np.floor((rot - lo + rng.uniform(0.0, 1.0, 2) * cell) / cell).astype(int)
    idx = np.clip(idx, 0, cells_per_side - 1)
    present, site_cell = np.unique(idx[:, 0] * cells_per_side + idx[:, 1],
                                   return_inverse=True)
    lookup = dict(zip(uniq.tolist(), site_cell.tolist()))
    row_groups = np.array([lookup[i] for i in ids.tolist()])
    diag = dict(blocks=int(len(present)), n_sites=int(len(uniq)),
                cells_per_side=int(cells_per_side),
                rotation_deg=float(np.degrees(theta)),
                cluster_sizes={int(k): int(v) for k, v in
                               zip(*np.unique(row_groups, return_counts=True))})
    return row_groups, diag


def grid_resolution(coords, ids, target_groups: int, seeds, cap: int = 14) -> int:
    """
    Smallest square-grid resolution that gives at least `target_groups` non-empty
    cells for EVERY seed, so the partitions being compared all have the same
    grain and GroupKFold always has enough groups.  Returns cap if none is.
    """
    for side in range(2, cap + 1):
        if all(len(np.unique(grid_groups(coords, ids, side, s)[0])) >= target_groups
               for s in seeds):
            return side
    return cap


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(y, pred) -> dict:
    """Pooled out-of-fold metrics on the ORIGINAL target scale."""
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    resid = pred - y
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    rmse = float(np.sqrt(sse / len(y)))
    iqr = float(np.subtract(*np.percentile(y, [75, 25])))
    return dict(
        r2=float(1 - sse / sst) if sst > 0 else float("nan"),
        rmse=rmse,
        mae=float(np.mean(np.abs(resid))),
        # Ratio of performance to interquartile distance (Bellon-Maurel et al. 2010)
        rpiq=float(iqr / rmse) if rmse > 0 else float("nan"),
        n=int(len(y)),
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    steps: list = field(default_factory=list)     # (label, estimator) before the learner
    learner: object = None
    grid: dict = field(default_factory=dict)
    scale: bool = False
    target_transform: bool = True                 # False for the mean baseline
    default_params: dict = field(default_factory=dict)
    grid_jobs: int = 1                            # 1 = tune in-process (cheap learners)


def build_pipeline(spec: ModelSpec, log_target: bool = False, seed: int = 0):
    """Imputer -> [scaler] -> learner, optionally wrapped in a target transform.

    The learner step is always labelled "model", so grid keys are stable.  When
    a log target is used the learner is wrapped in TransformedTargetRegressor:
    it fits on log1p(y) but PREDICTS in original units, which is what keeps the
    reported metrics honest.
    """
    steps = [("impute", SimpleImputer(strategy="median"))]
    if spec.scale:
        steps.append(("scale", StandardScaler()))
    # clone(): the spec is reused across every fold, and Pipeline.fit mutates
    # its steps, so sharing one estimator instance would leak fitted state.
    learner = clone(spec.learner)
    if hasattr(learner, "random_state"):
        # Pin every stochastic learner to the run's seed.  Left unset, the RF
        # would draw from the global RNG and results would depend on the order
        # experiments happened to run in - i.e. not reproducible at all.
        learner.random_state = seed
    if spec.target_transform and log_target:
        learner = TransformedTargetRegressor(learner, func=np.log1p, inverse_func=np.expm1)
    steps.append(("model", learner))
    return Pipeline(steps)


def _grid_prefix(pipe: Pipeline) -> str:
    # TransformedTargetRegressor exposes its learner as `regressor`.
    return ("model__regressor__"
            if isinstance(pipe.named_steps["model"], TransformedTargetRegressor)
            else "model__")


def tunable_grid(spec: ModelSpec, pipe: Pipeline) -> dict:
    """{pipeline-key: list of values} for GridSearchCV."""
    p = _grid_prefix(pipe)
    return {p + k: v for k, v in spec.grid.items()}


def apply_params(pipe: Pipeline, params: dict | None) -> Pipeline:
    """Set fixed hyperparameters on an unfitted pipeline."""
    if params:
        p = _grid_prefix(pipe)
        pipe.set_params(**{p + k: v for k, v in params.items()})
    return pipe


def model_specs() -> dict[str, ModelSpec]:
    """
    Mean baseline, ridge, PLS, random forest, gradient boosting.
    Grids are deliberately tiny: they are tuned by nested CV inside each
    training fold, and a large grid would only buy selection noise.
    """
    specs = [
        ModelSpec("mean_baseline", learner=DummyRegressor(strategy="mean"),
                  target_transform=False),
        ModelSpec("ridge", scale=True, learner=Ridge(),
                  grid={"alpha": [0.3, 1.0, 3.0, 10.0]}),
        ModelSpec("pls", scale=True, learner=PLSRegression(max_iter=500),
                  grid={"n_components": [1, 2, 3]}),
        ModelSpec("random_forest",
                  learner=RandomForestRegressor(n_estimators=100, n_jobs=1),
                  grid={"min_samples_leaf": [1, 3], "max_features": [None, 0.5]},
                  default_params={"min_samples_leaf": 1, "max_features": None},
                  grid_jobs=GRID_JOBS),
        ModelSpec("gradient_boosting",
                  learner=GradientBoostingRegressor(n_estimators=100, random_state=0),
                  grid={"learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
                  default_params={"learning_rate": 0.05, "max_depth": 2},
                  grid_jobs=GRID_JOBS),
    ]
    return {s.name: s for s in specs}


# ---------------------------------------------------------------------------
# Fold machinery
# ---------------------------------------------------------------------------

def outer_splits(scheme: str, X, y, row_groups, seed: int):
    """The only place folds are decided.  Depends on (scheme, seed, groups) alone."""
    if scheme == "spatial":
        return list(GroupKFold(FOLDS).split(X, y, row_groups))
    return list(KFold(FOLDS, shuffle=True, random_state=seed).split(X))


def inner_splits(scheme: str, X_tr, groups_tr, seed: int):
    """Inner folds for tuning: blocked the same way as the outer scheme."""
    if scheme == "spatial":
        k = min(INNER_FOLDS, len(np.unique(groups_tr)))
        if k >= 2:
            return list(GroupKFold(k).split(X_tr, groups=groups_tr))
    return list(KFold(INNER_FOLDS, shuffle=True, random_state=seed).split(X_tr))


def _nan_sd(X: np.ndarray) -> np.ndarray:
    sd = np.nanstd(X, axis=0)
    return np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)


def _add_noise(X: np.ndarray, sd: np.ndarray, level: float, rng) -> np.ndarray:
    """
    Gaussian sensor error, scaled per feature by the TRAINING-fold SD.
    Missing values stay missing: noise cannot invent a reading that was never taken.
    """
    if level <= 0:
        return X
    return X + rng.normal(0.0, 1.0, X.shape) * sd * level


# ---------------------------------------------------------------------------
# One CV run
# ---------------------------------------------------------------------------


def run_cv(X, y, row_groups, spec: ModelSpec, scheme: str, seed: int,
           log_target: bool, params: dict | None = None, tune: bool = True,
           noise_level: float = 0.0, arm: str = "train_clean") -> dict:
    """
    Score one model under one validation scheme.

    tune=True runs a nested grid search inside every training fold and returns
    the parameters each fold selected.  tune=False uses `params` as given.
    Noise is drawn per fold from training-fold SD (see _nan_sd), so no test
    statistic ever enters the fit.
    """
    X = np.asarray(X, float); y = np.asarray(y, float)
    oof = np.full(len(y), np.nan)
    chosen, fold_rows = [], []
    for k, (tr, te) in enumerate(outer_splits(scheme, X, y, row_groups, seed)):
        rng = np.random.default_rng([seed, k, int(round(noise_level * 1000)),
                                     0 if arm == "train_clean" else 1])
        sd = _nan_sd(X[tr])                       # training-fold only: no test statistic
        Xtr = _add_noise(X[tr], sd, noise_level, rng) if arm == "train_noisy" else X[tr]
        Xte = _add_noise(X[te], sd, noise_level, rng)

        if tune and spec.grid:
            pipe = build_pipeline(spec, log_target, seed)
            gs = GridSearchCV(pipe, tunable_grid(spec, pipe), scoring="r2",
                              cv=inner_splits(scheme, Xtr, row_groups[tr], seed),
                              n_jobs=spec.grid_jobs)
            best = gs.fit(Xtr, y[tr]).best_estimator_
            # strip the pipeline path, keep only the learner's own parameter names
            chosen.append(dict(fold=k, **{kk.split("__")[-1]: vv
                                          for kk, vv in gs.best_params_.items()}))
        else:
            pipe = apply_params(build_pipeline(spec, log_target, seed),
                                params if params is not None else spec.default_params)
            best = pipe.fit(Xtr, y[tr])
        oof[te] = best.predict(Xte)
        fold_rows.append(dict(fold=k, n_test=int(len(te)), **metrics(y[te], oof[te])))

    return dict(metrics=metrics(y, oof), per_fold=fold_rows, chosen_params=chosen,
                oof=oof.tolist())


# ---------------------------------------------------------------------------
# Split-conformal prediction interval under spatial CV
# ---------------------------------------------------------------------------

def conformal_spatial(X, y, row_groups, spec: ModelSpec, params, seed: int,
                      log_target: bool, calib_fraction: float = 0.25,
                      nominal: float = NOMINAL_COVERAGE) -> dict:
    """
    Honest split-conformal: for every test fold, the model is fitted on part of
    the training blocks, calibrated on HELD-OUT training blocks, and the
    interval is then scored on test blocks it has never seen.

    The old prototype took the 90th percentile of the same residuals it claimed
    to cover - which measures calibration fit, not coverage.
    """
    X = np.asarray(X, float); y = np.asarray(y, float)
    alpha = 1.0 - nominal
    widths, coverages, per_fold = [], [], []
    for k, (tr, te) in enumerate(outer_splits("spatial", X, y, row_groups, seed)):
        gtr = row_groups[tr]
        rng = np.random.default_rng([seed, k, 7])
        gs = np.unique(gtr)
        n_calib = max(1, int(round(calib_fraction * len(gs))))
        calib_g = set(rng.choice(gs, size=n_calib, replace=False).tolist())
        in_calib = np.array([g in calib_g for g in gtr])
        fit_idx, cal_idx = tr[~in_calib], tr[in_calib]
        if len(cal_idx) < 5 or len(fit_idx) < 10:
            continue
        model = apply_params(build_pipeline(spec, log_target, seed), params).fit(X[fit_idx], y[fit_idx])
        scores = np.abs(model.predict(X[cal_idx]) - y[cal_idx])
        n = len(scores)
        # Split-conformal finite-sample level: ceil((n+1)(1-alpha))/n, capped at 1.
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        q = float(np.quantile(scores, level))
        pred_te = model.predict(X[te])
        inside = np.abs(pred_te - y[te]) <= q
        widths.append(2 * q)
        coverages.append(float(inside.mean()))
        per_fold.append(dict(fold=k, n_calib=int(len(cal_idx)), half_width=q,
                             coverage=float(inside.mean())))
    if not coverages:
        return dict(nominal=nominal, empirical_coverage=None, note="not enough data")
    return dict(nominal=nominal,
                empirical_coverage=float(np.mean(coverages)),
                coverage_sd=float(np.std(coverages)) if len(coverages) > 1 else 0.0,
                mean_width=float(np.mean(widths)),
                median_target=float(np.median(y)),
                per_fold=per_fold,
                method="split-conformal, blocks held out for calibration, "
                       "scored on unseen spatial test blocks",
                n_folds=int(len(coverages)))


# ---------------------------------------------------------------------------
# Experiment drivers
# ---------------------------------------------------------------------------

def _seeded_seeds(seed0: int, n: int) -> list[int]:
    return [seed0 + i for i in range(n)]


def sweep(X, y, row_groups, specs, schemes, seeds, log_target, params_by_key,
          tune: bool, label: str, noise_levels=(0.0,), arms=("train_clean",),
          extra: dict | None = None) -> list[dict]:
    """
    Run every (model x scheme x seed x noise x arm) cell and return tidy rows.
    `params_by_key` supplies fixed hyperparameters when tune=False.
    """
    rows = []
    extra = dict(extra or {})
    for name, spec in specs.items():
        for scheme in schemes:
            for level in noise_levels:
                for arm in arms:
                    per_seed, params_seen = [], []
                    for seed in seeds:
                        params = (params_by_key or {}).get((name, scheme))
                        try:
                            r = run_cv(X, y, row_groups, spec, scheme, seed, log_target,
                                       params=params, tune=tune, noise_level=level, arm=arm)
                        except Exception as ex:                       # never fake a number
                            rows.append(dict(extra, experiment=label, model=name, cv=scheme,
                                             noise=level, arm=arm, seed=seed,
                                             error=f"{type(ex).__name__}: {ex}"))
                            continue
                        per_seed.append(dict(seed=seed, **r["metrics"]))
                        params_seen.append(r["chosen_params"])
                    if not per_seed:
                        continue
                    row = dict(experiment=label, model=name, cv=scheme, noise=level,
                               arm=arm, n_seeds=len(per_seed))
                    for m in ("r2", "rmse", "mae", "rpiq"):
                        v = np.array([p[m] for p in per_seed], float)
                        v = v[np.isfinite(v)]
                        row[f"{m}_mean"] = float(v.mean()) if len(v) else float("nan")
                        row[f"{m}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
                    row["per_seed"] = per_seed
                    row["tuned_params"] = params_seen
                    if extra:
                        row.update(extra)
                    rows.append(row)
    return rows
