"""
Week-1 experiment (Path A): can soil organic carbon be predicted from properties a
low-cost sensor could measure, and does that survive spatial validation and sensor error?

Data: AfSIS Phase I (registry.opendata.aws/afsis) – licence ODbL / ODC-By.
Credit: Africa Soil Information Service (AfSIS) and partners.

Setup:
  1. Download the AfSIS Phase I data, keeping its folder layout:
       <root>/Georeferences/georeferences.csv
       <root>/Wet_Chemistry/<LAB>/Wet_Chemistry_<LAB>.csv
  2. pip install -r requirements.txt
Run:
  python afsis_experiment.py --root afsis/2009-2013 --list-columns
  python afsis_experiment.py --root afsis/2009-2013 --target "<column>" \\
      --features "<col1>" "<col2>" --countries Uganda Kenya Tanzania \\
      --log-target --export export
  python afsis_experiment.py --synthetic   # smoke-test without real files

Column names are auto-detected; always check the printed choices and override
with --target / --features / --lat-col / --lon-col if they are wrong.
"""
import argparse, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0]   # sensor error as a fraction of each feature's SD


def load(root):
    """Load and merge AfSIS georeferences + wet-chemistry CSVs.

    Duplicated SSNs (same site measured by multiple labs) are kept; the caller
    is responsible for deduplication if required by the experimental design.
    """
    root = Path(root)
    geo = pd.read_csv(root / "Georeferences" / "georeferences.csv")
    parts = []
    for f in sorted((root / "Wet_Chemistry").glob("*/Wet_Chemistry_*.csv")):
        d = pd.read_csv(f, low_memory=False); d["lab"] = f.parent.name; parts.append(d)
    if not parts:
        sys.exit(f"No Wet_Chemistry CSV files found under {root / 'Wet_Chemistry'}.")
    wet = pd.concat(parts, ignore_index=True, sort=False)
    merged = geo.merge(wet, on="SSN", how="inner")
    n_before = len(merged)
    # Drop exact row-level duplicates; keep first occurrence per SSN
    merged = merged.drop_duplicates(subset="SSN", keep="first")
    if len(merged) < n_before:
        print(f"[load] Dropped {n_before - len(merged)} duplicate SSN rows (kept first lab's record).")
    return merged


def find(cols, pattern):
    for c in cols:
        if re.search(pattern, c, flags=re.I):
            return c
    return None


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def cv_scores(X, y, groups, spatial, model_fn, seed, log_target):
    splitter = GroupKFold(5).split(X, y, groups) if spatial else KFold(5, shuffle=True, random_state=seed).split(X)
    errs, base = [], []
    for tr, te in splitter:
        ytr = np.log1p(y[tr]) if log_target else y[tr]
        m = model_fn().fit(X[tr], ytr)
        p = m.predict(X[te]); p = np.expm1(p) if log_target else p
        errs.append(np.mean((p - y[te]) ** 2)); base.append(np.mean((y[tr].mean() - y[te]) ** 2))
    return np.sqrt(np.mean(errs)), np.sqrt(np.mean(base))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="AfSIS data root (required unless --synthetic is given)")
    ap.add_argument("--synthetic", action="store_true",
                    help="Generate synthetic AfSIS-shaped data and run a quick smoke-test "
                         "(no real files required; LABELLED SYNTHETIC – never used in results)")
    ap.add_argument("--target"); ap.add_argument("--features", nargs="+")
    ap.add_argument("--countries", nargs="+")
    ap.add_argument("--lat-col", default=None, help="Override latitude column name")
    ap.add_argument("--lon-col", default=None, help="Override longitude column name")
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--log-target", action="store_true")
    ap.add_argument("--list-columns", action="store_true")
    ap.add_argument("--baseline-csv", help="CSV with SSN and baseline_oc (e.g. iSDAsoil value at each sample)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--export", help="folder to write model.joblib and results.json for the API and web page")
    a = ap.parse_args()

    if a.synthetic:
        _run_synthetic_smoke_test(a)
        return
    if not a.root:
        ap.error("--root is required unless --synthetic is given.")

    df = load(a.root)
    if a.list_columns:
        cc_col = find(df.columns, r"(?i)country")
        if cc_col:
            print("Country counts:\n", df[cc_col].value_counts().to_string())
        else:
            print("(no country column found)")
        print("\nAll columns (" + str(len(df.columns)) + "):\n" + "\n".join(df.columns))
        print("\nMissing value counts (top 20):\n",
              df.isnull().sum().sort_values(ascending=False).head(20).to_string())
        return
    cc  = find(df.columns, r"(?i)country")
    lat = a.lat_col or find(df.columns, r"(?i)^lat")
    lon = a.lon_col or find(df.columns, r"(?i)^lon")
    if not (lat and lon):
        sys.exit("Could not find latitude/longitude columns; "
                 "run --list-columns and pass --lat-col / --lon-col.")
    if a.countries and cc:
        before = len(df)
        df = df[df[cc].isin(a.countries)]
        print(f"[filter] Kept {len(df)} / {before} rows matching countries: {a.countries}")
    target = a.target or find(df.columns, r"(total.?c\b|^tc$|organic.?c|^oc$|^c$)")
    feats = a.features or [c for c in [find(df.columns, r"(^|[^a-z])ph([^a-z]|$)"),
                                       find(df.columns, r"(^ec($|[^a-z])|electrical)"),
                                       find(df.columns, r"(?i)clay"),
                                       find(df.columns, r"(?i)sand")] if c]
    print("Target:", target, "| Features:", feats, "| Rows before cleaning:", len(df))
    if not target or not feats:
        sys.exit("Auto-detection failed. Run --list-columns and pass --target and --features.")
    d = df.dropna(subset=[target, lat, lon] + feats).copy()
    d = d[d[target] > 0] if a.log_target else d
    # ---- data card ----
    print("\n=== DATA CARD (REAL AfSIS DATA) ===")
    print(f"  Rows used (complete cases): {len(d)}")
    print(f"  Countries: {d[cc].value_counts().to_dict() if cc else 'n/a'}")
    print(f"  Target   : {target}  mean={d[target].mean():.2f}  sd={d[target].std():.2f}  "
          f"min={d[target].min():.2f}  max={d[target].max():.2f}")
    for f in feats:
        print(f"  {f:20s}: mean={d[f].mean():.2f}  sd={d[f].std():.2f}  "
              f"missing_pct={df[f].isnull().mean()*100:.1f}%")
    print(f"  Labs: {d['lab'].value_counts().to_dict() if 'lab' in d else 'n/a'}")
    print("==================================\n")
    if len(d) < 50:
        sys.exit("Too few usable rows (< 50) after filtering.")

    X0 = d[feats].to_numpy(float); y = d[target].to_numpy(float)
    # sd is the *population* spread of each feature, used to calibrate the noise
    # level scale (H2).  It is computed on the full dataset because it sets the
    # unit of the noise knob, not a learned transformation – no leakage.
    sd = X0.std(axis=0); sd[sd == 0] = 1
    rows = []
    models = {"Ridge": lambda: make_pipeline(StandardScaler(), Ridge(1.0)),
              "RandomForest": lambda: RandomForestRegressor(200, min_samples_leaf=2, n_jobs=-1, random_state=0)}
    for rep in range(a.repeats):
        rng = np.random.default_rng(rep)
        groups = KMeans(a.blocks, n_init=5, random_state=rep).fit_predict(d[[lat, lon]].to_numpy(float))
        for nl in NOISE_LEVELS:
            X = X0 + rng.normal(0, 1, X0.shape) * sd * nl
            for mname, fn in models.items():
                for spatial in (False, True):
                    e, b = cv_scores(X, y, groups, spatial, fn, rep, a.log_target)
                    rows.append(dict(model=mname, cv="spatial" if spatial else "random", noise=nl, rmse=e, baseline_rmse=b, rep=rep))
    res = pd.DataFrame(rows)
    summ = res.groupby(["model", "cv", "noise"]).agg(rmse=("rmse", "mean"), sd=("rmse", "std"), baseline=("baseline_rmse", "mean")).reset_index()
    summ.to_csv(f"{a.out}_summary.csv", index=False); print(summ.round(3).to_string(index=False))

    rf = summ[(summ.model == "RandomForest") & (summ.cv == "spatial")].sort_values("noise")
    worse = rf[rf.rmse >= rf.baseline]
    print("\nBreak-even noise (RF, spatial CV vs field-mean baseline):",
          f"{worse.noise.iloc[0]} x feature SD" if len(worse) else f"not reached up to {NOISE_LEVELS[-1]} x SD")

    if a.export:
        import json, joblib
        out_dir = Path(a.export); out_dir.mkdir(parents=True, exist_ok=True)
        groups = KMeans(a.blocks, n_init=5, random_state=0).fit_predict(d[[lat, lon]].to_numpy(float))
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X0, y, groups):
            m = models["RandomForest"]().fit(X0[tr], np.log1p(y[tr]) if a.log_target else y[tr])
            p = m.predict(X0[te]); oof[te] = np.expm1(p) if a.log_target else p
        half = float(np.quantile(np.abs(oof - y), 0.9))
        final = models["RandomForest"]().fit(X0, np.log1p(y) if a.log_target else y)
        joblib.dump(final, out_dir / "model.joblib")
        payload = dict(
            target=target, features=feats, log_target=bool(a.log_target), n_rows=int(len(d)),
            countries={str(k): int(v) for k, v in d[cc].value_counts().items()} if cc else {},
            cv_summary=[{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in r.items()} for r in summ.to_dict("records")],
            break_even_noise=float(worse.noise.iloc[0]) if len(worse) else None,
            interval=dict(level=0.9, half_width=half, source="90th percentile of absolute spatial-CV residuals (Random Forest)"),
            feature_ranges={f: [float(np.percentile(d[f], 5)), float(np.percentile(d[f], 95)), float(d[f].median())] for f in feats},
            noise_levels=NOISE_LEVELS)
        json.dump(payload, open(out_dir / "results.json", "w"), indent=1)
        print("Exported model.joblib and results.json to", out_dir)
    if a.baseline_csv:
        bl = pd.read_csv(a.baseline_csv).merge(d[["SSN", target]], on="SSN")
        print("Map baseline RMSE:", round(rmse(bl[target], bl["baseline_oc"]), 3), "on", len(bl), "samples")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for (m, c), g in summ.groupby(["model", "cv"]):
            ax.plot(g.noise, g.rmse, marker="o", label=f"{m}, {c} CV")
        ax.plot(rf.noise, rf.baseline, "k--", label="field-mean baseline")
        ax.set_xlabel("sensor noise (x feature SD)"); ax.set_ylabel("RMSE"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{a.out}_plot.png", dpi=150); print("Saved plot.")
    except Exception as ex:
        print("Plot skipped:", ex)


# ---------------------------------------------------------------------------
# Synthetic smoke-test (no real files needed)
# ---------------------------------------------------------------------------

def _run_synthetic_smoke_test(a):
    """Generate synthetic AfSIS-shaped data and run the full pipeline.

    OUTPUT IS LABELLED SYNTHETIC and must never appear in final result tables.
    """
    import tempfile
    print("=" * 60)
    print("SYNTHETIC SMOKE-TEST – data is fabricated, not real AfSIS.")
    print("=" * 60)
    rng = np.random.default_rng(0)
    n = 200
    ssns  = [f"SYN{i:04d}" for i in range(n)]
    lats  = rng.uniform(-5, 5, n)
    lons  = rng.uniform(30, 42, n)
    ctry  = rng.choice(["Kenya", "Uganda", "Tanzania"], n)
    oc    = np.clip(rng.lognormal(1.0, 0.6, n), 0.1, 80)
    ph    = np.clip(rng.normal(6.0, 0.7, n), 4.0, 9.0)
    ec    = np.clip(rng.exponential(0.2, n), 0.01, 2.0)
    clay  = np.clip(rng.normal(30, 12, n), 3, 80)
    sand  = np.clip(rng.normal(40, 15, n), 3, 90)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "afsis_synth" / "2009-2013"
        geo_dir = root / "Georeferences"; geo_dir.mkdir(parents=True)
        wc_dir  = root / "Wet_Chemistry" / "SYNTH"; wc_dir.mkdir(parents=True)
        pd.DataFrame({"SSN": ssns, "Latitude": lats, "Longitude": lons,
                      "Country": ctry}).to_csv(geo_dir / "georeferences.csv", index=False)
        pd.DataFrame({"SSN": ssns, "OC": oc, "pH_H2O": ph,
                      "EC": ec, "Clay": clay, "Sand": sand}).to_csv(
            wc_dir / "Wet_Chemistry_SYNTH.csv", index=False)
        # Override args to match synthetic columns
        a.root        = str(root)
        a.target      = "OC"
        a.features    = ["pH_H2O", "EC", "Clay", "Sand"]
        a.countries   = ["Kenya", "Uganda", "Tanzania"]
        a.log_target  = True
        a.repeats     = 2
        a.blocks      = 5
        a.lat_col     = "Latitude"
        a.lon_col     = "Longitude"
        a.list_columns = False
        a.baseline_csv = None
        # Run experiment inside the temp dir so exports also go there
        import json, joblib
        df = load(a.root)
        cc  = find(df.columns, r"(?i)country")
        lat = a.lat_col; lon = a.lon_col
        df = df[df[cc].isin(a.countries)]
        d  = df.dropna(subset=[a.target, lat, lon] + a.features).copy()
        d  = d[d[a.target] > 0]
        X0 = d[a.features].to_numpy(float); y_arr = d[a.target].to_numpy(float)
        sd = X0.std(axis=0); sd[sd == 0] = 1
        rows = []
        models = {"Ridge": lambda: make_pipeline(StandardScaler(), Ridge(1.0)),
                  "RandomForest": lambda: RandomForestRegressor(50, min_samples_leaf=2, n_jobs=-1, random_state=0)}
        for rep in range(a.repeats):
            rng2 = np.random.default_rng(rep)
            groups = KMeans(a.blocks, n_init=3, random_state=rep).fit_predict(d[[lat, lon]].to_numpy(float))
            for nl in NOISE_LEVELS:
                X = X0 + rng2.normal(0, 1, X0.shape) * sd * nl
                for mname, fn in models.items():
                    for spatial in (False, True):
                        e, b = cv_scores(X, y_arr, groups, spatial, fn, rep, a.log_target)
                        rows.append(dict(model=mname, cv="spatial" if spatial else "random",
                                         noise=nl, rmse=e, baseline_rmse=b, rep=rep))
        res  = pd.DataFrame(rows)
        summ = res.groupby(["model", "cv", "noise"]).agg(
            rmse=("rmse", "mean"), sd=("rmse", "std"),
            baseline=("baseline_rmse", "mean")).reset_index()
        print("\n[SYNTHETIC] Result summary:")
        print(summ.round(3).to_string(index=False))
        rf    = summ[(summ.model == "RandomForest") & (summ.cv == "spatial")].sort_values("noise")
        worse = rf[rf.rmse >= rf.baseline]
        print("\n[SYNTHETIC] Break-even:",
              f"{worse.noise.iloc[0]} x SD" if len(worse) else "not reached")
        print("\nSynthetic smoke-test complete. Results above are SYNTHETIC – do not report.")


if __name__ == "__main__":
    main()
