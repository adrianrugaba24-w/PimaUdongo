"""
Week-1 experiment (Path A): can soil organic carbon be predicted from properties a
low-cost sensor could measure, and does that survive spatial validation and sensor error?

Setup:
  1. Download the AfSIS Phase I data (see registry.opendata.aws/afsis), keeping its folder layout:
       <root>/Georeferences/georeferences.csv
       <root>/Wet_Chemistry/<LAB>/Wet_Chemistry_<LAB>.csv
  2. pip install pandas numpy scikit-learn matplotlib
Run:
  python afsis_experiment.py --root afsis/2009-2013 --list-columns
  python afsis_experiment.py --root afsis/2009-2013 --target "<column>" --features "<col1>" "<col2>" --countries Uganda Kenya Tanzania

NOT yet run on the real AfSIS files. Column names are auto-detected only as a guess:
check the printed choices and override with --target / --features if they are wrong.
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
    root = Path(root)
    geo = pd.read_csv(root / "Georeferences" / "georeferences.csv")
    parts = []
    for f in sorted((root / "Wet_Chemistry").glob("*/Wet_Chemistry_*.csv")):
        d = pd.read_csv(f, low_memory=False); d["lab"] = f.parent.name; parts.append(d)
    wet = pd.concat(parts, ignore_index=True, sort=False)
    return geo.merge(wet, on="SSN", how="inner")


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
    ap.add_argument("--root", required=True)
    ap.add_argument("--target"); ap.add_argument("--features", nargs="+")
    ap.add_argument("--countries", nargs="+")
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--log-target", action="store_true")
    ap.add_argument("--list-columns", action="store_true")
    ap.add_argument("--baseline-csv", help="CSV with SSN and baseline_oc (e.g. iSDAsoil value at each sample)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--export", help="folder to write model.joblib and results.json for the API and web page")
    a = ap.parse_args()

    df = load(a.root)
    if a.list_columns:
        print(df.country.value_counts() if "country" in df else df.filter(regex="(?i)country").iloc[:, 0].value_counts())
        print("\nColumns:\n" + "\n".join(df.columns)); return
    cc = find(df.columns, r"country"); lat = find(df.columns, r"^lat"); lon = find(df.columns, r"^lon")
    if not (lat and lon):
        sys.exit("Could not find latitude/longitude columns; run --list-columns.")
    if a.countries and cc:
        df = df[df[cc].isin(a.countries)]
    target = a.target or find(df.columns, r"(total.?c\b|^tc$|organic.?c|^oc$|^c$)")
    feats = a.features or [c for c in [find(df.columns, r"(^|[^a-z])ph([^a-z]|$)"),
                                      find(df.columns, r"(^ec($|[^a-z])|electrical)"),
                                      find(df.columns, r"clay"), find(df.columns, r"sand")] if c]
    print("Target:", target, "| Features:", feats, "| Rows before cleaning:", len(df))
    if not target or not feats:
        sys.exit("Auto-detection failed. Run --list-columns and pass --target and --features.")
    d = df.dropna(subset=[target, lat, lon] + feats).copy()
    d = d[d[target] > 0] if a.log_target else d
    print("Rows used:", len(d), "| Countries:", d[cc].value_counts().to_dict() if cc else "n/a")
    if len(d) < 50:
        sys.exit("Too few usable rows.")

    X0 = d[feats].to_numpy(float); y = d[target].to_numpy(float)
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


if __name__ == "__main__":
    main()
