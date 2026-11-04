"""
Phase 1 - data validity: loading, provenance, and an *auditable* cleaning pass.

Every row that leaves the pipeline leaves with a logged reason and a count.
Nothing here touches the target-based split; cleaning is deliberately blind to
validation design so that the two concerns never mix.

Column detection is a *guess against the real AfSIS schema*, which has not been
verified on this machine.  Every detected name is therefore reported in the data
card so a human can override it with --target / --features / --lat-col etc.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

# Target depth interval for SOC (decision 2026-11-04: surface only).
DEPTH_TARGET_CM = 5.0

# Physically impossible values.  A value outside these bounds is a recording or
# unit error, not a measurement, so it is set to NaN (features) or dropped
# (target) and counted.
BOUNDS = {
    "percent": (0.0, 100.0),     # clay / silt / sand
    "ph": (0.0, 14.0),
    "ec": (0.0, 50.0),           # dS/m (≈ mmho/cm)
    "oc_percent": (0.0, 100.0),
    "oc_g_per_kg": (0.0, 1000.0),
    "depth_cm": (0.0, 250.0),
}

# Column-name patterns, tried in order.  Patterns are deliberately narrow:
# a carbon-derived column must never end up in the feature list.
PATTERNS = {
    "lat": [r"(?i)^lat(itude)?$"],
    "lon": [r"(?i)^lon(g|gitude)$"],
    "country": [r"(?i)^country", r"(?i)country_?name"],
    "ssn": [r"(?i)^ssn$"],
    "site_id": [r"(?i)^site_?id$", r"(?i)^siteid$"],
    "profile_id": [r"(?i)^profile_?id$"],
    "depth_from": [r"(?i)depth_?from", r"(?i)^from_?cm$", r"(?i)top_?depth", r"(?i)^depth$"],
    "depth_to": [r"(?i)depth_?to", r"(?i)^to_?cm$", r"(?i)bottom_?depth"],
    # Target: explicit organic/total carbon.  "TCC"/"TC" are included because
    # AfSIS reports total carbon; --target lets you pin the exact column.
    "target": [r"(?i)^soc$", r"(?i)organic.?c(?!l)", r"(?i)^t(otal)?_?c$",
               r"(?i)^c$"],
    # Features: properties a low-cost sensor could plausibly proxy.
    "feature": {
        "ph": [r"(^|[^a-z])ph([^a-z]|$)"],
        "ec": [r"(?i)^ec($|[^a-z])", r"(?i)electrical"],
        "clay": [r"(?i)^clay"],
        "sand": [r"(?i)^sand"],
        "silt": [r"(?i)^silt"],
    },
    # Never allowed as a feature: anything carbon- or target-like.
    "forbidden_feature": [r"(?i)organic.?c", r"(?i)^t?c$", r"(?i)total.?c",
                          r"(?i)^soc", r"(?i)carbon"],
}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_manifest(root: Path) -> list[dict]:
    """Sorted per-file hashes, so the dataset version is reproducible."""
    files = sorted(p for p in Path(root).rglob("*.csv") if p.is_file())
    return [{"file": str(p.relative_to(root)).replace("\\", "/"),
             "bytes": p.stat().st_size,
             "sha256": sha256_file(p)} for p in files]


def dataset_version(manifest: list[dict]) -> str:
    """Short, stable id over the input files: h(<n files>:<hash of hashes>)."""
    agg = hashlib.sha256("".join(m["sha256"] for m in manifest).encode()).hexdigest()
    return f"afsis-{len(manifest)}f-{agg[:12]}"


def git_version() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def package_versions() -> dict:
    import sklearn, numpy, pandas
    return {"scikit-learn": sklearn.__version__, "numpy": numpy.__version__,
            "pandas": pandas.__version__}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw(root: str | Path) -> tuple[pd.DataFrame, list[dict]]:
    """
    Merge Georeferences x every Wet_Chemistry lab table, WITHOUT cleaning.

    Returns (rows, file_manifest).  Duplicated SSNs and depth increments are
    deliberately kept here - `clean()` decides what to do with them and logs it.
    """
    root = Path(root)
    geo_p = root / "Georeferences" / "georeferences.csv"
    if not geo_p.is_file():
        raise FileNotFoundError(f"{geo_p} not found. Check --root and the AfSIS "
                                f"folder layout (<root>/Georeferences/..., "
                                "<root>/Wet_Chemistry/<LAB>/...).")
    geo = pd.read_csv(geo_p)
    parts = []
    for f in sorted((root / "Wet_Chemistry").glob("*/Wet_Chemistry_*.csv")):
        d = pd.read_csv(f, low_memory=False)
        d["lab"] = f.parent.name
        parts.append(d)
    if not parts:
        raise FileNotFoundError(f"No Wet_Chemistry CSV files under "
                                f"{root / 'Wet_Chemistry'}.")
    wet = pd.concat(parts, ignore_index=True, sort=False)
    if "SSN" not in geo.columns or "SSN" not in wet.columns:
        raise ValueError("Merged tables have no SSN column to join on.")
    merged = geo.merge(wet, on="SSN", how="inner")
    return merged, file_manifest(root)


def load(root: str | Path) -> pd.DataFrame:
    """Backwards-compatible wrapper: raw merge only, no cleaning."""
    return load_raw(root)[0]


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def find(cols, pattern: str):
    """First column matching `pattern` (case-insensitive). Kept for the tests."""
    for c in cols:
        if re.search(pattern, c, flags=re.I):
            return c
    return None


def _find_any(cols, patterns: list[str]):
    for pat in patterns:
        hit = find(cols, pat)
        if hit:
            return hit
    return None


@dataclass
class Schema:
    """Resolved column names, plus how each was obtained (auto | override)."""
    lat: str | None = None
    lon: str | None = None
    country: str | None = None
    ssn: str | None = None
    site_id: str | None = None
    profile_id: str | None = None
    depth_from: str | None = None
    depth_to: str | None = None
    target: str | None = None
    features: list[str] = field(default_factory=list)
    origin: dict = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)

    def group_key(self) -> str:
        """Fold-grouping column: profile > site > SSN (never an arbitrary row)."""
        return self.profile_id or self.site_id or self.ssn


def detect_schema(df: pd.DataFrame, overrides: dict | None = None) -> Schema:
    cols = list(df.columns)
    ov = overrides or {}
    s = Schema()
    for role in ("lat", "lon", "country", "ssn", "site_id", "profile_id",
                 "depth_from", "depth_to", "target"):
        if ov.get(role):
            setattr(s, role, ov[role])
            s.origin[role] = "override"
        else:
            setattr(s, role, _find_any(cols, PATTERNS[role]))
            s.origin[role] = "auto" if getattr(s, role) else "missing"

    if ov.get("features"):
        s.features = list(ov["features"])
        s.origin["features"] = "override"
    else:
        picked, banned = [], [s.target] if s.target else []
        for name, pats in PATTERNS["feature"].items():
            hit = _find_any(cols, pats)
            if hit and hit not in picked and hit not in banned:
                picked.append(hit)
        s.features = picked
        s.origin["features"] = "auto"

    # Guard against a target-derived feature: never keep a carbon-like column.
    forbidden = [c for c in s.features if any(re.search(p, c) for p in PATTERNS["forbidden_feature"])]
    if forbidden:
        s.features = [c for c in s.features if c not in forbidden]
        s.unmatched = forbidden
    missing = [c for c in [s.target, s.lat, s.lon] + s.features if c and c not in cols]
    if missing:
        raise ValueError(f"Configured columns not in the data: {missing}")
    return s


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def _bounds_for(col: str, target_units: str | None):
    c = col.lower()
    if col.startswith("pH") or re.search(r"(^|[^a-z])ph([^a-z]|$)", col):
        return BOUNDS["ph"]
    if re.search(r"(?i)^ec($|[^a-z])|electrical", c):
        return BOUNDS["ec"]
    if re.search(r"(?i)clay|sand|silt|percent|%", c):
        return BOUNDS["percent"]
    return None


def infer_target_units(values: pd.Series) -> tuple[str, str]:
    """Return (units, how).  AfSIS reports carbon either as % or g/kg."""
    med = float(np.nanmedian(values)) if len(values) else float("nan")
    if not np.isfinite(med):
        return "unknown", "no data"
    return ("percent (g/100 g)", f"median {med:.2f} < 15, read as a percentage") if med < 15 \
        else ("g/kg", f"median {med:.2f} >= 15, read as g/kg")


def clean(df: pd.DataFrame, schema: Schema, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """
    Auditable cleaning pass.  Returns (rows, log) where each log entry records
    what was removed, how many rows, and why.  Row removals use `drop`;
    implausible *feature* values are set to NaN instead, so the row survives and
    imputation happens inside the training folds (Phase 2).
    """
    log: list[dict] = []
    d = df.copy()

    def step(name: str, before: int, reason: str, removed: int = 0, **extra):
        log.append(dict(step=name, n_before=int(before), n_removed=int(removed),
                        n_after=int(len(d)), reason=reason, **extra))

    n0 = len(d)
    step("loaded", n0, "rows after Georeferences x Wet_Chemistry merge")

    # --- 1. exact duplicates first ------------------------------------------
    # Removing pure re-recorded rows before anything else keeps the later
    # duplicate steps counting *real* replicates rather than copy-paste noise.
    before = len(d)
    d = d.drop_duplicates()
    step("exact_duplicate_rows", before, "rows identical in every column",
         removed=before - len(d))

    # --- 2. identifiers -----------------------------------------------------
    gk = schema.group_key()
    if gk is None:
        raise ValueError("No site/profile/SSN identifier column found; cannot "
                         "guarantee that records from one site stay in one fold.")
    before = len(d)
    d = d[d[gk].notna()]
    step("missing_identifier", before, "no profile/site/SSN id -> cannot group folds",
         removed=before - len(d), key_column=gk)

    # --- 3. coordinates -----------------------------------------------------
    before = len(d)
    d = d[d[schema.lat].notna() & d[schema.lon].notna()]
    step("missing_coordinates", before, "latitude or longitude absent",
         removed=before - len(d))

    before = len(d)
    lat = pd.to_numeric(d[schema.lat], errors="coerce")
    lon = pd.to_numeric(d[schema.lon], errors="coerce")
    bad = (lat.abs() > 90) | (lon.abs() > 180) | ((lat == 0) & (lon == 0)) | lat.isna() | lon.isna()
    d = d[~bad.to_numpy()]
    step("impossible_coordinates", before,
         "lat/lon out of range, non-numeric, or a 0/0 placeholder",
         removed=before - len(d))

    # --- 4. repeated rows for one record inside one lab ---------------------
    rec_key = [gk] + [c for c in (schema.depth_from, schema.depth_to) if c]
    lab = "lab" if "lab" in d.columns else None
    if lab:
        before = len(d)
        d = d.sort_values(rec_key + [lab]).drop_duplicates(subset=rec_key + [lab], keep="first")
        step("same_record_same_lab", before,
             "repeated rows for one site/depth within one lab (post-cleaning repeats)",
             removed=before - len(d), key=rec_key + [lab])

    # --- 5. depth selection -------------------------------------------------
    d, depth_info = _select_surface(d, schema, cfg.get("depth", DEPTH_TARGET_CM), lab)
    log.append(depth_info)

    # --- 6. multi-lab replicates --------------------------------------------
    key = rec_key
    dup_mode = cfg.get("lab_dupes", "first")
    before = len(d)
    n_dupe_rows = int(d.duplicated(subset=key, keep=False).sum())
    if dup_mode == "first":
        d = d.sort_values(key + ([lab] if lab else [])).drop_duplicates(subset=key, keep="first")
        step("multi_lab_replicates", before,
             f"same site/depth measured by >1 lab; kept the alphabetically first lab "
             f"(mode=first; {n_dupe_rows} rows were replicates of the same record)",
             removed=before - len(d), key=key)
    elif dup_mode == "mean":
        num = [c for c in d.select_dtypes("number").columns if c not in key]
        attrs = [c for c in (schema.lat, schema.lon, schema.country) if c and c in d.columns]
        keep_attrs = d.drop_duplicates(key)[key + attrs]
        d = d.groupby(key, as_index=False)[num].mean().merge(keep_attrs, on=key, how="left")
        step("multi_lab_replicates", before,
             f"same site/depth measured by >1 lab; numeric columns averaged (mode=mean; "
             f"lab column and per-row attributes are not averaged)",
             removed=before - len(d), key=key)
    else:  # keep -> replicate rows stay in the data and share one group
        step("multi_lab_replicates", before,
             f"kept all {n_dupe_rows} multi-lab replicate rows (mode=keep); they share a "
             "site id, so they cannot leak across folds", removed=0, key=key)

    # --- 5. target ----------------------------------------------------------
    units, how = infer_target_units(pd.to_numeric(d[schema.target], errors="coerce"))
    if cfg.get("target_units") and cfg["target_units"] != "auto":
        units = {"percent": "percent (g/100 g)", "g_per_kg": "g/kg"}.get(
            cfg["target_units"], cfg["target_units"])
        how = f"forced by --target-units={cfg['target_units']}"
    lo, hi = BOUNDS["oc_percent"] if units.startswith("percent") else BOUNDS["oc_g_per_kg"]

    before = len(d)
    d[schema.target] = pd.to_numeric(d[schema.target], errors="coerce")
    d = d[d[schema.target].notna()]
    step("missing_target", before,
         "target absent; a row with no SOC label cannot be trained on",
         removed=before - len(d))

    before = len(d)
    bad = (d[schema.target] <= lo) | (d[schema.target] > hi)
    d = d[~bad.to_numpy()]
    step("impossible_target", before,
         f"target outside ({lo}, {hi}] for units={units}", removed=before - len(d))

    # --- 6. features: implausible -> NaN (row kept) -------------------------
    for f in schema.features:
        b = _bounds_for(f, units)
        if b is None:
            continue
        d[f] = pd.to_numeric(d[f], errors="coerce")
        before = len(d)
        bad = (d[f] < b[0]) | (d[f] > b[1])
        n_bad = int(bad.sum())
        if n_bad:
            d.loc[bad.to_numpy(), f] = np.nan
        log.append(dict(step=f"impossible_value:{f}", n_before=before, n_removed=n_bad,
                        n_after=before,
                        reason=f"{n_bad} values outside [{b[0]}, {b[1]}] set to NaN "
                               "(imputed inside training folds, not deleted)",
                        bounds=[b[0], b[1]]))

    # Texture sum: clay + sand > 100 is impossible; NaN both, keep the row.
    clay = next((c for c in schema.features if "clay" in c.lower()), None)
    sand = next((c for c in schema.features if "sand" in c.lower()), None)
    if clay and sand:
        bad = (d[clay] + d[sand]) > 100.5
        n_bad = int(bad.sum())
        if n_bad:
            d.loc[bad.to_numpy(), [clay, sand]] = np.nan
        log.append(dict(step="impossible_texture_sum", n_before=len(d), n_removed=n_bad,
                        n_after=len(d),
                        reason=f"{n_bad} rows with clay+sand > 100.5 -> both set to NaN"))

    step("final", len(d), "usable rows (features may still contain NaN; "
                          "imputation happens inside each training fold)",
         n_sites=int(d[gk].nunique()), target_units=units, units_basis=how)
    return d.reset_index(drop=True), log


def _select_surface(d: pd.DataFrame, schema: Schema, max_cm: float, lab: str | None = None):
    """
    Keep exactly one 0-max_cm record per site/profile/lab and log the decision.

    The deduplication key deliberately keeps `lab`, so genuine multi-lab
    replicates survive this step and are accounted for separately afterwards.

    If the table has no depth column at all, the rule cannot be honoured and
    that is reported as `applied=False` rather than silently ignored.
    """
    n_start = len(d)
    info = dict(step="depth_selection", n_before=n_start, n_removed=0, n_after=n_start,
                reason="no depth column detected - the 0-5 cm rule could NOT be "
                       "applied; rows are whatever the source table provides",
                applied=False, target_cm=max_cm)
    if schema.depth_to is None and schema.depth_from is None:
        return d, info

    d = d.copy()
    for c in (schema.depth_from, schema.depth_to):
        if c:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if schema.depth_from and schema.depth_to:
        top, base = d[schema.depth_from], d[schema.depth_to]
    elif schema.depth_to:                       # bottom depth only -> assume it starts at 0
        top, base = pd.Series(0.0, index=d.index), d[schema.depth_to]
    else:                                       # top depth only -> assume a 5 cm slice
        top, base = d[schema.depth_from], d[schema.depth_from] + 5.0
    # Work on real columns so sorting/grouping stays label-based.
    d["_top_cm"], d["_base_cm"] = top, base

    bad = ((d["_top_cm"] < 0) | (d["_base_cm"] <= d["_top_cm"])
           | (d["_base_cm"] > BOUNDS["depth_cm"][1])
           | d["_top_cm"].isna() | d["_base_cm"].isna())
    n_bad_depth = int(bad.sum())
    d = d[~bad.to_numpy()]

    gk = schema.group_key()
    rec_key = [gk] + [c for c in (schema.depth_from, schema.depth_to) if c]
    if lab:
        rec_key = rec_key + [lab]
    surface = ((d["_top_cm"] <= max_cm) & (d["_base_cm"] <= max_cm + 0.5)).fillna(False)
    if surface.any():
        keep = d[surface.to_numpy()].sort_values(rec_key + ["_top_cm", "_base_cm"]) \
            .drop_duplicates(rec_key, keep="first")
        reason = (f"kept one 0-{max_cm:.0f} cm record per site and lab; deeper "
                  f"intervals dropped")
    else:
        keep = d.sort_values(rec_key + ["_top_cm", "_base_cm"]).groupby(rec_key).head(1)
        reason = (f"no interval within 0-{max_cm:.0f} cm existed anywhere; kept the "
                  f"shallowest record per site instead")
    keep = keep.drop(columns=["_top_cm", "_base_cm"]).reset_index(drop=True)
    # n_before is the row count on ENTRY to this step, so the broken-depth rows
    # and the discarded deeper intervals are both accounted for: the ledger must
    # stay additive (n_before - n_removed == n_after) for every row-level step.
    info.update(n_removed=n_start - len(keep), n_after=len(keep), applied=True,
                n_bad_depth_rows=n_bad_depth, reason=reason)
    return keep, info


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def data_card(d: pd.DataFrame, schema: Schema, log: list[dict], focus_country: str) -> dict:
    """Row counts per step + the focus-country subset reported separately."""
    countries = {}
    if schema.country:
        countries = {str(k): int(v) for k, v in d[schema.country].value_counts().items()}
    gk = schema.group_key()
    card = dict(
        n_rows=int(len(d)),
        n_sites=int(d[gk].nunique()),
        group_column=gk,
        countries=countries,
        target=schema.target,
        target_units=next((l["target_units"] for l in log if "target_units" in l), "unknown"),
        features=list(schema.features),
        n_feature_nan={f: int(d[f].isna().sum()) for f in schema.features},
    )
    if schema.country and focus_country in countries:
        sub = d[d[schema.country] == focus_country]
        card["focus_subset"] = dict(
            country=focus_country, n_rows=int(len(sub)),
            n_sites=int(sub[gk].nunique()),
            share_of_rows=float(len(sub) / max(len(d), 1)),
            target_mean=float(sub[schema.target].mean()),
            target_sd=float(sub[schema.target].std()),
        )
    else:
        card["focus_subset"] = dict(country=focus_country, n_rows=0, n_sites=0,
                                    note=f"{focus_country} not present after cleaning")
    return card


def print_card(card: dict, log: list[dict], schema: Schema):
    print("\n=== DATA CARD ===")
    print(f"  columns: target={schema.target} ({schema.origin.get('target')})  "
          f"features={schema.features} ({schema.origin.get('features')})")
    print(f"  lat/lon: {schema.lat}/{schema.lon}   group key: {schema.group_key()}")
    if schema.unmatched:
        print(f"  BLOCKED as features (carbon/target-like): {schema.unmatched}")
    print("  cleaning ledger:")
    for l in log:
        print(f"    {l['step']:28s} before={l['n_before']:>7} removed={l['n_removed']:>6} "
              f"after={l['n_after']:>7}  {l['reason']}")
    print(f"  rows={card['n_rows']}  sites={card['n_sites']}  units={card['target_units']}")
    print(f"  countries: {card['countries']}")
    f = card["focus_subset"]
    print(f"  {f['country']} subset: {f.get('n_rows', 0)} rows / {f.get('n_sites', 0)} sites "
          f"({f.get('share_of_rows', 0) * 100:.1f}% of rows) vs the full African set "
          f"({card['n_rows']} rows)")
    print(f"  NaN left in features (imputed per fold): {card['n_feature_nan']}")
    print("=================\n")
