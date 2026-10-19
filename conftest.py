"""
Root conftest.py  – shared fixtures for the PimaUdongo test suite.

Synthetic data only.  All generated numbers are labelled as such and
are never used in any final result table.
"""
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N_SYNTH = 120          # samples – enough for 5-fold CV with spatial blocks
COUNTRIES = ["Kenya", "Uganda", "Tanzania"]


def _make_synthetic_afsis_root(tmp_path: Path) -> Path:
    """
    Write a minimal synthetic AfSIS folder layout:
        <root>/Georeferences/georeferences.csv
        <root>/Wet_Chemistry/ICRAF/Wet_Chemistry_ICRAF.csv
        <root>/Wet_Chemistry/CROPNUTS/Wet_Chemistry_CROPNUTS.csv

    Column names match the patterns searched by afsis_experiment.find().
    Values are plausible but entirely fabricated – clearly labelled SYNTHETIC.
    """
    root = tmp_path / "afsis_synth" / "2009-2013"

    # ---- georeferences ----
    geo_dir = root / "Georeferences"
    geo_dir.mkdir(parents=True)

    ssns = [f"ICP{i:04d}" for i in range(N_SYNTH)]
    lats = RNG.uniform(-5, 5, N_SYNTH)          # East-African bounding box
    lons = RNG.uniform(30, 42, N_SYNTH)
    countries = RNG.choice(COUNTRIES, N_SYNTH)

    geo = pd.DataFrame({"SSN": ssns, "Latitude": lats, "Longitude": lons,
                        "Country": countries})
    geo.to_csv(geo_dir / "georeferences.csv", index=False)

    # ---- wet chemistry (two labs, split evenly) ----
    wc_dir = root / "Wet_Chemistry"

    def _wc_block(ssns_sub):
        n = len(ssns_sub)
        return pd.DataFrame({
            "SSN":        ssns_sub,
            # target: soil organic carbon (g/kg), log-normal-ish
            "OC":         np.clip(RNG.lognormal(1.0, 0.6, n), 0.1, 80),
            # features: sensor-measurable proxies
            "pH_H2O":     np.clip(RNG.normal(6.0, 0.7, n), 4.0, 9.0),
            "EC":         np.clip(RNG.exponential(0.2, n), 0.01, 2.0),
            "Clay":       np.clip(RNG.normal(30, 12, n), 3, 80),
            "Sand":       np.clip(RNG.normal(40, 15, n), 3, 90),
        })

    half = N_SYNTH // 2
    for lab, sl in [("ICRAF", ssns[:half]), ("CROPNUTS", ssns[half:])]:
        lab_dir = wc_dir / lab
        lab_dir.mkdir(parents=True)
        _wc_block(sl).to_csv(lab_dir / f"Wet_Chemistry_{lab}.csv", index=False)

    return root


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_afsis_root(tmp_path_factory):
    """Temporary synthetic AfSIS root directory (session-scoped for speed)."""
    tmp = tmp_path_factory.mktemp("afsis")
    return _make_synthetic_afsis_root(tmp)


@pytest.fixture(scope="session")
def synthetic_export(tmp_path_factory, synthetic_afsis_root):
    """
    Run the experiment on the synthetic fixture and export model + results.json
    into a temp folder.  Returns the export Path.
    (SYNTHETIC – never used in final results.)
    """
    import subprocess, sys
    export_dir = tmp_path_factory.mktemp("export")
    result = subprocess.run(
        [
            sys.executable, "afsis_experiment.py",
            "--root", str(synthetic_afsis_root),
            "--target", "OC",
            "--features", "pH_H2O", "EC", "Clay", "Sand",
            "--countries", *COUNTRIES,
            "--log-target",
            "--repeats", "2",   # fast
            "--blocks", "5",
            "--export", str(export_dir),
            "--out", str(export_dir / "results"),
        ],
        capture_output=True, text=True, cwd=Path(__file__).parent
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Synthetic experiment failed:\nSTDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return export_dir
