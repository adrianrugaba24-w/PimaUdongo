"""
Root conftest.py - shared fixtures for the PimaUdongo test suite.

Everything here runs on the SYNTHETIC stand-in generator.  No test touches real
AfSIS files, and no number produced here may appear in a reported result.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent

# Small on purpose: the suite must stay runnable.  The generator's injected
# problems are ABSOLUTE counts, so a small frame still exercises every cleaning
# branch.
N_SITES = 160
N_CLUSTERS = 8
GEN_SEED = 4242

import synthetic_afsis as S      # noqa: E402
import afsis_data as A           # noqa: E402
import afsis_validation as V     # noqa: E402

# Fold blocking used by the unit-test fixtures.  Small so that a 160-site frame
# still leaves >= 2 sites per held-out block.
TEST_BLOCKS = 6


@pytest.fixture(scope="session")
def synthetic_afsis_root(tmp_path_factory):
    """A small synthetic AfSIS-shaped dataset on disk (SYNTHETIC)."""
    tmp = tmp_path_factory.mktemp("afsis")
    root = tmp / "afsis_stand_in"
    S.write_dataset(root, seed=GEN_SEED, n_sites=N_SITES, n_clusters=N_CLUSTERS)
    return root


@pytest.fixture(scope="session")
def synthetic_export(tmp_path_factory, synthetic_afsis_root):
    """
    Run the real experiment end to end on the small stand-in and export the
    bundle the API needs.  Session-scoped: one run serves every API test.
    """
    export_dir = tmp_path_factory.mktemp("export")
    res = subprocess.run(
        [sys.executable, "afsis_experiment.py",
         "--root", str(synthetic_afsis_root),
         "--quick", "--seeds", "2", "--blocks", "6",
         "--export", str(export_dir),
         "--out", str(export_dir / "results")],
        capture_output=True, text=True, cwd=str(ROOT))
    if res.returncode != 0:
        raise RuntimeError(
            f"Synthetic experiment failed ({res.returncode}).\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    for f in ("results.json", "model.joblib", "model_meta.json"):
        assert (export_dir / f).is_file(), f"experiment did not write {f}"
    return export_dir


@pytest.fixture(scope="session")
def meta(synthetic_export):
    return json.loads((synthetic_export / "results.json").read_text())


@pytest.fixture(scope="session")
def api(synthetic_export):
    """
    soil_api imported against the synthetic export.

    Importing IS the start-up check: load_bundle() raises BundleError if the
    model, its metadata or results.json are missing or inconsistent, so a
    fixture that returns a module also proves the service was willing to start.
    """
    os.environ["MODEL_DIR"] = str(synthetic_export)
    sys.path.insert(0, str(ROOT))
    sys.modules.pop("soil_api", None)
    import soil_api
    return soil_api


@pytest.fixture(scope="session")
def client(api):
    from fastapi.testclient import TestClient
    return TestClient(api.app)


@pytest.fixture(scope="session")
def raw_frame(synthetic_afsis_root):
    df, manifest = A.load_raw(synthetic_afsis_root)
    return df, manifest


@pytest.fixture(scope="session")
def schema(raw_frame):
    return A.detect_schema(raw_frame[0])


@pytest.fixture(scope="session")
def cleaned(raw_frame, schema):
    """(rows, cleaning_log) from the audited cleaning pass."""
    return A.clean(raw_frame[0], schema, dict(depth=A.DEPTH_TARGET_CM,
                                              target_units="auto", lab_dupes="first"))


@pytest.fixture(scope="session")
def cleaned_keep_dupes(raw_frame, schema):
    """Cleaning with multi-lab replicates retained, so one site has >1 row."""
    return A.clean(raw_frame[0], schema, dict(depth=A.DEPTH_TARGET_CM,
                                              target_units="auto", lab_dupes="keep"))


@pytest.fixture(scope="session")
def pipeline_inputs(schema, cleaned):
    """(X, y, ids, coords, row_groups) exactly as the experiment builds them."""
    d, log = cleaned
    ids = d[schema.group_key()].to_numpy()
    coords = d[[schema.lat, schema.lon]].to_numpy(float)
    groups, diag = V.spatial_groups(coords, ids, TEST_BLOCKS, 0)
    return dict(d=d, log=log, schema=schema,
                X=d[schema.features].to_numpy(float),
                y=d[schema.target].to_numpy(float),
                ids=ids, coords=coords, groups=groups, diag=diag,
                log_target=bool(np.all(d[schema.target].to_numpy(float) > 0)))


@pytest.fixture(scope="session")
def dup_pipeline_inputs(schema, cleaned_keep_dupes):
    """Same, but with replicate rows kept: one site then occupies several rows."""
    d, log = cleaned_keep_dupes
    ids = d[schema.group_key()].to_numpy()
    coords = d[[schema.lat, schema.lon]].to_numpy(float)
    groups, diag = V.spatial_groups(coords, ids, TEST_BLOCKS, 0)
    return dict(d=d, log=log, schema=schema,
                X=d[schema.features].to_numpy(float),
                y=d[schema.target].to_numpy(float),
                ids=ids, coords=coords, groups=groups, diag=diag)
