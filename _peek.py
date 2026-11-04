"""Are the 'different' partitions actually different, or only relabelled?"""
import sys

import numpy as np

import synthetic_afsis as S
import afsis_data as A
import afsis_validation as V
from pathlib import Path

# --- the reported run's data -------------------------------------------------
df, _ = A.load_raw(Path("data/afsis_stand_in"))
schema = A.detect_schema(df)
d, log = A.clean(df, schema, dict(depth=A.DEPTH_TARGET_CM, target_units="auto",
                                 lab_dupes="first"))
ids = d[schema.group_key()].to_numpy()
coords = d[[schema.lat, schema.lon]].to_numpy(float)


def canon(groups):
    """Partition as a set of sets - invariant to how the clusters are numbered."""
    out = []
    for g in np.unique(groups):
        out.append(frozenset(np.flatnonzero(groups == g).tolist()))
    return frozenset(out)


for blocks, seeds in ((10, [20261104, 20261105, 20261106, 20261107, 20261108]),
                      (6, [0, 1, 987654321])):
    gs = [V.spatial_groups(coords, ids, blocks, s)[0] for s in seeds]
    cs = [canon(g) for g in gs]
    raw = len({tuple(map(int, g)) for g in gs})
    print(f"blocks={blocks} seeds={len(seeds)} raw-label-distinct={raw} "
          f"canonically-distinct={len(set(cs))}")
    print("   same sizes?", [sorted(len(c) for c in x)[:3] for x in cs])

# --- the small test fixture --------------------------------------------------
tmp = Path("_check_fixture")
S.write_dataset(tmp, seed=4242, n_sites=160, n_clusters=8)
df2, _ = A.load_raw(tmp)
sc2 = A.detect_schema(df2)
d2, _ = A.clean(df2, sc2, dict(depth=A.DEPTH_TARGET_CM, target_units="auto",
                               lab_dupes="first"))
id2, co2 = d2[sc2.group_key()].to_numpy(), d2[[sc2.lat, sc2.lon]].to_numpy(float)
g0, _ = V.spatial_groups(co2, id2, 6, 0)
g1, _ = V.spatial_groups(co2, id2, 6, 987654321)
print("fixture: raw-differ", not np.array_equal(g0, g1),
      "| canonically-differ", canon(g0) != canon(g1))
