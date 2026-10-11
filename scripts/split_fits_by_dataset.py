#!/usr/bin/env python
"""Split a Wrap fit directory into one directory per dataset, at base resolution.

canonicalise_corpus.py takes one directory per dataset and needs the TEMPLATE
topology, but the ladder writes every ear into one directory at Wrap's
subdivided resolution. The first len(template) vertices are the base mesh --
subdivision appends, it does not reorder -- so the split is a truncation plus
the template's own faces.
"""
import argparse, pathlib, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ear3d.wrap import load_mesh                                   # noqa: E402

WRAP = pathlib.Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/wrap_work")


def dataset_of(name):
    """HUTUBS ears are ppNN_side; SONICOM are PNNNN_side."""
    return "hutubs" if name.startswith("pp") else "sonicom"


def one(job):
    src, dsts, nb, F = job
    p = pathlib.Path(src)
    V, _ = load_mesh(p)
    if len(V) < nb:
        return (p.stem, f"only {len(V)} vertices, expected >= {nb}")
    dst = pathlib.Path(dsts[dataset_of(p.stem)]) / p.name
    with open(dst, "w") as fh:
        for q in V[:nb]:
            fh.write(f"v {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}\n")
        for t in F:
            fh.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
    return (p.stem, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out-prefix", required=True,
                   help="writes <prefix>_hutubs and <prefix>_sonicom")
    p.add_argument("--template", default=str(WRAP / "templates/ear_L_template.obj"))
    p.add_argument("--jobs", type=int, default=12)
    a = p.parse_args()

    Vt, Ft = load_mesh(a.template)
    dsts = {d: pathlib.Path(f"{a.out_prefix}_{d}") for d in ("hutubs", "sonicom")}
    for d in dsts.values():
        d.mkdir(parents=True, exist_ok=True)
    files = sorted(pathlib.Path(a.src).glob("*.obj"))
    jobs = [(str(f), {k: str(v) for k, v in dsts.items()}, len(Vt), Ft)
            for f in files]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(one, jobs))
    bad = [(n, e) for n, e in res if e]
    for d, path in dsts.items():
        print(f"  {d:8s} {len(list(path.glob('*.obj'))):4d} -> {path}")
    if bad:
        print(f"  FAILED {len(bad)}: {bad[:5]}")
        return 1
    print(f"  split {len(res)} fits to base resolution ({len(Vt)}v {len(Ft)}f)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
