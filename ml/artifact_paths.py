"""
Where the large ML artifacts live now that they are out of Git.

Trained models and bulky generated outputs were moved, byte-for-byte, into the
Git-ignored data/ directory. ml/artifact_relocation_manifest.json records every
move (old path -> new path, SHA-256 before and after). The old paths remain the
paths recorded in historical results and in the frozen test manifests; they are
provenance, not current locations.

Final Axis 2 V3 model (50-tree RF + natural-offset HGB + dynamic gate):

    default    <repo>/data/models/axis2_v3/
    override   AXIS2_MODEL_DIR=/path/to/dir   (must contain the same three files)

final_model_paths() returns the three paths and, by default, checks each file's
SHA-256 against the relocation manifest first. Do that before loading: .joblib
(and torch .pt without weights_only=True) deserialize via pickle, which executes
code, so only load files whose hash matches.

Nothing in backend/ uses this module; production still scores Axis 2 with the
Phase A curve. It exists so a future integration has one place to find models.

Historical experiment scripts are left byte-identical, because the frozen test
manifests record their SHA-256 hashes. They read artifacts from the old
in-tree paths, so to re-run one, first copy its inputs back with `restore`
(hash-verified; restored copies are Git-ignored):

    python ml/artifact_paths.py verify
    python ml/artifact_paths.py where ml/experiments/axis2_v3_random_forest/random_forest_model.joblib
    python ml/artifact_paths.py restore ml/experiments/axis2_v3_random_forest/
    python ml/artifact_paths.py restore --all --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "ml" / "artifact_relocation_manifest.json"
DEFAULT_MODEL_DIR = REPO / "data" / "models" / "axis2_v3"
MODEL_DIR_ENV = "AXIS2_MODEL_DIR"

# Role -> filename inside the model directory. Roles match the keys used in
# ml/experiments/axis2_v3_final_test/final_model_manifest_*_test.json.
FINAL_MODEL_FILES = {
    "rf_model": "selected_rf_model.joblib",
    "natural_model": "natural_offset_model.joblib",
    "dynamic_gate": "selected_dynamic_gate.pt",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> Dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _entries() -> List[Dict[str, Any]]:
    return load_manifest()["artifacts"]


def model_dir() -> Path:
    """The final-model directory: $AXIS2_MODEL_DIR if set, else data/models/axis2_v3."""
    override = os.environ.get(MODEL_DIR_ENV)
    return Path(override).expanduser().resolve() if override else DEFAULT_MODEL_DIR


def _current_location(entry: Dict[str, Any]) -> Path:
    # Final models follow AXIS2_MODEL_DIR; everything else lives at its recorded new_path.
    if entry["category"] == "final_model":
        return model_dir() / Path(entry["new_path"]).name
    return REPO / entry["new_path"]


def final_model_paths(verify: bool = True) -> Dict[str, Path]:
    """
    Paths to the three frozen Axis 2 V3 model files.

    Raises FileNotFoundError if any is missing, and ValueError if verify=True
    and any file's SHA-256 differs from the relocation manifest.
    """
    expected = {Path(e["new_path"]).name: e["relocated_sha256"]
                for e in _entries() if e["category"] == "final_model"}
    d = model_dir()
    paths = {role: d / name for role, name in FINAL_MODEL_FILES.items()}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Axis 2 model files not found in {d}: {missing}. The models are not in Git; "
            f"provision them into data/models/axis2_v3/ or point {MODEL_DIR_ENV} at a directory holding them.")
    if verify:
        bad = [str(p) for p in paths.values() if sha256(p) != expected[p.name]]
        if bad:
            raise ValueError(f"SHA-256 mismatch for {bad}; refusing to treat them as the frozen model.")
    return paths


def resolve(old_path: str) -> Path:
    """
    Current location of an artifact given its original repo-relative path.
    Prefers the in-tree copy if one has been restored there.
    """
    old_path = Path(old_path).as_posix()  # normalises a leading "./"
    if (REPO / old_path).is_file():
        return REPO / old_path
    for e in _entries():
        if e["old_path"] == old_path:
            p = _current_location(e)
            if p.is_file():
                return p
            raise FileNotFoundError(f"{old_path} was relocated to {p}, which does not exist")
    raise FileNotFoundError(f"{old_path} is not in the tree and not in {MANIFEST_PATH.name}")


def _cmd_verify(_: argparse.Namespace) -> int:
    bad = 0
    for e in _entries():
        p = _current_location(e)
        if not p.is_file():
            status, bad = "MISSING", bad + 1
        elif sha256(p) != e["relocated_sha256"]:
            status, bad = "MISMATCH", bad + 1
        else:
            status = "ok"
        print(f"{status:8} {p.relative_to(REPO) if p.is_relative_to(REPO) else p}")
    print(f"{len(_entries()) - bad}/{len(_entries())} verified")
    return 1 if bad else 0


def _cmd_where(args: argparse.Namespace) -> int:
    print(resolve(args.old_path))
    return 0


def _cmd_restore(args: argparse.Namespace) -> int:
    prefixes = [Path(p).as_posix() + ("/" if p.endswith("/") else "") for p in args.prefixes]
    if not prefixes and not args.all:
        print("give one or more old-path prefixes, or --all", file=sys.stderr)
        return 2
    chosen = [e for e in _entries() if args.all or any(e["old_path"].startswith(p) for p in prefixes)]
    if not chosen:
        print("no relocated artifacts match", file=sys.stderr)
        return 1
    for e in chosen:
        src, dst = _current_location(e), REPO / e["old_path"]
        if dst.exists():
            if sha256(dst) != e["original_sha256"]:
                print(f"REFUSING  {e['old_path']} exists with different contents", file=sys.stderr)
                return 1
            print(f"present  {e['old_path']}")
            continue
        if sha256(src) != e["relocated_sha256"]:
            print(f"REFUSING  source {src} fails its hash check", file=sys.stderr)
            return 1
        if args.dry_run:
            print(f"would copy {src} -> {e['old_path']}")
            continue
        shutil.copy2(src, dst)
        if sha256(dst) != e["original_sha256"]:
            dst.unlink()
            print(f"FAILED   {e['old_path']} copy did not verify; removed", file=sys.stderr)
            return 1
        print(f"restored {e['old_path']}")
    return 0


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify", help="hash-check every relocated artifact at its current location")
    w = sub.add_parser("where", help="print the current location of an artifact by its original path")
    w.add_argument("old_path")
    r = sub.add_parser("restore", help="copy archived artifacts back to their original in-tree paths")
    r.add_argument("prefixes", nargs="*")
    r.add_argument("--all", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    return {"verify": _cmd_verify, "where": _cmd_where, "restore": _cmd_restore}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
