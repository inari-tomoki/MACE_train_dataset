#!/usr/bin/env python3
"""
research_manager.py
Safe helper for the MACE + CP2K active-learning workflow.

Main goals
----------
- Inspect environment and repo health.
- Validate known ionic species before they enter training.
- Summarize CP2K successes/failures and detect odd-electron aborts.
- Quarantine invalid known-species structures WITHOUT deleting them.
- Optionally launch a round only after validation.

This script is deliberately conservative:
- `clean` is dry-run by default.
- `run-round` only prints commands unless `--execute` is given.
- Unknown radical multiplicities are never guessed automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------
# Known chemistry configuration
# ---------------------------

# CP2K GTH valence q-values used in your current KIND_PARAMS.
VALENCE_Q = {
    "H": 1,
    "Li": 3,
    "B": 3,
    "C": 4,
    "N": 5,
    "O": 6,
    "F": 7,
    "P": 5,
    "S": 6,
}

DEFAULT_SPECIES = {
    "TFSI": {
        "composition": {"C": 2, "F": 6, "N": 1, "O": 4, "S": 2},
        "charge": -1,
        "multiplicity": 1,
        "uks": False,
        "periodic": False,
    },
    "FSI": {
        "composition": {"F": 2, "N": 1, "O": 4, "S": 2},
        "charge": -1,
        "multiplicity": 1,
        "uks": False,
        "periodic": False,
    },
    "BF4": {
        "composition": {"B": 1, "F": 4},
        "charge": -1,
        "multiplicity": 1,
        "uks": False,
        "periodic": False,
    },
    "PF6": {
        "composition": {"P": 1, "F": 6},
        "charge": -1,
        "multiplicity": 1,
        "uks": False,
        "periodic": False,
    },
    "PYR13": {
        # N-methyl-N-propylpyrrolidinium
        "composition": {"C": 8, "H": 18, "N": 1},
        "charge": +1,
        "multiplicity": 1,
        "uks": False,
        "periodic": False,
    },
    "LI": {
        "composition": {"Li": 1},
        "charge": +1,
        "multiplicity": 1,
        "uks": False,
        "periodic": False,
    },
}


@dataclass
class Issue:
    severity: str  # OK, WARN, ERROR
    message: str


def c(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def ok(msg: str) -> None:
    print(c(f"[OK]   {msg}", "32"))


def warn(msg: str) -> None:
    print(c(f"[WARN] {msg}", "33"))


def err(msg: str) -> None:
    print(c(f"[ERR]  {msg}", "31"))


def info(msg: str) -> None:
    print(c(f"[INFO] {msg}", "36"))


def load_shell_exports(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not path.exists():
        return result
    pattern = re.compile(r'^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(["\']?)(.*?)\2\s*$')
    for line in path.read_text(errors="ignore").splitlines():
        m = pattern.match(line)
        if m:
            key, _, value = m.groups()
            result[key] = os.path.expandvars(value)
    return result


def find_repo(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "src" / "active_pipeline.py").exists() and (p / "configs").exists():
            return p
    return start


def load_species_config(repo: Path) -> Dict[str, dict]:
    cfg = repo / "configs" / "species_config.json"
    if not cfg.exists():
        return DEFAULT_SPECIES
    user = json.loads(cfg.read_text())
    merged = dict(DEFAULT_SPECIES)
    for k, v in user.items():
        merged[k.upper()] = v
    return merged


def normalize_comp(comp: Dict[str, int]) -> Dict[str, int]:
    return {k: int(v) for k, v in comp.items() if int(v) != 0}


def composition_of_atoms(atoms) -> Dict[str, int]:
    return dict(Counter(atoms.get_chemical_symbols()))


def expected_atoms(spec: dict) -> int:
    return sum(int(v) for v in spec["composition"].values())


def valence_electrons(comp: Dict[str, int], charge: int) -> Optional[int]:
    total = 0
    for el, n in comp.items():
        if el not in VALENCE_Q:
            return None
        total += VALENCE_Q[el] * int(n)
    # Positive charge removes electrons; negative charge adds electrons.
    return total - int(charge)


def formula_string(comp: Dict[str, int]) -> str:
    # Readable, deterministic formula (not Hill-order critical).
    preferred = ["C", "H", "Li", "B", "N", "O", "F", "P", "S"]
    pieces = []
    for el in preferred + sorted(set(comp) - set(preferred)):
        n = comp.get(el, 0)
        if n:
            pieces.append(el if n == 1 else f"{el}{n}")
    return "".join(pieces)


def import_ase():
    try:
        from ase.io import read, write
        return read, write
    except Exception as e:
        err(f"ASE import failed: {e}")
        print("Activate your mace_env first.")
        raise SystemExit(2)


def read_configs_csv(repo: Path) -> List[dict]:
    path = repo / "configs" / "configs.csv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def detect_species(name: str, info_dict: Optional[dict] = None) -> Optional[str]:
    hay = name.upper()
    if info_dict:
        st = str(info_dict.get("system_type", "")).upper()
        hay += " " + st
    # Long/specific names first.
    for sp in ["PYR13", "TFSI", "FSI", "BF4", "PF6", "LI"]:
        if sp in hay:
            return sp
    return None


def cmd_status(repo: Path, species_cfg: Dict[str, dict]) -> int:
    print(f"Repo: {repo}")
    print()

    # Python / env
    ok(f"Python: {sys.executable}")
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    if env_name:
        ok(f"Conda env: {env_name}")
    else:
        warn("CONDA_DEFAULT_ENV is not set")

    # CUDA
    try:
        import torch
        if torch.cuda.is_available():
            ok(f"CUDA: {torch.cuda.get_device_name(0)}")
        else:
            warn("CUDA is not available to PyTorch")
    except Exception as e:
        warn(f"PyTorch/CUDA check failed: {e}")

    # CP2K
    cp2k = shutil.which("cp2k.ssmp")
    if cp2k:
        ok(f"CP2K: {cp2k}")
    else:
        err("cp2k.ssmp not found in PATH")

    # Local config / model
    exports = load_shell_exports(repo / "config.local.sh")
    model = os.environ.get("MACE_FOUNDATION_MODEL") or exports.get("MACE_FOUNDATION_MODEL")
    if model:
        model_path = Path(os.path.expanduser(model))
        if model_path.exists():
            ok(f"MACE model exists: {model_path}")
        else:
            err(f"MACE model path does not exist: {model_path}")
    else:
        warn("MACE_FOUNDATION_MODEL not found in environment/config.local.sh")

    cp2k_lib = os.environ.get("CP2K_LIBDIR") or exports.get("CP2K_LIBDIR")
    if cp2k_lib:
        p = Path(os.path.expanduser(cp2k_lib))
        if (p / "BASIS_MOLOPT").exists() and (p / "GTH_POTENTIALS").exists():
            ok(f"CP2K data dir: {p}")
        else:
            err(f"CP2K data files missing in: {p}")
    else:
        warn("CP2K_LIBDIR not found")

    # Important repo files
    for rel in [
        "configs/constants.py",
        "configs/configs.csv",
        "configs/E0s.json",
        "configs/round_configs/round1_active_pipeline.py",
        "configs/round_configs/round1_neb_geo_run.py",
    ]:
        p = repo / rel
        (ok if p.exists() else err)(f"{rel}: {'present' if p.exists() else 'missing'}")

    # Chemistry summary
    print("\nKnown species:")
    for sp, cfg in species_cfg.items():
        comp = normalize_comp(cfg["composition"])
        ne = valence_electrons(comp, cfg["charge"])
        parity = "even" if ne is not None and ne % 2 == 0 else "odd/unknown"
        print(
            f"  {sp:<6} {formula_string(comp):<14} "
            f"charge={cfg['charge']:+d} mult={cfg['multiplicity']} "
            f"UKS={cfg['uks']} electrons={ne} ({parity})"
        )
    return 0


def validate_known_structure(path: Path, species: str, spec: dict) -> List[Issue]:
    read, _ = import_ase()
    issues: List[Issue] = []
    try:
        atoms = read(path)
    except Exception as e:
        return [Issue("ERROR", f"{path}: cannot read structure ({e})")]

    actual = composition_of_atoms(atoms)
    expected = normalize_comp(spec["composition"])
    if actual == expected:
        issues.append(Issue("OK", f"{path}: {len(atoms)} atoms {formula_string(actual)}"))
    else:
        issues.append(
            Issue(
                "ERROR",
                f"{path}: expected {formula_string(expected)} ({expected_atoms(spec)} atoms), "
                f"found {formula_string(actual)} ({len(atoms)} atoms)",
            )
        )
    return issues


def validate_candidate_files(repo: Path, species_cfg: Dict[str, dict]) -> List[Issue]:
    read, _ = import_ase()
    issues: List[Issue] = []
    root = repo / "geo_opt_results" / "al_candidates"
    if not root.exists():
        issues.append(Issue("WARN", f"{root} does not exist yet"))
        return issues

    for path in sorted(root.glob("*.extxyz")):
        try:
            frames = read(path, ":")
        except Exception as e:
            issues.append(Issue("ERROR", f"{path}: cannot read ({e})"))
            continue
        for i, atoms in enumerate(frames):
            sp = detect_species(path.name, atoms.info)
            if sp and sp in species_cfg:
                actual = composition_of_atoms(atoms)
                expected = normalize_comp(species_cfg[sp]["composition"])
                if actual != expected:
                    issues.append(
                        Issue(
                            "ERROR",
                            f"{path.name} frame {i}: {sp} expected "
                            f"{formula_string(expected)}, found {formula_string(actual)}",
                        )
                    )
                else:
                    issues.append(
                        Issue(
                            "OK",
                            f"{path.name} frame {i}: {sp} {formula_string(actual)} ({len(atoms)} atoms)",
                        )
                    )
    return issues


def validate_e0(repo: Path, configured_elements: Iterable[str]) -> List[Issue]:
    issues: List[Issue] = []
    path = repo / "configs" / "E0s.json"
    if not path.exists():
        return [Issue("ERROR", "configs/E0s.json is missing")]
    try:
        e0 = json.loads(path.read_text())
    except Exception as e:
        return [Issue("ERROR", f"E0s.json cannot be parsed: {e}")]

    atomic_numbers = {
        "H": 1, "Li": 3, "B": 5, "C": 6, "N": 7,
        "O": 8, "F": 9, "P": 15, "S": 16,
    }
    for el in sorted(set(configured_elements)):
        z = atomic_numbers.get(el)
        if z is None:
            issues.append(Issue("WARN", f"No atomic-number mapping for {el}"))
        elif str(z) not in e0:
            issues.append(Issue("ERROR", f"E0 missing for {el} (Z={z})"))
    if not any(i.severity == "ERROR" for i in issues):
        issues.append(Issue("OK", f"E0s.json contains all elements required by configs.csv"))
    return issues


def collect_configured_elements(repo: Path) -> List[str]:
    read, _ = import_ase()
    els: List[str] = []
    for row in read_configs_csv(repo):
        for key in ("initial", "final"):
            raw = (row.get(key) or "").strip()
            if not raw:
                continue
            p = (repo / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
            if not p.exists():
                continue
            try:
                atoms = read(p)
                els.extend(atoms.get_chemical_symbols())
            except Exception:
                pass
    return els


def cmd_validate(repo: Path, species_cfg: Dict[str, dict]) -> int:
    issues: List[Issue] = []

    rows = read_configs_csv(repo)
    if not rows:
        issues.append(Issue("ERROR", "configs/configs.csv is missing or empty"))
    else:
        for row in rows:
            name = (row.get("Name") or row.get("name") or "").strip()
            sp = detect_species(name)
            for key in ("initial", "final"):
                raw = (row.get(key) or "").strip()
                if not raw:
                    continue
                p = (repo / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
                if not p.exists():
                    issues.append(Issue("ERROR", f"{name} {key}: file not found: {p}"))
                    continue
                if sp and sp in species_cfg:
                    issues.extend(validate_known_structure(p, sp, species_cfg[sp]))
                else:
                    issues.append(Issue("WARN", f"{name}: unknown species rule; only existence checked ({p})"))

    issues.extend(validate_candidate_files(repo, species_cfg))
    issues.extend(validate_e0(repo, collect_configured_elements(repo)))

    print("Validation report")
    print("=" * 72)
    for i in issues:
        {"OK": ok, "WARN": warn, "ERROR": err}[i.severity](i.message)

    nerr = sum(i.severity == "ERROR" for i in issues)
    nwarn = sum(i.severity == "WARN" for i in issues)
    print("=" * 72)
    print(f"Errors: {nerr} | Warnings: {nwarn}")
    return 1 if nerr else 0


def cp2k_out_status(path: Path) -> Tuple[str, str]:
    if not path.exists():
        return "MISSING", "no output"
    text = path.read_text(errors="ignore")
    if "PROGRAM ENDED AT" in text:
        return "OK", "normal termination"
    if "Use the LSD option for an odd number of electrons" in text:
        return "FAILED_ODD_ELECTRON", "odd electron count; UKS/LSD required"
    if "[ABORT]" in text or "ABORT" in text:
        # Extract last useful abort line if possible.
        lines = [ln.strip() for ln in text.splitlines() if "ABORT" in ln or "ERROR" in ln]
        return "FAILED", lines[-1] if lines else "CP2K abort"
    if text.strip():
        return "INCOMPLETE", "output exists but no normal termination marker"
    return "EMPTY", "empty output"


def parse_cp2k_input_composition(path: Path) -> Dict[str, int]:
    comp: Counter = Counter()
    text = path.read_text(errors="ignore").splitlines()
    in_coord = False
    for line in text:
        s = line.strip()
        if s.upper().startswith("&COORD"):
            in_coord = True
            continue
        if in_coord and s.upper().startswith("&END COORD"):
            break
        if in_coord and s and not s.startswith("#"):
            parts = s.split()
            if parts and re.fullmatch(r"[A-Z][a-z]?", parts[0]):
                comp[parts[0]] += 1
    return dict(comp)


def cp2k_input_flags(path: Path) -> Dict[str, Optional[str]]:
    text = path.read_text(errors="ignore")
    def find(pattern: str) -> Optional[str]:
        m = re.search(pattern, text, flags=re.I | re.M)
        return m.group(1) if m else None
    return {
        "charge": find(r"^\s*CHARGE\s+([+-]?\d+)"),
        "multiplicity": find(r"^\s*MULTIPLICITY\s+(\d+)"),
        "uks": "TRUE" if re.search(r"^\s*(?:UKS|LSD)\s+(?:T|TRUE|\.TRUE\.)", text, flags=re.I | re.M) else "FALSE",
        "periodic": find(r"^\s*PERIODIC\s+(\w+)"),
    }


def cmd_cp2k_status(repo: Path, species_cfg: Dict[str, dict], round_num: Optional[int]) -> int:
    dirs = []
    if round_num is not None:
        d = repo / f"cp2k_sp_round{round_num}"
        if d.exists():
            dirs = [d]
    else:
        dirs = sorted(repo.glob("cp2k_sp_round*"))

    if not dirs:
        warn("No cp2k_sp_round* directories found")
        return 1

    total = ok_n = fail_n = 0
    for d in dirs:
        print(f"\n{d.name}")
        print("-" * 72)
        for inp in sorted(d.glob("*.inp")):
            total += 1
            out = inp.with_suffix(".out")
            status, reason = cp2k_out_status(out)
            comp = parse_cp2k_input_composition(inp)
            flags = cp2k_input_flags(inp)
            sp = detect_species(inp.name)
            extra = ""
            if sp and sp in species_cfg:
                expected = normalize_comp(species_cfg[sp]["composition"])
                if comp != expected:
                    extra = f" | BAD_COMPOSITION expected={formula_string(expected)} got={formula_string(comp)}"
            if status == "OK":
                ok_n += 1
                ok(f"{inp.stem}: {status} | {formula_string(comp)}{extra}")
            else:
                fail_n += 1
                warn(
                    f"{inp.stem}: {status} | {formula_string(comp)} | {reason} "
                    f"| charge={flags['charge']} mult={flags['multiplicity']} UKS={flags['uks']}{extra}"
                )

    print("\nSummary")
    print("=" * 72)
    print(f"Total: {total} | OK: {ok_n} | non-OK: {fail_n}")
    return 1 if fail_n else 0


def make_backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(path.name + f".bak_{stamp}")
    shutil.copy2(path, backup)
    return backup


def cmd_clean(repo: Path, species_cfg: Dict[str, dict], apply: bool) -> int:
    read, write = import_ase()
    candidates_root = repo / "geo_opt_results" / "al_candidates"
    bad_found = 0

    print("Candidate cleanup")
    print("=" * 72)

    if candidates_root.exists():
        for path in sorted(candidates_root.glob("*.extxyz")):
            frames = read(path, ":")
            good = []
            bad = []
            for idx, atoms in enumerate(frames):
                sp = detect_species(path.name, atoms.info)
                if sp and sp in species_cfg:
                    actual = composition_of_atoms(atoms)
                    expected = normalize_comp(species_cfg[sp]["composition"])
                    if actual != expected:
                        bad.append((idx, atoms, sp, actual, expected))
                        continue
                good.append(atoms)

            if bad:
                bad_found += len(bad)
                for idx, _, sp, actual, expected in bad:
                    warn(
                        f"{path.name} frame {idx}: quarantine candidate "
                        f"{sp} {formula_string(actual)} (expected {formula_string(expected)})"
                    )
                if apply:
                    backup = make_backup(path)
                    write(path, good)
                    qdir = candidates_root / "quarantine"
                    qdir.mkdir(exist_ok=True)
                    for idx, atoms, sp, _, _ in bad:
                        qfile = qdir / f"{path.stem}_frame{idx}_{sp}.extxyz"
                        write(qfile, atoms)
                    ok(f"Rewrote {path.name}; backup: {backup.name}")

    # CP2K known-species composition quarantine.
    for d in sorted(repo.glob("cp2k_sp_round*")):
        qdir = d / "quarantine_bad_composition"
        for inp in sorted(d.glob("*.inp")):
            sp = detect_species(inp.name)
            if not sp or sp not in species_cfg:
                continue
            comp = parse_cp2k_input_composition(inp)
            expected = normalize_comp(species_cfg[sp]["composition"])
            if comp and comp != expected:
                bad_found += 1
                warn(
                    f"{d.name}/{inp.stem}: quarantine CP2K job "
                    f"{formula_string(comp)} (expected {formula_string(expected)})"
                )
                if apply:
                    qdir.mkdir(exist_ok=True)
                    stem = inp.stem
                    for f in d.glob(stem + ".*"):
                        if f.is_file():
                            shutil.move(str(f), str(qdir / f.name))
                    ok(f"Moved {stem} files to {qdir.relative_to(repo)}")

    if not apply:
        print("\nDRY RUN only. Nothing was changed.")
        print("Run with `clean --apply` only after reviewing the list.")
    else:
        print("\nCleanup applied. Nothing was deleted; invalid data was backed up/quarantined.")

    if bad_found == 0:
        ok("No known invalid structures found")
    else:
        warn(f"Found {bad_found} invalid item(s)")
    return 0


def resolve_model(repo: Path, explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    exports = load_shell_exports(repo / "config.local.sh")
    return os.environ.get("MACE_FOUNDATION_MODEL") or exports.get("MACE_FOUNDATION_MODEL")


def cmd_run_round(
    repo: Path,
    species_cfg: Dict[str, dict],
    round_num: int,
    runs: int,
    model: Optional[str],
    execute: bool,
) -> int:
    # Hard gate: known-structure validation must pass.
    rc = cmd_validate(repo, species_cfg)
    if rc != 0:
        err("Validation failed. Round launch blocked.")
        return rc

    model = resolve_model(repo, model)
    if not model:
        err("No model supplied and MACE_FOUNDATION_MODEL could not be resolved.")
        return 2

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")

    geo_cmd = [
        sys.executable,
        str(repo / "src" / "neb_geo_run.py"),
        "--round", str(round_num),
        "--model", model,
        "--skip-neb",
        "--skip-aimd",
        "--skip-plumed",
    ]
    al_cmd = [
        sys.executable,
        str(repo / "src" / "active_pipeline.py"),
        "--runs", str(runs),
        "--model", model,
        "--target", str(round_num),
    ]

    print("\nPlanned commands:")
    print("  " + " ".join(geo_cmd))
    print("  " + " ".join(al_cmd))
    print("\nNOTE: This does not launch CP2K submit scripts automatically.")

    if not execute:
        warn("DRY RUN. Add --execute to actually run MACE GeoOpt + candidate generation.")
        return 0

    info("Running geometry optimization stage...")
    subprocess.run(geo_cmd, cwd=repo, env=env, check=True)
    info("Running active-learning candidate generation stage...")
    subprocess.run(al_cmd, cwd=repo, env=env, check=True)
    ok("Round preparation finished. Review `cp2k-status` before running CP2K.")
    return 0


def cmd_init_config(repo: Path) -> int:
    out = repo / "configs" / "species_config.json"
    if out.exists():
        warn(f"{out} already exists; not overwriting")
        return 1
    out.write_text(json.dumps(DEFAULT_SPECIES, indent=2))
    ok(f"Created {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Safe manager for MACE + CP2K active-learning research workflow."
    )
    p.add_argument(
        "--repo",
        default=None,
        help="Path to mace_active_learning repo (default: auto-detect from cwd).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Check environment, model, CP2K and core files.")
    sub.add_parser("validate", help="Validate configs.csv, known species, candidates and E0 coverage.")

    s = sub.add_parser("cp2k-status", help="Summarize CP2K .inp/.out success and common failures.")
    s.add_argument("--round", type=int, default=None)

    s = sub.add_parser("clean", help="Find invalid known-species frames/jobs and quarantine them.")
    s.add_argument("--apply", action="store_true", help="Actually quarantine/rewrite. Default is dry-run.")

    s = sub.add_parser("run-round", help="Prepare a round; dry-run unless --execute.")
    s.add_argument("round", type=int)
    s.add_argument("--runs", type=int, default=50)
    s.add_argument("--model", default=None)
    s.add_argument("--execute", action="store_true")

    sub.add_parser("init-config", help="Create configs/species_config.json with known ion settings.")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    start = Path(args.repo).expanduser() if args.repo else Path.cwd()
    repo = find_repo(start)
    species_cfg = load_species_config(repo)

    if args.command == "status":
        return cmd_status(repo, species_cfg)
    if args.command == "validate":
        return cmd_validate(repo, species_cfg)
    if args.command == "cp2k-status":
        return cmd_cp2k_status(repo, species_cfg, args.round)
    if args.command == "clean":
        return cmd_clean(repo, species_cfg, args.apply)
    if args.command == "run-round":
        return cmd_run_round(repo, species_cfg, args.round, args.runs, args.model, args.execute)
    if args.command == "init-config":
        return cmd_init_config(repo)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
