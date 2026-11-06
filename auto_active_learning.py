#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import hashlib
import os
import random
import re
import shutil
import subprocess
import sys

from ase.io import read, write


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

MASTER = ROOT / "master_train_pool.xyz"
CLEAN = ROOT / "training_clean.xyz"
BAD = ROOT / "training_bad.xyz"

HELDOUT = ROOT / "held_out_fixed.xyz"
TRAIN_FIT = ROOT / "training_fit.xyz"

FOUNDATION = ROOT / "mace-mp-0b3-medium.model"


def base_env():
    env = os.environ.copy()
    old = env.get("PYTHONPATH", "")
    paths = [str(ROOT), str(ROOT / "src")]
    if old:
        paths.append(old)
    env["PYTHONPATH"] = ":".join(paths)
    return env


def run(cmd, log=None, env=None):
    print("\n" + "=" * 72)
    print("[RUN]", " ".join(map(str, cmd)))
    print("=" * 72)

    if env is None:
        env = base_env()

    if log:
        log_path = ROOT / log
        with log_path.open("w") as fh:
            p = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in p.stdout:
                print(line, end="")
                fh.write(line)

            rc = p.wait()

        if rc != 0:
            raise RuntimeError(
                f"Command failed ({rc}): {' '.join(map(str, cmd))}"
            )
    else:
        subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            check=True,
        )


def nframes(path):
    if not path.exists():
        return 0

    frames = read(str(path), ":")
    return len(frames)


def geometry_hash(a):
    h = hashlib.sha256()
    h.update(a.get_atomic_numbers().tobytes())
    h.update(a.get_positions().round(4).tobytes())
    return h.hexdigest()


def latest_round_config_before(round_number):
    candidates = []

    for p in (ROOT / "configs/round_configs").glob(
        "round*_active_pipeline.py"
    ):
        m = re.match(r"round(\d+)_active_pipeline\.py", p.name)
        if m:
            r = int(m.group(1))
            if r < round_number:
                candidates.append((r, p))

    if not candidates:
        raise RuntimeError("No previous active-pipeline config found.")

    return max(candidates)[1]


def make_round_config(round_number, runs, reico_num):
    dst = (
        ROOT
        / "configs/round_configs"
        / f"round{round_number}_active_pipeline.py"
    )

    if dst.exists():
        print(f"[OK] Round config already exists: {dst}")
        return

    src = latest_round_config_before(round_number)
    text = src.read_text()

    text = re.sub(
        r"round:\s*int\s*=\s*\d+",
        f"round: int = {round_number}",
        text,
        count=1,
    )

    text = re.sub(
        r"n_select_total:\s*int\s*=\s*\d+",
        f"n_select_total: int = {runs}",
        text,
        count=1,
    )

    reico_enabled = reico_num > 0

    text = re.sub(
        r"reico_sampling:\s*bool\s*=\s*(True|False)",
        f"reico_sampling: bool = {reico_enabled}",
        text,
        count=1,
    )

    text = re.sub(
        r"reico_num:\s*int\s*=\s*\d+",
        f"reico_num: int = {reico_num}",
        text,
        count=1,
    )

    dst.write_text(text)

    print(f"[+] Created {dst} from {src.name}")


def archive_old_candidates(round_number):
    src = ROOT / "geo_opt_results/al_candidates"

    if not src.exists():
        return

    files = list(src.glob("*.extxyz"))

    if not files:
        return

    dst = (
        ROOT
        / "geo_opt_results"
        / f"al_candidates_archive_auto_before_round{round_number}"
    )
    dst.mkdir(parents=True, exist_ok=True)

    for f in files:
        target = dst / f.name

        if target.exists():
            target = dst / f"{f.stem}_old{f.suffix}"

        shutil.move(str(f), str(target))

    print(f"[+] Archived {len(files)} old candidate files -> {dst}")



def ensure_bootstrap_candidate():
    """Keep one already-labelled frame so active_pipeline can start REICO sampling."""
    candidate_dir = ROOT / "geo_opt_results/al_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    existing = list(candidate_dir.glob("*.extxyz"))
    if existing:
        print(f"[OK] Candidate directory already has {len(existing)} extxyz file(s)")
        return

    if not CLEAN.exists():
        raise RuntimeError(f"Clean training pool not found: {CLEAN}")

    frames = read(str(CLEAN), ":")
    if not frames:
        raise RuntimeError("Clean training pool is empty")

    # This geometry is already in the training pool, so the later hash check
    # should skip it for CP2K. It only bootstraps active_pipeline so that
    # REICO/new candidates can be generated.
    a = frames[-1].copy()
    a.info["auto_bootstrap"] = True

    out = candidate_dir / "bootstrap_known_pool_frame.extxyz"
    write(str(out), a, format="extxyz")

    print(f"[+] Bootstrap candidate created: {out}")
    print(f"    system_type = {a.info.get('system_type')}")
    print(f"    atoms       = {len(a)}")


def run_active_round(round_number, model, runs, reico_num):
    make_round_config(round_number, runs, reico_num)

    archive_old_candidates(round_number)
    ensure_bootstrap_candidate()

    run(
        [
            PYTHON,
            "src/active_pipeline.py",
            "--target",
            str(round_number),
            "--runs",
            str(runs),
            "--model",
            str(model),
        ],
        log=f"auto_round{round_number}_select.log",
    )

    cp2k_dir = ROOT / f"cp2k_sp_round{round_number}"
    submit = cp2k_dir / "submit_missing.sh"

    if not submit.exists():
        print(
            f"[!] No CP2K submit script for round {round_number}. "
            "Probably no novel structures were selected."
        )
        return False

    inp_files = list(cp2k_dir.glob("*.inp"))

    if not inp_files:
        print(
            f"[!] Round {round_number}: no CP2K inputs. "
            "No novel candidate structures."
        )
        return False

    run(
        ["bash", str(submit)],
        log=f"auto_round{round_number}_cp2k.log",
    )

    run(
        [
            PYTHON,
            "src/active_pipeline.py",
            "--target",
            str(round_number),
            "--parse",
        ],
        log=f"auto_round{round_number}_parse.log",
    )

    run(
        [PYTHON, "src/check_residuals.py"],
        log=f"auto_round{round_number}_residual.log",
    )

    return True


def make_fixed_holdout(seed=42, fraction=0.20):
    if HELDOUT.exists():
        print(f"[OK] Fixed held-out already exists: {HELDOUT}")
        return

    frames = read(str(CLEAN), ":")

    by_system = {}

    for a in frames:
        system = a.info.get("system_type", "unknown")
        by_system.setdefault(system, []).append(a)

    rng = random.Random(seed)

    held = []

    for system, group in sorted(by_system.items()):
        # Never remove the only example of a system.
        if len(group) < 2:
            continue

        k = max(1, round(len(group) * fraction))
        k = min(k, len(group) - 1)

        chosen = rng.sample(group, k)
        held.extend(chosen)

        print(
            f"[held-out] {system}: "
            f"{len(chosen)}/{len(group)} selected"
        )

    if not held:
        raise RuntimeError(
            "Could not make held-out set. "
            "Need repeated examples for at least one system_type."
        )

    write(str(HELDOUT), held, format="extxyz")

    print(f"[+] Fixed held-out created: {len(held)} frames")


def update_training_fit():
    if not HELDOUT.exists():
        shutil.copy2(CLEAN, TRAIN_FIT)
        return

    clean_frames = read(str(CLEAN), ":")
    held_frames = read(str(HELDOUT), ":")

    held_hashes = {geometry_hash(a) for a in held_frames}

    train = [
        a
        for a in clean_frames
        if geometry_hash(a) not in held_hashes
    ]

    write(str(TRAIN_FIT), train, format="extxyz")

    print(
        f"[+] training_fit: {len(train)} frames "
        f"(held-out={len(held_frames)})"
    )


def train_model(round_number, foundation):
    update_training_fit()

    env = base_env()

    env["EVAL_CONFIGS"] = str(HELDOUT)
    env["MACE_TRAINING_PATH"] = str(TRAIN_FIT)
    env["MACE_FOUNDATION_MODEL"] = str(foundation)

    run(
        [
            "bash",
            "train_active_learning.sh",
            "--round",
            str(round_number),
            "--foundation",
            str(foundation),
            "--training",
            str(TRAIN_FIT),
        ],
        log=f"auto_round{round_number}_train.log",
        env=env,
    )

    final_model = (
        ROOT
        / f"mace_V{round_number}_active_learning_final.model"
    )

    if not final_model.exists():
        raise RuntimeError(
            f"Training finished but model not found: {final_model}"
        )

    return final_model


def evaluate_models(round_number, old_model, new_model):
    outdir = ROOT / "comparison_results" / f"round{round_number}"

    run(
        [
            PYTHON,
            "analysis/compare_models.py",
            "--test",
            str(HELDOUT),
            "--models",
            str(old_model),
            str(new_model),
            "--labels",
            "previous",
            f"round{round_number}",
            "--outdir",
            str(outdir),
            "--device",
            "cuda",
        ],
        log=f"auto_round{round_number}_compare.log",
    )

    metrics_file = outdir / "metrics_summary.csv"

    if not metrics_file.exists():
        raise RuntimeError(f"Metrics not found: {metrics_file}")

    with metrics_file.open() as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        raise RuntimeError("Metrics CSV is empty")

    row = rows[-1]

    energy = float(row["E-RMSE(meV/at)"])
    force = float(row["F-RMSE(meV/A)"])

    return energy, force


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start-round", type=int, default=5)
    parser.add_argument("--max-round", type=int, default=10)

    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Maximum CP2K structures selected per round",
    )

    parser.add_argument(
        "--reico-num",
        type=int,
        default=0,
        help="REICO random candidates per round (0 = disabled)",
    )

    parser.add_argument(
        "--seed-target",
        type=int,
        default=30,
        help="Collect this many clean DFT frames before first training",
    )

    parser.add_argument(
        "--force-stop",
        type=float,
        default=150.0,
        help="Held-out force RMSE stop threshold [meV/A]",
    )

    parser.add_argument(
        "--energy-stop",
        type=float,
        default=20.0,
        help="Held-out energy RMSE stop threshold [meV/atom]",
    )

    parser.add_argument(
        "--consecutive",
        type=int,
        default=2,
        help="Required consecutive converged rounds",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    for f in [
        MASTER,
        CLEAN,
        FOUNDATION,
        ROOT / "src/active_pipeline.py",
        ROOT / "src/check_residuals.py",
        ROOT / "train_active_learning.sh",
        ROOT / "analysis/compare_models.py",
    ]:
        if not f.exists():
            raise SystemExit(f"[X] Required file missing: {f}")

    print("=" * 72)
    print("AUTO ACTIVE LEARNING")
    print("=" * 72)
    print("Master frames :", nframes(MASTER))
    print("Clean frames  :", nframes(CLEAN))
    print("Bad frames    :", nframes(BAD))
    print("Start round   :", args.start_round)
    print("Max round     :", args.max_round)
    print("Seed target   :", args.seed_target)
    print("Runs/round    :", args.runs)

    if args.dry_run:
        print("\n[DRY RUN] Pre-flight passed. No calculations started.")
        return

    current_model = FOUNDATION
    converged_count = 0

    # If a previously trained model already exists, use the latest one.
    trained_models = []

    for p in ROOT.glob("mace_V*_active_learning_final.model"):
        m = re.match(
            r"mace_V(\d+)_active_learning_final\.model",
            p.name,
        )
        if m:
            trained_models.append((int(m.group(1)), p))

    if trained_models:
        latest_r, latest_model = max(trained_models)
        current_model = latest_model
        print(
            f"[+] Existing trained model detected: "
            f"Round {latest_r} -> {latest_model.name}"
        )

    for round_number in range(
        args.start_round,
        args.max_round + 1,
    ):
        print("\n" + "#" * 72)
        print(f"# ROUND {round_number}")
        print("#" * 72)

        before = nframes(CLEAN)

        produced = run_active_round(
            round_number,
            current_model,
            args.runs,
            args.reico_num,
        )

        after = nframes(CLEAN)

        print(
            f"\n[Round {round_number}] "
            f"clean pool: {before} -> {after}"
        )

        if not produced or after == before:
            print(
                "[STOP] No new clean DFT frames were produced. "
                "Candidate generation needs more diversity."
            )
            break

        # Seed collection stage
        if after < args.seed_target:
            print(
                f"[SEED] {after}/{args.seed_target} clean frames. "
                "Continue collecting DFT labels before training."
            )
            continue

        # Fixed validation begins only once enough seed data exists.
        make_fixed_holdout()
        update_training_fit()

        new_model = train_model(
            round_number,
            current_model,
        )

        energy_rmse, force_rmse = evaluate_models(
            round_number,
            current_model,
            new_model,
        )

        print("\n" + "=" * 72)
        print(f"ROUND {round_number} VALIDATION")
        print(
            f"Energy RMSE = {energy_rmse:.2f} meV/atom"
        )
        print(
            f"Force RMSE  = {force_rmse:.2f} meV/A"
        )
        print("=" * 72)

        current_model = new_model

        if (
            energy_rmse <= args.energy_stop
            and force_rmse <= args.force_stop
        ):
            converged_count += 1

            print(
                f"[CONVERGENCE] "
                f"{converged_count}/{args.consecutive}"
            )
        else:
            converged_count = 0

        if converged_count >= args.consecutive:
            print("\n" + "=" * 72)
            print("ACTIVE LEARNING CONVERGED")
            print(f"Final model: {current_model}")
            print(
                f"Clean frames: {nframes(CLEAN)}"
            )
            print("=" * 72)
            return

    print("\n" + "=" * 72)
    print("AUTO ACTIVE LEARNING STOPPED")
    print(f"Current model : {current_model}")
    print(f"Clean frames  : {nframes(CLEAN)}")
    print(
        "Reason: convergence not reached, max round reached, "
        "or no novel candidates."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
