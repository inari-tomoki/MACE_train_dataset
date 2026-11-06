"""
constants.py
Chemistry and infrastructure facts that do NOT change between active-learning
rounds. If you're tuning a value between Round 5 and Round 6, it does not
belong here — put it in the relevant round_configs/roundN_config.py instead.

This file should rarely change. When it does, it's because you added a new
element to the system or moved to a new cluster (new LIBDIR path).
"""

import os

# --- CP2K basis/pseudopotential library location ---
# Must be set via config.local.sh (sourced before running any pipeline script)
LIBDIR ="/home/inaritomoki/anaconda3/envs/mace_env/share/cp2k/data/"
if not LIBDIR:
    raise ValueError(
        "CP2K_LIBDIR environment variable is not set! "
        "Did you source config.local.sh?"
    )

# MACE MODEL PATHS
FOUNDATION_MODEL_PATH="/home/inaritomoki/work/MACE_TRAIN/mace_active_learning/mace-mp-0b3-medium.model"
FINETUNED_MODEL_PATH="/home/user/Documents/Programs/Python/ASE/MACE/active_learning/mace_V5_active_learning_stagetwo.model"

# --- Atomic number mapping ---
Z_MAP = {"H": 1, "Li": 3,"B":5, "C": 6, "N": 7,"O": 8, "F": 9, "P": 15, "S": 16}

# --- Elements treated as metals (affects e.g. slab-height triage logic) ---
METALS = {}  # extend if other transition metals are added later

# --- Fragments of molecules that have specific charges ---
IONIC_SPECIES_CHARGE = {
    "BF4": -1, "TFSI": -1, "FSI": -1, "PF6": -1,
    "PYR13": +1,
    
}
# --- Location of training files ---
CLEAN_TRAIN  = "training_clean.xyz"
BAD_TRAIN    = "training_bad.xyz"
MASTER_TRAIN = "master_train_pool.xyz"
HELD_OUT     = "held_out.xyz"

# --- CP2K basis set / pseudopotential per element: (basis, potential) ---
KIND_PARAMS = {
    "H":  ("DZVP-MOLOPT-SR-GTH-q1",  "GTH-PBE-q1"),
    "C":  ("DZVP-MOLOPT-SR-GTH-q4",  "GTH-PBE-q4"),
    "O":  ("DZVP-MOLOPT-SR-GTH-q6",  "GTH-PBE-q6"),
    "F":  ("DZVP-MOLOPT-SR-GTH-q7",  "GTH-PBE-q7"),
    "S":  ("DZVP-MOLOPT-SR-GTH-q6",  "GTH-PBE-q6"),
    "Pt": ("DZVP-MOLOPT-SR-GTH-q18", "GTH-PBE-q18"),
    "B":  ("DZVP-MOLOPT-SR-GTH-q3",  "GTH-PBE-q3"),
    "P":  ("DZVP-MOLOPT-SR-GTH-q5",  "GTH-PBE-q5"),
    "Li": ("DZVP-MOLOPT-SR-GTH-q3", "GTH-PBE-q3"),
    "N":  ("DZVP-MOLOPT-SR-GTH-q5", "GTH-PBE-q5"),

}

# --- Default simulation cell (a, b, c) in Angstrom, keyed by substring
# matched against the structure/system name ---
DEFAULT_CELLS = {
    "default":            (20.0, 20.0,  20.0)  

}

# --- E0 reference data file (isolated-atom energies, produced once per
# round via the E0 calculation step, but the filename convention is stable) ---
E0_JSON = "/home/inaritomoki/work/MACE_TRAIN/mace_active_learning/configs/E0s.json"
E0_CELL_SIZE = 20.0  # Angstrom, cubic box for isolated-atom E0 calculations

# --- External training-set filename tags that should never be pruned
# from the training set during cleaning, regardless of other rules ---
EXTERNAL_SYSTEM_TYPES = ("mptrj", "oc25", "reico")
