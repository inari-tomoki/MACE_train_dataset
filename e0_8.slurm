#!/bin/bash
#SBATCH --job-name=e0_atoms
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=e0_atoms_%j.log

set -euo pipefail

cd "$HOME/work/mace_active_learning"

CP2K="$HOME/.micromamba/envs/cp2k_hydrangea/bin/cp2k.ssmp"
INPUT_DIR="cp2k_e0_consistency_drafts"
OUTPUT_DIR="cp2k_e0_consistency_results"

export OMP_NUM_THREADS=4
mkdir -p "$OUTPUT_DIR"

for element in H Li C N O F P S; do
    name="e0_${element}_consistency"
    inp="$INPUT_DIR/${name}.inp"
    out="$OUTPUT_DIR/${name}.out"

    test -s "$inp"

    echo "START: $element $(date)"

    srun "$CP2K" -i "$inp" -o "$out"

    if ! grep -q "PROGRAM ENDED AT" "$out"; then
        echo "ERROR: $element"
        exit 1
    fi

    echo "SUCCESS: $element $(date)"
done

echo "ALL 8 E0 CALCULATIONS COMPLETED"
