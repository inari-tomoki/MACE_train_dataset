"""Audit Round 8 without modifying the production training datasets."""
from pathlib import Path
import csv
import json
import re
import numpy as np
from ase.io import read, write
from ase.data import covalent_radii

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
e0 = {int(k): v for k, v in json.loads((ROOT / 'configs/E0s.json').read_text()).items()}
candidates = read(ROOT / 'al_selected_round8.xyz', ':')
pool = read(ROOT / 'master_train_pool.xyz', ':')
rows, parsed = [], []
ha = 27.211386245988
for inp in sorted((ROOT / 'cp2k_sp_round8').glob('*.inp')):
    index = int(inp.stem.rsplit('_', 1)[1])
    a = candidates[index].copy()
    text = inp.with_suffix('.out').read_text()
    itext = inp.read_text()
    coord = re.search(r'&COORD\s*\n(.*?)&END COORD', itext, re.S)[1]
    entries = [line.split() for line in coord.splitlines() if line.strip()]
    assert [x[0] for x in entries] == a.get_chemical_symbols(), inp
    delta = np.max(np.abs(np.array([[float(v) for v in x[1:4]] for x in entries]) - a.positions))
    assert delta < 1e-5, (inp, delta)
    energies = re.findall(r'ENERGY\| Total FORCE_EVAL \( QS \) energy \[a\.u\.\]:\s+(\S+)', text)
    blocks = re.findall(r'ATOMIC FORCES in \[a\.u\.\](.*?)SUM OF ATOMIC FORCES', text, re.S)
    assert len(energies) == len(blocks) == 1, inp
    forces = np.array([[float(v) for v in line.split()[3:6]] for line in blocks[0].splitlines()
                       if len(line.split()) == 6 and line.split()[0].isdigit()]) * ha / 0.529177210903
    assert forces.shape == (len(a), 3) and np.isfinite(forces).all(), inp
    energy = float(energies[0]) * ha
    residual = (energy - sum(e0[int(z)] for z in a.numbers)) / len(a)
    norms = np.linalg.norm(forces, axis=1)
    d = a.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    radii_sum = covalent_radii[a.numbers][:, None] + covalent_radii[a.numbers][None, :]
    ratio = d / radii_sum
    i, j = np.unravel_index(np.argmin(ratio), ratio.shape)
    duplicates = sum(len(b) == len(a) and np.array_equal(b.numbers, a.numbers)
                     and np.array_equal(np.round(b.positions, 4), np.round(a.positions, 4)) for b in pool)
    basic_pass = (-20 < residual < 7 and norms.max() <= 40 and np.isfinite(energy)
                  and 'PROGRAM ENDED AT' in text and 'SCF run converged' in text
                  and 'SCF run NOT converged' not in text)
    rows.append(dict(name=inp.stem, system=a.info['system_type'], n_atoms=len(a),
                     energy_eV=energy, residual_eV_per_atom=residual,
                     max_force_eV_A=float(norms.max()), max_force_atom=int(norms.argmax()) + 1,
                     max_force_element=a[int(norms.argmax())].symbol,
                     min_distance_A=float(d.min()), min_covalent_ratio=float(ratio[i, j]),
                     closest_ratio_pair=f'{a[i].symbol}{i+1}-{a[j].symbol}{j+1}',
                     input_coordinate_max_delta_A=float(delta), duplicates_in_pool=duplicates,
                     basic_dft_checks_pass=bool(basic_pass),
                     candidate_id=a.info.get('candidate_id', '')))
    a.info.update(REF_energy=energy, al_round=8, source='cp2k_sp', cp2k_name=inp.stem,
                  stress_missing=True)
    a.arrays['REF_forces'] = forces
    a.calc = None
    if basic_pass:
        parsed.append(a)
with (OUT / 'structure_diagnostics.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
write(OUT / 'round8_dft_basic_checks.extxyz', parsed)
lines = ['Round 8 DFT audit', f'Parsed: {len(rows)}; basic checks passed: {len(parsed)}.',
         'Production pool and training_clean.xyz were not modified.',
         'Basic checks are not a model validation or a decision to exclude high-uncertainty frames.',
         'Stress is not extracted in this staged dataset.', '']
for system in sorted({r['system'] for r in rows}):
    group = [r for r in rows if r['system'] == system]
    lines.append(f"{system}: n={len(group)}, residual={min(r['residual_eV_per_atom'] for r in group):.4f}..{max(r['residual_eV_per_atom'] for r in group):.4f} eV/atom, max force={max(r['max_force_eV_A'] for r in group):.4f} eV/A")
lines += ['', 'Largest forces:']
for r in sorted(rows, key=lambda r: r['max_force_eV_A'], reverse=True)[:10]:
    lines.append(f"{r['name']}: Fmax={r['max_force_eV_A']:.4f}, atom={r['max_force_element']}{r['max_force_atom']}, min distance={r['min_distance_A']:.4f}, min covalent ratio={r['min_covalent_ratio']:.4f}")
lines += ['', f"Duplicates in existing pool: {sum(r['duplicates_in_pool'] > 0 for r in rows)}"]
(OUT / 'summary.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
