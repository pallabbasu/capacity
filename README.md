# Entanglement capacity at holographic interfaces

This repository contains the reproduction code and reference data for Vishnu Jejjala and Pallab Basu's *Bounds on entanglement fluctuations at holographic interfaces*. The calculations cover Janus with constant scalar potential, a cosine-potential family, the weak-interface CFT kernel, and an independent polar-coordinate check of the quartic Janus coefficient.

## Install

The reference environment used Python 3.14.10. The dependency versions are pinned in [requirements.txt](requirements.txt).

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the checks

Run these commands from the repository root. Each script prints its results to standard output.

```sh
python reproduction/verify_sharp_capacity_certificate.py
python reproduction/check_cft_kernel.py
python reproduction/reproduce_finite_jump.py
python reproduction/potential_beta_spectral.py --beta .7 1 --dps 25
```

These check the angular trial kernel, the CFT replica kernel and regulator terms, the free boson capacity in two integration variables, and the cosine-potential spectral integral. At `b=0.7`, the cosine-potential calculation gives a capacity to entropy ratio of approximately `1.22874721`.

## Reproduce the curves

```sh
python generate_figure_data.py
python plot_capacity.py --data-dir results
```

The first command writes the two CSVs and a mesh and precision comparison to `results/`. The second creates `results/capacity_profiles.pdf`. To plot the supplied reference CSVs, run `python plot_capacity.py`.

Each central charge coefficient in the CSVs is divided by the common central charge `c`. The Janus coordinate `delta` is the scalar jump. In `figure_beta.csv`, the column `beta` is the cosine-potential parameter `b`, which equals the entropy coefficient divided by `c`.

## Reproduce the quartic calculation

```sh
python reproduction/run_quartic_convergence.py
```

The driver runs three source regulators at two resolutions with two replica indices per case. It writes the raw solves and a summary table to `results/quartic_convergence.json` and `results/quartic_convergence.csv`. Use `--resume` to continue from a checkpoint. A smaller run with a separate checkpoint is

```sh
python reproduction/run_quartic_convergence.py --resolution coarse --regulators hard --output results/quartic_coarse_hard.json
```

The full run is sequential. The fine grid uses 9,600 radial intervals and 4,096 angular points and requires substantially more memory than the coarse grid. The JSON field `q4_anchor` gives the quartic estimator from the centered replica derivative of `w4`, using the exact sharp Janus entropy and quadratic response. The three fine estimates range from 0.080318 to 0.080382. The grid and source cutoff change together, so the spread is a convergence diagnostic. See the [regulator definitions and error discussion](reference/quartic_convergence_notes.md).

## Reference results

The [reference directory](reference/) contains the two plotted datasets, validation outputs, and all twelve raw replica solves. The quartic JSON records the solver hash, numerical environment, grid parameters, and independent constraint residuals. Fresh runs write to `results/`.
