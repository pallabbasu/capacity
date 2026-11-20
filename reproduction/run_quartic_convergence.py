#!/usr/bin/env python3
"""Reproduce a two-resolution, three-regulator polar Janus quartic check.

Run from any directory with Python and NumPy installed:
    python3 run_quartic_convergence.py

The default sequential run writes quartic_convergence.json and .csv in the
repository's results directory. --resume preserves completed replica solves.

The reported q4_anchor is 17/96 - 2/9 - d w4/dn at n=1.  It uses exact
sharp-interface entropy coefficients and is an estimator of the sharp Janus
gamma^4 coefficient.  At finite source smoothing it is not the quartic
coefficient of a separate exact Janus solution.  q4_unanchored_diagnostic
instead uses central averages of both numerical entropy coefficients and the
formal sharp-interface charge conversion; it exposes their slower regulator
convergence.  Neither estimator comes with a certified continuum error bar.
"""

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import time

from janus_polar_quartic import np, solve


BASE = Path(__file__).resolve().parent
SOLVER = BASE / "janus_polar_quartic.py"
RESOLUTIONS = {
    "coarse": {"nr": 4800, "nt": 2048, "hard": 256, "gaussian": 128, "tanh": 128},
    "fine": {"nr": 9600, "nt": 4096, "hard": 512, "gaussian": 256, "tanh": 256},
}


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def summarize(case, step):
    if len(case["raw"]) != 2:
        return
    minus, plus = sorted(case["raw"], key=lambda row: row["n"])
    w2_prime = (plus["w2"] - minus["w2"]) / (2 * step)
    w4_prime = (plus["w4"] - minus["w4"]) / (2 * step)
    w2_average = (plus["w2"] + minus["w2"]) / 2
    w4_average = (plus["w4"] + minus["w4"]) / 2
    case.update({
        "w2_prime": w2_prime,
        "w4_prime": w4_prime,
        "w2_average": w2_average,
        "w4_average": w4_average,
        "q4_anchor": 17 / 96 - 2 / 9 - w4_prime,
        "q4_unanchored_diagnostic": w4_average - w4_prime + (4 / 3) * (w2_average - w2_prime),
        "w2_entropy_error": w2_average + 1 / 4,
        "w4_entropy_error": w4_average - 17 / 96,
        "w2_derivative_error": w2_prime + 1 / 12,
        "quartic_constraint_rms_max": max(
            row[key] for row in case["raw"]
            for key in ("constraint_cross4", "constraint_rr4")
        ),
        "w4_reconstruction_discrepancy_max": max(
            abs(row["w4"] - row["w4_check"]) for row in case["raw"]
        ),
        "completed": True,
    })


def write_csv(path, payload):
    keys = ["resolution", "regulator", "nr", "nt", "modes", "rmax", "replica_step",
            "w4_prime", "q4_anchor", "q4_unanchored_diagnostic", "w2_average",
            "w4_average", "w2_derivative_error", "quartic_constraint_rms_max",
            "w4_reconstruction_discrepancy_max"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for case in payload["cases"]:
            if case.get("completed"):
                writer.writerow(case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", choices=["coarse", "fine", "both"], default="both")
    parser.add_argument("--regulators", nargs="+", choices=["hard", "gaussian", "tanh"],
                        default=["hard", "gaussian", "tanh"])
    parser.add_argument("--replica-step", type=float, default=0.00125)
    parser.add_argument("--rmax", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=BASE.parent / "results" / "quartic_convergence.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.suffix != ".json":
        parser.error("--output must specify a .json file")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(SOLVER.read_bytes()).hexdigest()
    metadata = {
        "solver_sha256": source_hash,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "replica_step": args.replica_step,
        "rmax": args.rmax,
        "physical_source_fixed": True,
        "q4_anchor_definition": "17/96 - 2/9 - (w4(1+h)-w4(1-h))/(2*h)",
        "q4_unanchored_definition": "w4_average - w4_prime + (4/3)*(w2_average-w2_prime)",
        "accuracy_status": "Finite grid and source regulators; no certified continuum error bound.",
        "resolution_note": "The fine resolution doubles radial and angular grids and source cutoff/steepness together.",
    }
    if args.resume and output.exists():
        payload = json.loads(output.read_text())
        for key in ("solver_sha256", "replica_step", "rmax", "physical_source_fixed"):
            if payload["metadata"][key] != metadata[key]:
                raise ValueError(f"Cannot resume with changed {key}")
    else:
        payload = {"created_utc": datetime.now(timezone.utc).isoformat(),
                   "metadata": metadata, "cases": []}
    selected = list(RESOLUTIONS) if args.resolution == "both" else [args.resolution]
    for resolution in selected:
        grid = RESOLUTIONS[resolution]
        for regulator in args.regulators:
            case_id = f"{resolution}:{regulator}"
            case = next((row for row in payload["cases"] if row["case_id"] == case_id), None)
            if case is None:
                case = {"case_id": case_id, "resolution": resolution, "regulator": regulator,
                        "nr": grid["nr"], "nt": grid["nt"], "modes": grid[regulator],
                        "rmax": args.rmax, "replica_step": args.replica_step,
                        "physical_source": True, "raw": [], "completed": False}
                payload["cases"].append(case)
            if case.get("completed"):
                print(json.dumps({"status": "resumed_complete", "case": case_id,
                                  "q4_anchor": case["q4_anchor"]}), flush=True)
                continue
            for n in (1 - args.replica_step, 1 + args.replica_step):
                if any(abs(row["n"] - n) < 1e-13 for row in case["raw"]):
                    continue
                print(json.dumps({"status": "starting", "case": case_id, "n": n,
                                  "nr": grid["nr"], "nt": grid["nt"],
                                  "modes": grid[regulator]}), flush=True)
                start = time.monotonic()
                result = solve(n, nr=grid["nr"], nt=grid["nt"], rmax=args.rmax,
                               physical_source=True, smoothing=regulator, modes=grid[regulator])
                result["elapsed_seconds"] = time.monotonic() - start
                case["raw"].append(result)
                write_json(output, payload)
                print(json.dumps({"status": "computed", "case": case_id, "n": n,
                                  "w4": result["w4"],
                                  "seconds": result["elapsed_seconds"]}), flush=True)
                gc.collect()
            summarize(case, args.replica_step)
            write_json(output, payload)
            write_csv(output.with_suffix(".csv"), payload)
            print(json.dumps({"status": "case_complete", "case": case_id,
                              "q4_anchor": case["q4_anchor"],
                              "q4_unanchored_diagnostic": case["q4_unanchored_diagnostic"],
                              "quartic_constraint_rms_max": case["quartic_constraint_rms_max"]}), flush=True)
    payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output, payload)
    write_csv(output.with_suffix(".csv"), payload)
    print(json.dumps({"status": "complete", "json": str(output),
                      "csv": str(output.with_suffix(".csv"))}), flush=True)


if __name__ == "__main__":
    main()
