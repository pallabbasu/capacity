#!/usr/bin/env python3
"""Plot the Janus and cosine-potential coefficients from the CSV tables."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CAPACITY = "#2363a4"
TRANSMISSION = "#008f65"
UPPER = "#777777"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "reference")
    parser.add_argument("--output", type=Path, default=ROOT / "results/capacity_profiles.pdf")
    args = parser.parse_args()

    janus = np.genfromtxt(args.data_dir / "figure_janus.csv", delimiter=",", names=True, ndmin=1)
    cosine = np.genfromtxt(args.data_dir / "figure_beta.csv", delimiter=",", names=True, ndmin=1)

    with plt.rc_context({"font.family": "serif", "mathtext.fontset": "cm", "text.usetex": False}):
        fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.2), layout="constrained")
        for ax in (left, right):
            ax.set_ylabel(r"Coefficient divided by $c$")
            ax.set_axisbelow(True)
            ax.grid(color="0.92", linewidth=0.7)

        left.set(xlim=(0, 8), ylim=(0, 3), xlabel=r"$\Delta\phi$",
                 title="(a) Janus: constant potential")
        left.plot(janus["delta"], janus["capacity"], color=CAPACITY, label=r"$c_{\mathrm{cap}}/c$")
        left.plot(janus["delta"], janus["entropy"], "k--", label=r"$c_{\mathrm{eff}}/c$")
        left.plot(janus["delta"], janus["transmission"], color=TRANSMISSION,
                  linestyle="-.", label=r"$c_{LR}/c$")
        left.plot(janus["delta"], janus["upper"], color=UPPER, linestyle=":",
                  label=r"$c_{\mathrm{eff}}^2/(c\,c_{LR})$")
        left.legend(loc="upper left", frameon=False, fontsize=8)

        # Plot the exact envelopes over the full parameter interval.
        b = np.linspace(0, 1, 201)
        lower = 1 - 4 * b * (1 - b**2) / 9
        right.set(xlim=(0, 1), ylim=(0, 1.08), xlabel=r"$b=c_{\mathrm{eff}}/c$",
                  title="(b) Cosine-potential family")
        right.fill_between(b, lower, 1, color=CAPACITY, alpha=0.12)
        right.plot(cosine["beta"], cosine["capacity"], color=CAPACITY, label=r"$c_{\mathrm{cap}}/c$")
        right.plot(b, b, "k--", label=r"$c_{\mathrm{eff}}/c$")
        right.axhline(1, color=UPPER, linestyle=":", label="Upper bound")
        right.plot(b, lower, color="#153f69", linestyle="-.", label="Analytic lower bound")
        right.legend(loc="lower right", frameon=False, fontsize=8)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output)
        plt.close(fig)


if __name__ == "__main__":
    main()
