#!/usr/bin/env python3
"""Reproduce the finite-radius-jump audit using Python's standard library.

Run: python3 reproduction/reproduce_finite_jump.py

The capacity is integrated in two independent variables, x and omega.
The input is the known exact free-boson entanglement spectrum, not a fit.
"""

from math import asinh, cosh, exp, pi, sinh, sqrt


def adaptive_simpson(fn, lo=0.0, hi=40.0, tol=2e-13):
    """Return an integral with adaptive Simpson quadrature and Richardson correction."""
    mid = (lo + hi) / 2
    fa, fm, fb = fn(lo), fn(mid), fn(hi)
    whole = (hi - lo) * (fa + 4 * fm + fb) / 6

    def recurse(a, b, va, vm, vb, old, budget, depth):
        m = (a + b) / 2
        vl, vr = fn((a + m) / 2), fn((m + b) / 2)
        left = (m - a) * (va + 4 * vl + vm) / 6
        right = (b - m) * (vm + 4 * vr + vb) / 6
        delta = left + right - old
        if abs(delta) <= 15 * budget:
            return left + right + delta / 15
        if depth == 0:
            raise ArithmeticError("Quadrature did not reach requested tolerance")
        return recurse(a, m, va, vl, vm, left, budget / 2, depth - 1) + recurse(
            m, b, vm, vr, vb, right, budget / 2, depth - 1
        )

    return recurse(lo, hi, fa, fm, fb, whole, tol, 30)


def bose_weight(x):
    return 1.0 if x == 0 else (x / sinh(x)) ** 2


def capacity_original(s, tol=2e-13):
    if s == 0:
        return 0.0

    def integrand(x):
        return bose_weight(asinh(sinh(x) / s))

    return 6 / pi**2 * adaptive_simpson(integrand, tol=tol)


def capacity_transformed(s, tol=2e-13):
    if s == 0:
        return 0.0

    def integrand(omega):
        jacobian = s * cosh(omega) / sqrt(1 + s * s * sinh(omega) ** 2)
        return bose_weight(omega) * jacobian

    return 6 / pi**2 * adaptive_simpson(integrand, tol=tol)


def main():
    coefficient = 3 / (4 * pi**2) - 3 / 16
    print("Exact R^2 coefficient:", format(coefficient, ".14f"))
    print("radius ratio     R             capacity(x)       capacity(omega)   1-R/4")
    for ratio in (1.0, 1.5, 2.0, 3.0, 10.0):
        s = 2 * ratio / (1 + ratio * ratio)
        reflection = 1 - s * s
        original = capacity_original(s)
        transformed = capacity_transformed(s)
        if abs(original - transformed) > 2e-11:
            raise ArithmeticError("Independent integral representations disagree")
        print(f"{ratio:9.3f}   {reflection:12.9f}   {original:.12f}   "
              f"{transformed:.12f}   {1-reflection/4:.12f}")

    print("\nSmall-R extraction: [capacity(R)-1+R/4]/R^2")
    for reflection in (0.01, 0.003, 0.001, 0.0003):
        cap = capacity_transformed(sqrt(1 - reflection), tol=3e-14)
        extracted = (cap - 1 + reflection / 4) / reflection**2
        print(f"R={reflection:.4g}: {extracted:.10f}")

    i1 = adaptive_simpson(lambda x: bose_weight(x) / cosh(x)**2)
    i2 = adaptive_simpson(lambda x: bose_weight(x) / cosh(x)**4)
    print("\nIntegral checks:")
    print("I1 numerical / exact:", format(i1, ".14f"), format(pi**2/12, ".14f"))
    print("I2 numerical / exact:", format(i2, ".14f"), format(pi**2/36+1/3, ".14f"))
    print("capacity(s=0):", capacity_original(0.0))
    cutoff = 40.0
    tail_bound = (6 / pi**2 * exp(-2 * cutoff)
                  * (2 * cutoff**2 + 2 * cutoff + 1)
                  / (1 - exp(-2 * cutoff))**2)
    print("Upper bound on omitted capacity tail:", format(tail_bound, ".3e"))


if __name__ == "__main__":
    main()
