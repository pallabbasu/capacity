"""Independent checks of the CFT kernel and regulator in Supplemental Section S8.

Run: python3 reproduction/check_cft_kernel.py
Requires mpmath. No network, external data, or fitted coefficients.
"""
import json
import mpmath as mp

mp.mp.dps = 60


def j_integral(n):
    return mp.quad(
        lambda u: (mp.tanh(u / n) - n * mp.tanh(u)) / mp.expm1(2 * u),
        [0, 1, mp.inf],
    )


def h(t):
    if abs(t) < mp.mpf("1e-12"):
        return t / 12 - t**3 / 720 + t**5 / 30240
    return mp.coth(t / 2) / 2 - 1 / t


def sigma(n):
    return mp.quad(lambda t: h(t) / (2 * mp.sinh(n * t / 2)), [0, 1, mp.inf])


def strip_integral(n, epsilon=mp.mpf("1e-14")):
    """Sum all Fourier modes before taking the strip cutoff to zero."""
    def integrand(v):
        aa = mp.atanh(mp.exp(-v))
        bb = mp.polylog(2, mp.exp(-v)) - mp.polylog(2, mp.exp(-2 * v)) / 4
        return 2 * (
            aa * (1 / (n * mp.sinh(v / n)**2) - n / mp.sinh(v)**2)
            + bb * (
                mp.cosh(v / n) / (n**2 * mp.sinh(v / n)**3)
                - n * mp.cosh(v) / mp.sinh(v)**3
            )
        )
    return mp.quad(integrand, [epsilon, mp.mpf("0.01"), 1, mp.inf])


def audit_i(p, epsilon=mp.mpf("1e-5")):
    direct = mp.quad(
        lambda u: mp.exp(-p * u) * (p * mp.sinh(u) + mp.cosh(u)) / mp.sinh(u)**3,
        [epsilon, 1, mp.inf],
    )
    expansion = (
        1 / (2 * epsilon**2) - mp.mpf(1) / 6
        + p**2 / 2 * (
            mp.log(2 * epsilon) + mp.euler - mp.mpf(1) / 2
            + mp.digamma(1 + p / 2)
        ) - p / 2
    )
    # Independently retain the next Taylor coefficient as a convergence check.
    error_after_linear = direct - expansion - (p - p**3) * epsilon / 3
    assert abs(error_after_linear) < 5 * epsilon**2
    return {
        "p": str(p),
        "remainder_over_epsilon": float((direct - expansion) / epsilon),
        "predicted_limit": float((p - p**3) / 3),
    }


def main():
    result = {"I_asymptotic": [audit_i(mp.mpf(p)) for p in [0, 1, "1.3", 2, 3]]}
    assert abs(sigma(1) - mp.log(2) / 2) < mp.mpf("1e-40")
    comparisons = []
    for n in map(mp.mpf, ["0.5", "1.1", "2", "3"]):
        jj = j_integral(n)
        jb = n * (sigma(n) - sigma(1))
        strip = strip_integral(n)
        rotational = strip + mp.pi**2 / 24 * (1 / n - n)
        assert abs(jj - jb) < mp.mpf("1e-40")
        assert abs(rotational + jj) < mp.mpf("1e-11")
        comparisons.append({
            "n": float(n), "J": float(jj),
            "Binet_error": float(jb - jj),
            "strip_plus_rotational_contact": float(rotational),
            "regulator_check_error": float(rotational + jj),
        })
    result["kernel_comparisons"] = comparisons
    d1, d2 = mp.diff(j_integral, 1, 1), mp.diff(j_integral, 1, 2)
    assert abs(d1 + mp.pi**2 / 16) < mp.mpf("1e-40")
    assert abs(d2 - mp.pi**2 / 24) < mp.mpf("1e-40")
    result["derivatives"] = {"J_prime_1": float(d1), "J_double_prime_1": float(d2)}
    result["status"] = "all checks passed"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
