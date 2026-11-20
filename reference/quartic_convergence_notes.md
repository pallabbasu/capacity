# Independent polar quartic convergence calculation

All six regulator/resolution cases were freshly computed, with two replica solves per case, using the unmodified `janus_polar_quartic.py` solver.
The fine raw numerical results exactly reproduce every numerical field in the prior `janus_regulator_fine.json` file (maximum absolute difference zero).
The twelve replica solves took 74.26 seconds in total on the reference machine.

Run from the repository root.

```sh
python reproduction/run_quartic_convergence.py
```

The reference run used Python 3.14.10 with NumPy 2.3.4.
The output records the solver SHA-256 hash and all raw per-replica results, including independent Einstein-constraint residuals, direct metric reconstruction, and exact entropy benchmarks.
Execution is sequential to limit memory consumption.
`--resume` preserves completed cases and completed replica points after interruption.

## Quantity estimated

For the sharp Janus half-jump $a_\phi=\Delta\phi/2=\gamma+2\gamma^3/3+\cdots$, the refined-entropy coefficient is

\[
\widetilde c(n)/c=[1+a_\phi^2w_2(n)+a_\phi^4w_4(n)+\cdots]/n.
\]

The exact sharp-interface benchmarks are $w_2(1)=-1/4$, $w_2'(1)=-1/12$, and $w_4(1)=17/96$.
Thus the reported estimator is

\[
q_4^{\rm est}=17/96-2/9-
\frac{w_4(1+h_n)-w_4(1-h_n)}{2h_n}.
\]

It inserts exact sharp-interface entropy and quadratic-response data, and numerically estimates the remaining derivative.
The JSON field `q4_anchor` stores this estimator of the sharp-interface coefficient.
The calculation holds the physical regulated source fixed through the cubic boundary correction $F_{3,\infty}=\vartheta_2 f_M'$, with $\vartheta_2'=p_{2,\infty}-q_{2,\infty}-w_2$.

## Actual two-resolution results

Every case uses $r_{\max}=10$, $h_n=0.00125$, and $n=0.99875,1.00125$.
The hard regulator keeps positive odd harmonics $m<M$ of the square wave, so $M$ is a harmonic cutoff rather than a count of odd harmonics.
The Gaussian regulator multiplies the square-wave Fourier coefficients by $\exp[-(m/M)^2]$.
The hyperbolic-tangent regulator is $\tanh(M\cos\theta)$.

| Regulator | $N_r$ | $N_\theta$ | $M$ | $w_4'(1)$ central estimate | $q_4^{\rm est}$ | Maximum quartic-constraint RMS |
|---|---:|---:|---:|---:|---:|---:|
| Hard Fourier | 4800 | 2048 | 256 | −0.125518146294401 | 0.080379257405512 | 2.04583003364e−6 |
| Hard Fourier | 9600 | 4096 | 512 | −0.125457226501391 | 0.080318337612502 | 5.07110905202e−7 |
| Gaussian Fourier | 4800 | 2048 | 128 | −0.125618795473326 | 0.080479906584437 | 2.07007739026e−6 |
| Gaussian Fourier | 9600 | 4096 | 256 | −0.125520556206882 | 0.080381667317993 | 5.23755134976e−7 |
| Hyperbolic tangent | 4800 | 2048 | 128 | −0.125578884996558 | 0.080439996107669 | 2.09120568934e−6 |
| Hyperbolic tangent | 9600 | 4096 | 256 | −0.125492525742954 | 0.080353636854065 | 5.16244619384e−7 |

The displayed digits are raw computed values, not certified significant digits.
The coarse-to-fine change doubles the radial grid, angular grid, and source cutoff/steepness simultaneously; it is a joint resolution and sharp-source refinement, not an isolated radial convergence test.
The regulator spread decreases from 0.000100649178925 to 0.000063329705491.
All three sequences decrease toward the independently computed spectral value approximately 0.08025.
The remaining differences from 0.08025 on the fine grid range from 0.00006834 to 0.00013167.
These data independently support the positive sign and a coefficient approximately 0.08; they do not establish equality with the spectral result to all five decimal places.

## Entropy benchmark and unanchored diagnostics

The raw JSON also records the symmetric averages $\bar w_j=[w_j(1+h_n)+w_j(1-h_n)]/2$ and the diagnostic

\[
q_4^{\rm unanchored}=\bar w_4-w_4'
+\frac43(\bar w_2-w_2').
\]

This diagnostic uses numerical entropy coefficients and the formal sharp-interface conversion to $\gamma$.
It is useful for exposing the slower source-regulator convergence of the entropy terms.
The symmetric averages also have $O(h_n^2)$ finite-difference bias.

| Regulator | Resolution | $\bar w_4-17/96$ | $q_4^{\rm unanchored}$ |
|---|---|---:|---:|
| Hard Fourier | Coarse | −0.000474477039930 | 0.080600560887885 |
| Hard Fourier | Fine | −0.000238582613266 | 0.080428267411981 |
| Gaussian Fourier | Coarse | −0.002290404072038 | 0.081663990639712 |
| Gaussian Fourier | Fine | −0.001139039657065 | 0.080992063038263 |
| Hyperbolic tangent | Coarse | −0.001433215964626 | 0.081190940032596 |
| Hyperbolic tangent | Fine | −0.000714024423986 | 0.080737158320700 |

## Interpretation and remaining errors

The small interior constraint residuals check the independently evaluated Einstein constraints but do not bound the final observable's error.
Hard Fourier sampling uses $N_\theta=8M$, sufficient to represent the quartic products of the retained harmonics without Fourier aliasing.
Gaussian and hyperbolic-tangent source profiles have infinite Fourier tails, so angular truncation remains a separate error source.
Other systematics are the finite outer radius, the replica finite-difference step, near-tip quadrature, and the sharp-source extrapolation.
The boundary reparametrization develops logarithmic structure, allowing corrections such as powers of $\log M$ divided by $M$; a plain inverse-cutoff extrapolation has not been justified.
No extrapolated continuum number or certified error interval is claimed here.

## Files

- [Sequential driver with restart support](../reproduction/run_quartic_convergence.py)
- [Finite-replica polar solver](../reproduction/janus_polar_quartic.py)
- [Raw results, diagnostics, numerical environment, and source hash](quartic_convergence.json)
- [Table of computed quantities](quartic_convergence.csv)
