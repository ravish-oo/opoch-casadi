# OPOCH CasADi Verification Layer

**Precision Refinement for Safety-Critical Optimization**

IPOPT (the industry-standard NLP solver) returns `Solve_Succeeded` even when KKT residuals are as high as 10⁻⁵. For safety-critical systems (rockets, robots, medical devices), higher precision is required. OPOCH drives solutions to machine precision (10⁻¹³), providing mathematical certification that constraints are satisfied.

```
IPOPT alone:     r_max = 10⁻⁵
IPOPT + OPOCH:   r_max = 10⁻¹³

Precision improvement: 10³x to 10⁸x
Computational overhead: ~8%
```

## Method

OPOCH performs post-hoc KKT verification on IPOPT solutions:

1. **Compute** the maximum KKT residual in unscaled space:
   ```
   r_max = max(r_primal, r_dual, r_complementarity)
   ```

2. **Certify** if r_max ≤ ε (default ε = 10⁻⁶)

3. **Refine** if verification fails: re-solve with `tol=10⁻¹²`, `nlp_scaling_method=none`, warm start from previous solution

The verification correctly handles IPOPT's multiplier sign convention for two-sided bounds.

**[Complete mathematical formulation →](MATH.md)**

## Results

27 benchmark problems tested. IPOPT returned `Solve_Succeeded` on all 27.

| Metric | IPOPT Default | After OPOCH Refinement |
|--------|---------------|------------------------|
| Certified (r_max ≤ 10⁻⁶) | 22/27 | 27/27 |
| Worst r_max | 2.58×10⁻⁴ | 2.57×10⁻¹⁰ |

5 problems had r_max > 10⁻⁶ despite `Solve_Succeeded`. All were brought to certification after refinement.

**[Full benchmark results →](RESULTS.md)**

## Quick Start

```bash
pip install casadi numpy
cd src/opoch_casadi
python precision_comparison.py
```

## Benchmarks

| Source | Count | Description |
|--------|-------|-------------|
| [CasADi Official](https://github.com/casadi/casadi/tree/main/docs/examples/python) | 6 | rocket, race_car, chain_qp, etc. |
| [Hock-Schittkowski](https://en.wikipedia.org/wiki/Hock-Schittkowski_collection) | 10 | Classic NLP test problems (1981) |
| [NIST StRD](https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml) | 6 | Nonlinear regression reference datasets |
| Custom | 5 | Optimal control, parameter estimation |

## Repository Structure

```
opoch-casadi/
├── README.md
├── MATH.md                        # Mathematical formulation
├── RESULTS.md                     # Full benchmark results
└── src/opoch_casadi/
    ├── kkt_verifier.py            # KKT verification with multiplier decomposition
    ├── nlp_contract.py            # NLP data structures
    ├── precision_comparison.py    # Reproduce benchmark results
    ├── casadi_official_examples.py
    ├── hock_schittkowski.py
    ├── suite_a_industrial.py
    └── suite_b_regression.py
```

## License

MIT
