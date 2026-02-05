# Benchmark Results: 27/27 Problems Certified

## Summary

| Metric | IPOPT Default | OPOCH Refinement |
|--------|---------------|------------------|
| Problems certified (r_max <= 1e-6) | 22/27 | **27/27** |
| Worst r_max | 2.58e-04 | 2.57e-10 |
| Best r_max | 0.00e+00 | 0.00e+00 |

**Key Finding**: IPOPT returned "Solve_Succeeded" for all 27 problems, but 5 had r_max > 1e-6 (not actually certified). OPOCH refinement fixed all of them.

---

## Full Results by Category

### Aerospace & Robotics (Optimal Control)

| Problem | Application | Vars | IPOPT r_max | OPOCH r_max | Improvement |
|---------|-------------|------|-------------|-------------|-------------|
| rocket_landing | SpaceX-style soft landing | 60 | 9.94e-05 | 9.69e-13 | 103M x |
| rocket | Fuel-optimal trajectory | 50 | 3.71e-09 | 4.05e-13 | 9,164x |
| race_car | F1-style optimal lap time | 303 | 6.83e-09 | 8.64e-13 | 7,907x |
| van_der_pol_ocp | Oscillator control (robotics) | 60 | 2.68e-09 | 6.18e-13 | 4,336x |
| vdp_multiple_shooting | Multi-stage trajectory | 62 | 9.02e-09 | 9.52e-13 | 9,473x |
| robot_arm_2d | Inverse kinematics | 2 | 1.46e-09 | 7.31e-14 | 19,992x |

### Pharma & Chemistry (NIST Curve Fitting)

| Problem | Application | Vars | IPOPT r_max | OPOCH r_max | Improvement |
|---------|-------------|------|-------------|-------------|-------------|
| gauss1 | Spectroscopy peak fitting | 8 | 2.58e-04 | 2.57e-10 | 1M x |
| misra1a | Drug metabolism decay | 2 | 4.65e-05 | 2.15e-09 | 21,643x |
| chwirut2 | Chemical reaction rates | 3 | 1.62e-05 | 3.63e-10 | 44,572x |
| lanczos1 | Multi-exponential decay | 6 | 7.30e-09 | 6.26e-13 | 11,663x |
| box3d | Reaction kinetics | 3 | 2.39e-09 | 1.19e-13 | 20,121x |
| parameter_estimation | System identification | 2 | 2.51e-09 | 1.25e-13 | 20,002x |

### Engineering Design (Hock-Schittkowski Benchmarks)

| Problem | Application | Vars | IPOPT r_max | OPOCH r_max | Improvement |
|---------|-------------|------|-------------|-------------|-------------|
| hs100 | Multi-constraint design | 7 | 1.27e-06 | 8.81e-13 | 1.4M x |
| hs065 | Chemical reactor design | 3 | 4.50e-07 | 1.21e-13 | 3,730,000x |
| hs071 | Process optimization | 4 | 2.46e-07 | 8.85e-13 | 277,591x |
| hs038 | Structural optimization | 4 | 1.98e-08 | 4.69e-14 | 422,808x |
| hs044 | Resource allocation | 4 | 1.18e-07 | 9.86e-13 | 119,712x |
| hs076 | Constrained design | 4 | 4.45e-08 | 9.27e-13 | 47,940x |
| hs035 | Linear constraints | 3 | 1.85e-08 | 4.35e-13 | 42,600x |
| chain_qp | Chain of masses (physics) | 80 | 3.72e-08 | 9.87e-13 | 37,661x |
| quadratic_constrained | Convex QP | 2 | 1.75e-08 | 8.75e-13 | 19,978x |

### Classic Test Functions (Solver Validation)

| Problem | Application | Vars | IPOPT r_max | OPOCH r_max | Improvement |
|---------|-------------|------|-------------|-------------|-------------|
| rosenbrock_10d | 10-D banana function | 10 | 3.61e-09 | 8.05e-14 | 44,787x |
| rosenbrock_5d | 5-D banana function | 5 | 3.48e-09 | 3.01e-14 | 115,931x |
| rosenbrock | 3-D constrained | 3 | 9.63e-33 | 9.63e-33 | 1x |
| rosenbrock_2d | Classic 2-D | 2 | 9.82e-10 | 3.94e-14 | 24,941x |
| rosenbrock_nist | NIST variant | 2 | 9.82e-10 | 3.94e-14 | 24,941x |
| simple_nlp | Basic NLP | 2 | 0.00e+00 | 0.00e+00 | - |

---

## Problems Where IPOPT Lied

These 5 problems returned "Solve_Succeeded" but had r_max > 1e-6:

| Problem | IPOPT Status | IPOPT r_max | OPOCH r_max | Fixed? |
|---------|--------------|-------------|-------------|--------|
| gauss1 | Solve_Succeeded | 2.58e-04 | 2.57e-10 | YES |
| rocket_landing | Solve_Succeeded | 9.94e-05 | 9.69e-13 | YES |
| misra1a | Solve_Succeeded | 4.65e-05 | 2.15e-09 | YES |
| chwirut2 | Solve_Succeeded | 1.62e-05 | 3.63e-10 | YES |
| hs100 | Solve_Succeeded | 1.27e-06 | 8.81e-13 | YES |

---

## Benchmark Sources

| Source | Count | Description |
|--------|-------|-------------|
| [CasADi Official](https://github.com/casadi/casadi/tree/main/docs/examples/python) | 6 | Direct copies from CasADi GitHub |
| [Hock-Schittkowski](https://en.wikipedia.org/wiki/Hock%E2%80%93Schittkowski_collection) | 10 | Classic optimization benchmarks (1981) |
| [NIST StRD](https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml) | 6 | Statistical reference datasets |
| Custom Industrial | 5 | Optimal control & robotics problems |

---

## Reproducing Results

```bash
pip install casadi numpy
cd src/opoch_casadi
python precision_comparison.py   # Generate this table
python run_all.py                # Full benchmark with details
```

---

## Methodology

1. **Round 0**: Run IPOPT with default settings (tol=1e-8)
2. **Verify**: Compute KKT residuals in unscaled space
3. **Round 1** (if needed): Re-run with strict settings (tol=1e-12, no scaling, warm start)
4. **Certify**: Solution is certified when r_max <= 1e-6

See [MATH.md](MATH.md) for complete mathematical formulation.
