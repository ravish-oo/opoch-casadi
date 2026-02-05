# OPOCH CasADi Verification Layer

**Precision Refinement for Safety-Critical Optimization**

IPOPT returns `Solve_Succeeded` even when KKT residuals are as high as 10⁻⁵. For safety-critical systems, this is unacceptable. OPOCH drives solutions to machine precision (10⁻¹³), providing mathematical certification.

```
IPOPT alone:     r_max = 10⁻⁵   →  "Approximately optimal"
IPOPT + OPOCH:   r_max = 10⁻¹³  →  "Machine precision certified"
```

## Results

**27/27 benchmark problems certified** across aerospace, pharma, and engineering domains.

| Metric | IPOPT Default | OPOCH Refinement |
|--------|---------------|------------------|
| Certified (r_max ≤ 10⁻⁶) | 22/27 | **27/27** |
| Worst case | 2.58×10⁻⁴ | 2.57×10⁻¹⁰ |

IPOPT returned "Success" on all 27, but 5 had residuals above certification threshold. OPOCH fixed all of them.

**[Full benchmark results →](RESULTS.md)**

## Method

1. **Verify**: Compute KKT residuals in unscaled space after IPOPT returns
2. **Repair**: If r_max > ε, re-run with strict tolerances (10⁻¹², no scaling, warm start)

We do not modify IPOPT's algorithm. We verify its output and repair when needed.

**[Mathematical formulation →](MATH.md)**

## Quick Start

```bash
pip install casadi numpy
cd src/opoch_casadi
python precision_comparison.py
```

## Benchmarks

| Source | Problems | Description |
|--------|----------|-------------|
| [CasADi Official](https://github.com/casadi/casadi/tree/main/docs/examples/python) | 6 | rocket, race_car, chain_qp, etc. |
| [Hock-Schittkowski](https://en.wikipedia.org/wiki/Hock-Schittkowski_collection) | 10 | Classic optimization benchmarks (1981) |
| [NIST StRD](https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml) | 6 | Statistical reference datasets |
| Industrial | 5 | Rocket landing, robotics, parameter estimation |

## Repository Structure

```
opoch-casadi/
├── README.md                      # This file
├── RESULTS.md                     # Full benchmark results
├── MATH.md                        # Mathematical formulation
│
└── src/opoch_casadi/
    ├── kkt_verifier.py            # Core verification logic
    ├── nlp_contract.py            # NLP data structures
    ├── precision_comparison.py    # Run this to reproduce results
    ├── casadi_official_examples.py
    ├── hock_schittkowski.py
    ├── suite_a_industrial.py
    └── suite_b_regression.py
```

## License

MIT
