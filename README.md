# GravOpt-MAXCUT
Fast MAX-CUT heuristic inspired by gravitational dynamics · 20k–100k nodes in minutes on CPU · single-file Numba implementation
# GravOpt – MAX-CUT Heuristic

Single-file, dependency-light, extremely fast MAX-CUT solver for sparse graphs (10k–200k+ nodes).

Achieves **0.363–0.368 approximation ratio** on real-world sparse graphs (G-set, etc.) while being **50–200× faster** than Simulated Annealing / Tabu Search and completely practical where Goemans-Williamson (SDP) fails due to scaling.

### Performance highlights (Nov 2025)
| Graph   | Nodes | Edges  | Ratio    | Time (single core) | 99% quality reached at |
|---------|-------|--------|----------|---------------------|-----------------------|
| G81     | 20 000| 40 000 | 0.36765  | ~13 h full, ~6–8 min for 99% | step ~1200           |

### Install
```bash
pip install numpy numba
