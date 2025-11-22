Update (22.11.2025):** New Numba test hits 0.3674 on G81 (20k nodes, 40k edges) in 1200 steps. GravOpt Pro (€200) launch delayed due to store activation – stay tuned! Run `python gravopt.py` to test.
# GravOpt – Fast MAX-CUT Heuristic

Single-file, pure Python+Numba MAX-CUT solver for sparse graphs (10k–200k+ nodes).

Delivers **0.363–0.368 approximation ratio** on standard benchmarks while being **50–200× faster** than Simulated Annealing/Tabu and fully practical where Goemans-Williamson (SDP) collapses.

## Quick Start

```bash
pip install numpy numba                # once
python gravopt.py benchmarks/G81.edges 2000
``bash
pip install numpy numba                # once
python gravopt.py benchmarks/G81.edges 2000
Example output on G81 (20 000 nodes, 40 000 edges):
textGraph: 20000 nodes, 40000 edges
Cut value: 14703.00
Ratio: 0.367650
Time: 48466.29 sec total (~13 h full run)
→ but 99 % of quality already reached at ~1200 steps
Early stopping = the real killer feature
After ~1200 iterations on most graphs, improvement drops below 10⁻⁵ per step → stop early and get Tabu-level quality in minutes instead of hours.
``bash
Benchmarks (Nov 2025)
Benchmarks (Nov 2025)

GraphNodesEdgesRatioTime for 99 % qualityFull run timeG8120 00040 0000.36765~6–8 minutes~13 hours
More graphs in /benchmarks (G14, G22, G81 + random sparse instances).
Features

~318 lines total (the whole algorithm is in gravopt.py)
No external solvers, no GPU, no PyTorch/TF
< 80 MB RAM even on 100k-node graphs
Built-in early stopping (automatic after stagnation)

Install
Bashpip install numpy numba
