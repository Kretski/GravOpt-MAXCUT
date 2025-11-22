import numpy as np
from numba import njit, prange
import time
import sys
import os
import urllib.request

@njit(parallel=True, fastmath=True)
def gravopt_maxcut(adj, max_steps=5000, early_stop_steps=80, patience_thr=1e-6):
    n = adj.shape[0]
     x = 2.0 * (np.random.randint(0, 2, size=n).astype(np.float64)) - 1.0
    best_cut = 0.0
    best_x = x.copy()
    no_improve_counter = 0
    prev_best = 0.0

    for step in range(max_steps):
        cut = 0.0
        for i in prange(n):
            for j in prange(i + 1, n):
                if adj[i, j] != 0:
                    cut += adj[i, j] * (1.0 - x[i] * x[j])
        cut *= 0.5

        if cut > best_cut + 1e-8:
            best_cut = cut
            best_x = x.copy()
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        # early stopping logic
        if step > 1000 and no_improve_counter >= early_stop_steps:
            improvement = (best_cut - prev_best) / best_cut if best_cut > 0 else 0
            if improvement < patience_thr:
                break
        prev_best = best_cut

        # GravOpt core – gravitational-style force
        force = np.zeros(n)
        for i in prange(n):
            f = 0.0
            for j in prange(n):
                if adj[i, j] != 0:
                    f += adj[i, j] * x[j] * (1.0 - x[i] * x[j])  # bad cut → push away
            force[i] = f

        alpha = 0.995 ** step
        x = np.sign(x + alpha * force)

        if np.all(x == 0):  # safety
            x[np.random.randint(n)] = 1.0

    return best_x, best_cut


def load_graph_edgelist(path):
    # Ако файла го няма – сваля G81 автоматично
    if not os.path.exists(path):
        print(f"{path} not found → downloading official G81 benchmark...")
        url = "https://raw.githubusercontent.com/Kretski/GravOpt-MAXCUT/main/G81.edges"
        urllib.request.urlretrieve(url, path)
        print("G81 downloaded and ready!")

    # Чете само първите две колони (u v)
    data = np.loadtxt(path, dtype=int, usecols=(0, 1))
    n = data.max() + 1
    adj = np.zeros((n, n), dtype=np.float32)

    # Ако има тегла (трети стълб) → |w|, иначе 1.0
    try:
        weights = np.loadtxt(path, dtype=float, usecols=2)
        for (u, v), w in zip(data, weights):
            adj[u-1, v-1] = abs(w)
            adj[v-1, u-1] = abs(w)
    except:
        for u, v in data:
            adj[u-1, v-1] = 1.0
            adj[v-1, u-1] = 1.0

    return adj


if __name__ == "__main__":
    # По подразбиране – пуска G81
    graph_path = "G81.edges" if len(sys.argv) < 2 else sys.argv[1]
    max_steps = 5000 if len(sys.argv) < 3 else int(sys.argv[2])

    adj = load_graph_edgelist(graph_path)
    print(f"Graph loaded: {adj.shape[0]} nodes, {int(adj.sum() / 2)} edges")

    start = time.time()
    partition, cut = gravopt_maxcut(adj, max_steps=max_steps)
    elapsed = time.time() - start

    total_weight = adj.sum() / 2
    ratio = cut / total_weight

    print(f"Cut value: {cut:.2f}")
    print(f"Ratio: {ratio:.6f}")
    print(f"Time: {elapsed:.1f} seconds")

