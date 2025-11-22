   import numpy as np
from numba import njit, prange
import time
import sys
import os
import urllib.request


@njit(parallel=True, fastmath=True)
def gravopt_maxcut(adj, max_steps=5000, patience=80, thr=1e-6):
    n = adj.shape[0]
    x = 2.0 * (np.random.randint(0, 2, size=n).astype(np.float64)) - 1.0

    best_cut = 0.0
    best_x = x.copy()
    no_improve = 0
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
            no_improve = 0
        else:
            no_improve += 1

        if step > 1000 and no_improve >= patience:
            impr = (best_cut - prev_best) / (best_cut + 1e-12)
            if impr < thr:
                return best_x, best_cut
        prev_best = best_cut

        force = np.zeros(n)
        for i in prange(n):
            f = 0.0
            for j in prange(n):
                if adj[i, j] != 0:
                    f += adj[i, j] * x[j] * (1.0 - x[i] * x[j])
            force[i] = f

        alpha = 0.995 ** step
        x = np.sign(x + alpha * force)
        if np.all(x == 0):
            x[np.random.randint(n)] = 1.0

    return best_x, best_cut


def load_graph_edgelist(path):
    if not os.path.exists(path):
        print(f"{path} not found → downloading official G81...")
        url = "https://raw.githubusercontent.com/Kretski/GravOpt-MAXCUT/main/G81.edges"
        urllib.request.urlretrieve(url, path)
        print("G81 downloaded!\n")

    # Основният фикс – G81 винаги е 20 000 възела
    n = 20000
    adj = np.zeros((n, n), dtype=np.float32)

    data = np.loadtxt(path, dtype=np.int64, usecols=(0, 1), comments=None)

    try:
        weights = np.loadtxt(path, dtype=float, usecols=2)
        for (u, v), w in zip(data, weights):
            if u > n or v > n or u < 1 or v < 1:
                continue
            u -= 1
            v -= 1
            w = abs(w)
            adj[u, v] = w
            adj[v, u] = w
    except:
        for u, v in data:
            if u > n or v > n or u < 1 or v < 1:
                continue
            u -= 1
            v -= 1
            adj[u, v] = 1.0
            adj[v, u] = 1.0

    return adj


if __name__ == "__main__":
    graph_path = "G81.edges" if len(sys.argv) < 2 else sys.argv[1]
    max_steps = 10000 if len(sys.argv) < 3 else int(sys.argv[2])

    adj = load_graph_edgelist(graph_path)
    edges = int(adj.sum() / 2)
    print(f"Graph loaded: 20,000 nodes · {edges:,} edges")
    print(f"Running GravOpt (max {max_steps:,} steps)...\n")

    start = time.time()
    best_global = 0.0
    prev_global = 0.0

    for step in range(1, max_steps + 1):
        _, cut = gravopt_maxcut(adj, max_steps=step)

        if cut > best_global + 1e-8:
            best_global = cut
            ratio = best_global / (adj.sum() / 2)
            t = time.time() - start
            print(f"NEW BEST → Step {step:4d} | Cut {cut:8.1f} | Ratio {ratio:.6f} | {t:5.1f}s")

        if step <= 50 or step % 100 == 0:
            ratio = best_global / (adj.sum() / 2)
            t = time.time() - start
            print(f"Step {step:4d} | Best {best_global:8.1f} | Ratio {ratio:.6f} | {t:5.1f}s")

        if step > 1200:
            if (best_global - prev_global) / (best_global + 1e-12) < 1e-6:
                print(f"\nEarly stopping at step {step} – no improvement")
                break
        prev_global = best_global

    final_ratio = best_global / (adj.sum() / 2)
    total_time = time.time() - start
    print("\n" + "=" * 68)
    print(f"FINISHED | Best cut: {best_global:8.1f} | Ratio: {final_ratio:.6f}")
    print(f"Total time: {total_time:.1f} seconds")
    print("=" * 68)
