import numpy as np
from numba import njit, prange
import time
import sys

@njit(parallel=True, fastmath=True)
def gravopt_maxcut(adj, max_steps=2000, early_stop_steps=80, patience_thr=1e-6):
    n = adj.shape[0]
    # random initial partition
    x = np.random.choice([-1.0, 1.0], size=n)
    
    best_cut = 0.0
    best_x = x.copy()
    no_improve_counter = 0
    
    for step in range(max_steps):
        cut = 0.0
        for i in prange(n):
            for j in prange(i+1, n):
                if adj[i,j] != 0:
                    cut += adj[i,j] * (1.0 - x[i] * x[j])
        cut *= 0.5
        
        if cut > best_cut + 1e-8:
            best_cut = cut
            best_x = x.copy()
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            
        # early stopping
        if no_improve_counter >= early_stop_steps and step > 1000:
            improv_per_step = patience_thr
            if best_cut > 0:
                # rough estimate of relative improvement
                improv_per_step = (best_cut - prev_best) / best_cut
            if improv_per_step < patience_thr:
                break
                
        prev_best = best_cut
        
        # === GravOpt core update ===
        force = np.zeros(n)
        for i in prange(n):
            f = 0.0
            for j in prange(n):
                if adj[i,j] != 0:
                    # gravitational-like force: pushes node to the opposite side of bad neighbors
                    f += adj[i,j] * x[j] * (1.0 - x[i] * x[j])  # only when edge is cut badly
            force[i] = f
            
        # adaptive step size (bigger early, smaller later)
        alpha = 0.995 ** step
        
        # momentum-style update
        x = np.sign(x + alpha * force)
        # avoid all-zero (should never happen but safety)
        if np.all(x == 0):
            x[np.random.randint(0, n)] = 1.0
            
    return best_x, best_cut

def load_graph_edgelist(path):
    edges = np.loadtxt(path, dtype=int)
    n = edges.max() + 1
    adj = np.zeros((n, n), dtype=np.float32)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gravopt.py graph.edges [max_steps]")
        sys.exit(1)
        
    adj = load_graph_edgelist(sys.argv[1])
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    
    print(f"Graph: {adj.shape[0]} nodes, {int(adj.sum()/2)} edges")
    start = time.time()
    partition, cut = gravopt_maxcut(adj, max_steps=max_steps)
    elapsed = time.time() - start
    
    total_weight = adj.sum() / 2
    ratio = cut / total_weight if total_weight > 0 else 0
    
    print(f"Cut value: {cut:.2f}")
    print(f"Ratio: {ratio:.6f}")
    print(f"Time: {elapsed:.2f} sec")
    print(f"Steps completed: {time.time() - start:.2f} sec total")