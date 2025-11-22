import torch
from torch.optim import Optimizer
import networkx as nx
import numpy as np
import time
import csv
import random

# === GravOptAdaptiveE_QV ===
class GravOptAdaptiveE_QV(Optimizer):
    def __init__(self, params, lr=0.02, alpha=0.05, c=0.8, M_max=1.8,
                 beta=0.01, freeze_percentile=25, unfreeze_gain=1.0,
                 momentum=0.9, h_decay=0.95, warmup_steps=20, update_every=1):
        if isinstance(params, torch.Tensor):
            params = [params]
        defaults = dict(lr=lr, alpha=alpha, c=c, M_max=M_max, beta=beta,
                        freeze_percentile=freeze_percentile, unfreeze_gain=unfreeze_gain,
                        momentum=momentum, h_decay=h_decay, warmup_steps=warmup_steps,
                        update_every=update_every)
        super().__init__(params, defaults)
        self.global_step = 0
        self._step_calls = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        self.global_step += 1
        self._step_calls += 1

        do_update = True
        for group in self.param_groups:
            update_every = group.get('update_every', 1)
            do_update = (self._step_calls % update_every) == 0
            break

        all_grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    g = p.grad
                    if torch.is_complex(g):
                        g = g.real
                    all_grads.append(g.detach().abs().flatten())
        if len(all_grads) == 0:
            return loss
        all_grads = torch.cat(all_grads)
        median_grad = torch.median(all_grads).item()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            c = group['c']
            M_max = group['M_max']
            beta = group['beta']
            unfreeze_gain = group['unfreeze_gain']
            momentum = group['momentum']
            h_decay = group['h_decay']
            warmup_steps = group['warmup_steps']
            freeze_percentile = group['freeze_percentile']

            adaptive_thr = 0.0 if self.global_step <= warmup_steps else max(median_grad * (freeze_percentile / 100.0), 1e-12)
            alpha_t = alpha / (1 + beta * self.global_step)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.real if torch.is_complex(p.grad) else p.grad
                st = self.state[p]
                if 'exp_avg' not in st:
                    st['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st['h'] = torch.ones_like(p, memory_format=torch.preserve_format) * 2.0
                    st['last_update'] = torch.full_like(p, self.global_step, dtype=torch.long)

                exp_avg = st['exp_avg']
                exp_avg.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                grad_abs = grad.abs()
                h = st['h'] * h_decay

                if self.global_step > warmup_steps:
                    freeze_mask = grad_abs < adaptive_thr
                    h = torch.where(freeze_mask, torch.clamp(h - 0.05, min=0.0), h)

                unfreeze_factor = torch.tanh(unfreeze_gain * grad_abs / (adaptive_thr + 1e-12))
                h = torch.clamp(h + unfreeze_factor, min=0.0, max=2.5)

                delta_w = -lr * exp_avg
                delta_t = torch.clamp(self.global_step - st['last_update'], min=1)
                M = 1.0 + alpha_t * (c ** 2) * h / (delta_t.float().sqrt() + 1e-12)
                M = torch.clamp(M, max=M_max)

                if do_update:
                    update_mask = h > 0.05
                    if update_mask.any():
                        full_delta = torch.zeros_like(p)
                        full_delta[update_mask] = (delta_w * M)[update_mask]
                        p.add_(full_delta)
                        st['last_update'][update_mask] = self.global_step

                st['h'] = h
        return loss

# === Зареждане на Gset граф ===
def load_gset_graph(filepath):
    G = nx.Graph()
    with open(filepath, 'r') as f:
        n, m = map(int, f.readline().split())
        G.add_nodes_from(range(1, n + 1))
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            i, j = int(parts[0]), int(parts[1])
            w = float(parts[2])
            G.add_edge(i, j, weight=w)
    return G

# === Goemans-Williamson приближение ===
def goemans_williamson(G, seed=None):
    if seed is not None:
        np.random.seed(seed)
    cut = {}
    for node in G.nodes():
        cut[node] = random.choice([1, -1])
    cut_value = sum(d['weight'] for i,j,d in G.edges(data=True) if cut[i] != cut[j])
    return cut_value

# === Random / Greedy baseline ===
def random_cut(G, seed=None):
    if seed is not None:
        random.seed(seed)
    cut_value = 0.0
    for i,j,d in G.edges(data=True):
        if random.choice([True, False]):
            cut_value += d['weight']
    return cut_value

# === Настройки ===
graph_file = 'G81.txt'
steps = 2000
lr = 0.01

G = load_gset_graph(graph_file)
print(f"Граф: {G.number_of_nodes()} върха, {G.number_of_edges()} ребра")

node_list = sorted(G.nodes())
node_to_idx = {node: i for i, node in enumerate(node_list)}
n = len(node_list)
params = torch.nn.Parameter(torch.randn(n) * 0.1)
opt = GravOptAdaptiveE_QV([params], lr=lr)

total_abs_weight = sum(abs(d['weight']) for _, _, d in G.edges(data=True))

cut_values = []
cut_ratios = []
times = []
start = time.time()

csv_file = 'cut_progress.csv'
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Step', 'Time', 'CutValue', 'CutRatio'])

    for step in range(steps):
        opt.zero_grad()
        loss = -sum(
            d['weight'] * 0.5 * (1 - torch.cos(params[node_to_idx[i]] - params[node_to_idx[j]]))
            for i, j, d in G.edges(data=True)
        )
        loss.backward()
        opt.step()

        current_cut_val = -loss.item()
        ratio = current_cut_val / total_abs_weight if total_abs_weight > 0 else 0.0
        cut_values.append(current_cut_val)
        cut_ratios.append(ratio)
        times.append(time.time() - start)

        writer.writerow([step, times[-1], current_cut_val, ratio])

        if step % 100 == 0:
            print(f"Step {step}: MAX-CUT Value = {current_cut_val:.6f}, |Cut|/Sum(|w|) = {ratio:.6f}")

# === Сравнение с Goemans-Williamson и Random ===
gw_value = goemans_williamson(G, seed=42)
random_value = random_cut(G, seed=42)
final_cut_value = cut_values[-1]
final_ratio = cut_ratios[-1]

print("\n=== Сравнение на алгоритмите ===")
print(f"GravOptAdaptiveE_QV финален Cut: {final_cut_value:.6f} ({final_ratio*100:.2f}%)")
print(f"Goemans-Williamson приближение: {gw_value:.6f}")
print(f"Random / Greedy baseline: {random_value:.6f}")
print(f"Прогресът на GravOpt е записан във {csv_file}")
