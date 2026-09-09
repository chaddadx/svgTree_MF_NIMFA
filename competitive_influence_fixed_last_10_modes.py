import os
import copy
import csv
import time
import math
import random
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from tabulate import tabulate

warnings.filterwarnings('ignore')

# ============================================================
# IMPORTANT TIME CONVENTION
# ============================================================
# We store/plot the network state at the START of each step t.
# Nodes active at time t are the only ones that influence during step t.
# After step t, they become silent and appear silent at time t+1.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class ExperimentConfig:
    N: int = 1000
    T: int = 8
    num_iterations: int = 100

    # Target average degree used for Tree + MFA baseline
    k_target: float = 5

    gamma1: float = 0.25
    gamma2: float = 0.20
    alpha12: float = 0.10
    alpha21: float = 0.15

    # Default: one initially active node per leader, but sweeps may overwrite it.
    m1: int = 1
    m2: int = 1

    random_seeds: bool = True
    use_fixed_k: bool = True

    graph_model: str = "ER"
    graph_params: dict = field(default_factory=dict)

    output_dir: str = "outputs_competitive_influence"
    save_figures: bool = False
    plot_network_states: bool = True
    plot_single_run_summary: bool = True
    verbose_tables: bool = True

    base_seed_graph: int = 42
    base_seed_rng: int = 12345

    # Dedicated lattice / NIMFA visualization setup
    lattice_demo_N: int = 100
    lattice_demo_T: int = 6
    lattice_demo_rows: int = 108
    lattice_demo_cols: int = 10
    


CONFIG = ExperimentConfig()


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clone_cfg(cfg: ExperimentConfig) -> ExperimentConfig:
    return copy.deepcopy(cfg)


def safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else 0.0


def write_csv(path: str, rows: List[dict]):
    if not rows:
        return
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(path: str, matrix: np.ndarray, row_label: str = "time", col_prefix: str = "node"):
    ensure_dir(os.path.dirname(path) or ".")
    T, N = matrix.shape
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row_label] + [f"{col_prefix}_{j}" for j in range(N)])
        for t in range(T):
            writer.writerow([t + 1] + list(map(float, matrix[t])))


def _fix_sizes_to_sum_N(sizes: List[int], N: int) -> List[int]:
    if not sizes:
        return [N]
    sizes = [int(max(1, s)) for s in sizes]
    ssum = sum(sizes)
    if ssum == N:
        return sizes
    scaled = [max(1, int(round(s * N / ssum))) for s in sizes]
    diff = N - sum(scaled)
    scaled[-1] = max(1, scaled[-1] + diff)
    return scaled


def _build_sbm_probs(sizes: List[int], p_in: float = 0.02, p_out: float = 0.001) -> List[List[float]]:
    B = len(sizes)
    p_in = float(np.clip(p_in, 0.0, 1.0))
    p_out = float(np.clip(p_out, 0.0, 1.0))
    return [[(p_in if i == j else p_out) for j in range(B)] for i in range(B)]


def _simple_graph(G: nx.Graph) -> nx.Graph:
    H = nx.Graph(G)
    H.remove_edges_from(nx.selfloop_edges(H))
    return H


def _best_divisor_near_target(N: int, target: int) -> int:
    target = max(2, min(target, N))
    divisors = []
    for d in range(2, int(math.sqrt(N)) + 1):
        if N % d == 0:
            divisors.append(d)
            if d != N // d:
                divisors.append(N // d)
    if not divisors:
        return max(2, min(target, N))
    return min(divisors, key=lambda d: abs(d - target))


def _grid_dims(N: int, rows: Optional[int] = None, cols: Optional[int] = None) -> Tuple[int, int]:
    if rows is not None and cols is not None and rows * cols == N:
        return rows, cols
    best_pair = (1, N)
    best_gap = N - 1
    for r in range(1, int(math.sqrt(N)) + 1):
        if N % r == 0:
            c = N // r
            gap = abs(c - r)
            if gap < best_gap:
                best_gap = gap
                best_pair = (r, c)
    return best_pair


def build_lattice_graph(N: int, rows: Optional[int] = None, cols: Optional[int] = None,
                        periodic: bool = False) -> nx.Graph:
    rows, cols = _grid_dims(N, rows, cols)
    G = nx.grid_2d_graph(rows, cols, periodic=periodic)
    H = nx.convert_node_labels_to_integers(G, ordering="sorted", label_attribute="coord")
    pos = {}
    for n, data in H.nodes(data=True):
        r, c = data["coord"]
        pos[n] = (c, -r)
    H.graph["rows"] = rows
    H.graph["cols"] = cols
    H.graph["pos"] = pos
    H.graph["graph_model"] = "LATTICE"
    return H


def get_graph_positions(G: nx.Graph, graph_model: str, seed: int = 42) -> Dict[int, Tuple[float, float]]:
    gm = graph_model.upper()
    if gm == "LATTICE" and "pos" in G.graph:
        return G.graph["pos"]
    nodes = list(G.nodes())
    if not nodes:
        return {}
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    lcc = comps[0]
    H = G.subgraph(lcc).copy()
    pos_lcc = nx.spring_layout(H, seed=seed, iterations=800)
    xs = np.array([pos_lcc[n][0] for n in lcc])
    ys = np.array([pos_lcc[n][1] for n in lcc])
    scale = max(np.max(np.abs(xs)), np.max(np.abs(ys)), 1e-9)
    for n in lcc:
        pos_lcc[n] = (pos_lcc[n][0] / scale, pos_lcc[n][1] / scale)
    pos = dict(pos_lcc)
    others = [n for n in nodes if n not in lcc]
    if others:
        R = 2.6
        for i, n in enumerate(others):
            ang = 2 * np.pi * i / max(1, len(others))
            pos[n] = (R * np.cos(ang), R * np.sin(ang))
    return pos


def lattice_seed_nodes_from_positions(G: nx.Graph) -> Tuple[List[int], List[int]]:
    rows = int(G.graph.get("rows", 1))
    cols = int(G.graph.get("cols", G.number_of_nodes()))
    target1 = (rows // 2, 0)
    target2 = (rows // 2, cols - 1)
    p1 = []
    p2 = []
    for n, data in G.nodes(data=True):
        coord = data.get("coord", None)
        if coord == target1:
            p1.append(n)
        if coord == target2:
            p2.append(n)
    if not p1:
        p1 = [0]
    if not p2:
        p2 = [G.number_of_nodes() - 1]
    return p1, p2


def matrix_with_diag(gamma: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    B = np.array(alpha, dtype=float, copy=True)
    np.fill_diagonal(B, gamma)
    return B


def validate_multi_params(gamma: np.ndarray, alpha: np.ndarray) -> Tuple[bool, str]:
    H = len(gamma)
    if alpha.shape != (H, H):
        return False, "alpha must be an H x H matrix"
    if np.any(gamma < 0) or np.any(alpha < 0):
        return False, "all gamma and alpha entries must be nonnegative"
    if np.any(gamma > 1) or np.any(alpha > 1):
        return False, "all gamma and alpha entries must be at most 1"
    if np.any(np.abs(np.diag(alpha)) > 1e-12):
        return False, "alpha diagonal must be zero"
    row_sums = gamma + alpha.sum(axis=1)
    if np.any(row_sums > 1 + 1e-12):
        bad = np.where(row_sums > 1 + 1e-12)[0][0]
        return False, f"row {bad + 1} violates gamma_h + sum_ell alpha_h,ell <= 1"
    return True, "ok"


def random_feasible_multi_params(H: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    gamma = np.zeros(H, dtype=float)
    alpha = np.zeros((H, H), dtype=float)
    for h in range(H):
        total = float(rng.uniform(0.20, 0.90))
        weights = rng.dirichlet(np.ones(H))
        gamma[h] = total * weights[0]
        others = [ell for ell in range(H) if ell != h]
        other_weights = weights[1:]
        other_weights = other_weights / other_weights.sum() if other_weights.sum() > 0 else np.ones(H - 1) / (H - 1)
        for idx, ell in enumerate(others):
            alpha[h, ell] = total * other_weights[idx]
    return gamma, alpha


def parse_float_list(prompt: str, expected_len: int) -> List[float]:
    while True:
        raw = input(prompt).strip()
        try:
            vals = [float(x.strip()) for x in raw.split(',') if x.strip() != ""]
            if len(vals) != expected_len:
                print(f"Please enter exactly {expected_len} comma-separated values.")
                continue
            return vals
        except ValueError:
            print("Could not parse the values. Please try again.")


def prompt_multi_project_parameters(H: int, base_seed_rng: int) -> Tuple[np.ndarray, np.ndarray]:
    print("Choose parameter mode for the multi-project experiment:")
    print("1. Manual heterogeneous parameters (provide all gamma_h and alpha_{h,ell})")
    print("2. Random heterogeneous parameters (automatically generated under feasibility constraints)")
    choice = input("Enter choice (1/2): ").strip()

    if choice == '1':
        while True:
            gamma = np.array(parse_float_list(
                f"Enter gamma values as {H} comma-separated numbers: ", H
            ), dtype=float)
            alpha = np.zeros((H, H), dtype=float)
            print("Now enter the alpha matrix row by row.")
            print("For each source project h, give H comma-separated values for alpha_{h,ell}.")
            print("The diagonal alpha_{h,h} will be forced to 0.")
            for h in range(H):
                row = np.array(parse_float_list(
                    f"Row {h + 1} of alpha: ", H
                ), dtype=float)
                row[h] = 0.0
                alpha[h, :] = row
            ok, msg = validate_multi_params(gamma, alpha)
            if ok:
                return gamma, alpha
            print(f"Invalid parameters: {msg}")
            print("Please enter them again.\n")

    rng = np.random.default_rng(base_seed_rng)
    gamma, alpha = random_feasible_multi_params(H, rng)
    print("Random heterogeneous parameters generated:")
    print("gamma =", np.round(gamma, 4))
    print("alpha =")
    print(np.round(alpha, 4))
    return gamma, alpha


def pretty_print_multi_params(gamma: np.ndarray, alpha: np.ndarray):
    H = len(gamma)
    print("\nMULTI-PROJECT PARAMETERS")
    print("gamma:")
    print(np.round(gamma, 4))
    print("alpha matrix (row h = source project h, column ell = adopted project ell):")
    print(np.round(alpha, 4))
    rows = []
    for h in range(H):
        rows.append([h + 1, round(float(gamma[h]), 4), round(float(alpha[h].sum()), 4), round(float(gamma[h] + alpha[h].sum()), 4)])
    print(tabulate(rows, headers=["source project", "gamma_h", "sum alpha_h,*", "row total"], tablefmt="grid"))


# ============================================================
# GRAPH FACTORY
# ============================================================

def generate_graph(graph_model: str, N: int, p: float, iteration_seed: int,
                   graph_params: Optional[dict] = None) -> nx.Graph:
    graph_params = graph_params or {}
    gm = graph_model.upper()
    seed = iteration_seed

    if gm == "ER":
        return nx.erdos_renyi_graph(N, p, seed=seed)

    if gm == "WS":
        k_ws = int(graph_params.get("k_ws", 6))
        beta = float(graph_params.get("beta", 0.2))
        if k_ws >= N:
            k_ws = max(2, N - 1)
        if k_ws % 2 == 1:
            k_ws += 1
            if k_ws >= N:
                k_ws = max(2, N - 2)
        return nx.watts_strogatz_graph(N, k_ws, beta, seed=seed)

    if gm == "BA":
        m = int(graph_params.get("m", 3))
        m = max(1, min(m, N - 1))
        return nx.barabasi_albert_graph(N, m, seed=seed)

    if gm == "RR":
        d = int(graph_params.get("d", 6))
        d = max(0, min(d, N - 1))
        if (N * d) % 2 == 1:
            d = max(0, d - 1)
        return nx.random_regular_graph(d, N, seed=seed)

    if gm == "SBM":
        sizes = graph_params.get("sizes", [N // 2, N - N // 2])
        sizes = _fix_sizes_to_sum_N(list(sizes), N)
        probs = graph_params.get("probs", None)
        if probs is None:
            p_in = float(graph_params.get("p_in", 0.02))
            p_out = float(graph_params.get("p_out", 0.001))
            probs = _build_sbm_probs(sizes, p_in=p_in, p_out=p_out)
        return nx.stochastic_block_model(sizes, probs, seed=seed)

    if gm == "HK":
        m = int(graph_params.get("m", max(1, int(round(max(2.0, p * N) / 2)))))
        triad_p = float(graph_params.get("triad_p", 0.35))
        m = max(1, min(m, N - 1))
        return nx.powerlaw_cluster_graph(N, m, triad_p, seed=seed)

    if gm == "RCAV":
        target_clique = int(graph_params.get("clique_size", max(3, int(round(p * N)) + 1)))
        k_clique = _best_divisor_near_target(N, target_clique)
        l_cliques = max(2, N // k_clique)
        rewire_p = float(graph_params.get("rewire_p", 0.08))
        G = nx.relaxed_caveman_graph(l_cliques, k_clique, rewire_p, seed=seed)
        return _simple_graph(G)

    if gm == "LATTICE":
        rows = graph_params.get("rows", None)
        cols = graph_params.get("cols", None)
        periodic = bool(graph_params.get("periodic", False))
        return build_lattice_graph(N, rows=rows, cols=cols, periodic=periodic)

    raise ValueError(f"Unknown graph_model={graph_model}")


def expected_degree_for_models(graph_model: str, N: int, p: float,
                               graph_params: Optional[dict] = None) -> float:
    graph_params = graph_params or {}
    gm = graph_model.upper()

    if gm == "ER":
        return p * (N - 1)
    if gm == "WS":
        return float(graph_params.get("k_ws", 6))
    if gm == "BA":
        return float(2 * int(graph_params.get("m", 3)))
    if gm == "RR":
        return float(graph_params.get("d", 6))
    if gm == "SBM":
        sizes = graph_params.get("sizes", [N // 2, N - N // 2])
        sizes = _fix_sizes_to_sum_N(list(sizes), N)
        probs = graph_params.get("probs", None)
        if probs is None:
            p_in = float(graph_params.get("p_in", 0.02))
            p_out = float(graph_params.get("p_out", 0.001))
            probs = _build_sbm_probs(sizes, p_in=p_in, p_out=p_out)
        B = len(sizes)
        exp_deg_per_block = []
        for b in range(B):
            deg_b = 0.0
            for b2 in range(B):
                pij = float(probs[b][b2])
                if b2 == b:
                    deg_b += pij * max(0, sizes[b2] - 1)
                else:
                    deg_b += pij * sizes[b2]
            exp_deg_per_block.append(deg_b)
        return float(sum(exp_deg_per_block[b] * sizes[b] for b in range(B)) / max(1, sum(sizes)))
    if gm == "HK":
        return float(2 * int(graph_params.get("m", 3)))
    if gm == "RCAV":
        k_clique = _best_divisor_near_target(N, int(graph_params.get("clique_size", 6)))
        return float(max(1, k_clique - 1))
    if gm == "LATTICE":
        rows = graph_params.get("rows", None)
        cols = graph_params.get("cols", None)
        rows, cols = _grid_dims(N, rows=rows, cols=cols)
        if rows == 1:
            return float(2 * (cols - 1) / max(1, cols))
        if cols == 1:
            return float(2 * (rows - 1) / max(1, rows))
        n_corner = 4
        n_edge = 2 * (rows - 2) + 2 * (cols - 2)
        n_inner = max(0, N - n_corner - n_edge)
        total_deg = 2 * n_corner + 3 * n_edge + 4 * n_inner
        return float(total_deg / N)
    raise ValueError(f"Unknown graph_model={graph_model}")


def sbm_block_of_node(i: int, sizes: List[int]) -> int:
    s = 0
    for b, sz in enumerate(sizes):
        if i < s + sz:
            return b
        s += sz
    return len(sizes) - 1


def print_graph_stats(G: nx.Graph, graph_model: str, graph_params: dict):
    N = G.number_of_nodes()
    E = G.number_of_edges()
    degs = np.array([d for _, d in G.degree()], dtype=float)

    print("" + "=" * 90)
    print("GRAPH STATS (last iteration graph)")
    print("=" * 90)
    print(f"Graph model={graph_model} | params={graph_params}")
    print(f"N={N}, E={E}, avg_degree={2 * E / max(1, N):.3f}")
    print(f"degree: min={degs.min():.0f}, mean={degs.mean():.3f}, median={np.median(degs):.0f}, max={degs.max():.0f}")
    print(f"connected components: {nx.number_connected_components(G)} | LCC size: {len(max(nx.connected_components(G), key=len))}")
    print(f"clustering coefficient (avg): {nx.average_clustering(G):.4f}")

    sizes = (graph_params or {}).get("sizes", None)
    if graph_model.upper() != "SBM" or sizes is None:
        return

    sizes = _fix_sizes_to_sum_N(list(sizes), N)
    B = len(sizes)
    blocks = np.array([sbm_block_of_node(i, sizes) for i in range(N)], dtype=int)

    print("" + "-" * 90)
    print("SBM COMMUNITY STATS")
    print("-" * 90)
    print("block | size | avg_degree | min_deg | max_deg")
    for b in range(B):
        idx = np.where(blocks == b)[0]
        bd = degs[idx]
        print(f"{b:>5} | {len(idx):>4} | {bd.mean():>10.3f} | {bd.min():>7.0f} | {bd.max():>7.0f}")
    print("NOTE: on SBM graphs, MFA can sometimes outperform NIMFA because the node-level NIMFA closure")
    print("      ignores neighbor-state correlations and uses a ratio-of-expectations approximation.")
    print("      When the graph is nearly block-homogeneous, the scalar/block-average behavior can match")
    print("      Monte Carlo better than the nodewise independence closure.")


# ============================================================
# TWO-LEADER MODEL
# ============================================================

class StochasticInfluenceModel:
    def __init__(self, N: int, p: float, T: int, gamma1: float, gamma2: float,
                 alpha12: float, alpha21: float, m1: int, m2: int,
                 random_seeds: bool, iteration: int, graph_model: str = "ER",
                 graph_params: Optional[dict] = None, base_seed_graph: int = 42,
                 fixed_graph: Optional[nx.Graph] = None):
        self.original_N = int(N)
        self.p = float(p)
        self.T = int(T)
        self.gamma1 = float(gamma1)
        self.gamma2 = float(gamma2)
        self.alpha12 = float(alpha12)
        self.alpha21 = float(alpha21)
        self.m1 = int(max(1, m1))
        self.m2 = int(max(1, m2))
        self.random_seeds = bool(random_seeds)
        self.iteration = int(iteration)
        self.graph_model = str(graph_model)
        self.graph_params = graph_params or {}
        self.base_seed_graph = int(base_seed_graph)
        # Optional fixed graph used by controlled what-if experiments.
        # When provided, Monte Carlo repetitions reuse exactly the same network.
        self.fixed_graph = fixed_graph

        self.G = None
        self.N = self.original_N
        self.states = None
        self.history = []
        self.seed_nodes_p1 = []
        self.seed_nodes_p2 = []
        self.seed_nodes = []
        self.initialize_graph()

    def initialize_graph(self):
        n_required = self.m1 + self.m2

        def pick_component_with_n_nodes(G, n_need):
            comps = sorted(nx.connected_components(G), key=len, reverse=True)
            for cc in comps:
                if len(cc) >= n_need:
                    return sorted(list(cc))
            return None

        if self.fixed_graph is None:
            self.G = generate_graph(self.graph_model, self.original_N, self.p,
                                    self.base_seed_graph + self.iteration, self.graph_params)
            self.G = _simple_graph(self.G)
        else:
            # Reuse one realized network so that a what-if experiment changes
            # only the quantity being studied (seed position or one parameter).
            self.G = _simple_graph(self.fixed_graph.copy())
        self.N = self.G.number_of_nodes()
        self.states = np.zeros(self.N, dtype=int)

        comp = pick_component_with_n_nodes(self.G, n_required)
        if comp is None and self.fixed_graph is None:
            for tries in range(60):
                self.G = generate_graph(self.graph_model, self.original_N, self.p,
                                        10000 + self.iteration * 97 + tries, self.graph_params)
                self.G = _simple_graph(self.G)
                self.N = self.G.number_of_nodes()
                self.states = np.zeros(self.N, dtype=int)
                comp = pick_component_with_n_nodes(self.G, n_required)
                if comp is not None:
                    break

        if comp is None:
            nodes = list(range(self.N))
            if self.N < n_required:
                raise ValueError("Not enough nodes for initial active nodes")
            chosen = random.sample(nodes, n_required)
        else:
            chosen = random.sample(comp, n_required)

        self.seed_nodes_p1 = chosen[:self.m1]
        self.seed_nodes_p2 = chosen[self.m1:self.m1 + self.m2]
        self.seed_nodes = self.seed_nodes_p1 + self.seed_nodes_p2
        self.reset_states_from_seeds()

    def reset_states_from_seeds(self):
        self.states = np.zeros(self.N, dtype=int)
        self.states[self.seed_nodes_p1] = 1
        self.states[self.seed_nodes_p2] = 2
        self.history = []

    def set_manual_seeds(self, seed_nodes_p1: List[int], seed_nodes_p2: List[int]):
        self.seed_nodes_p1 = list(seed_nodes_p1)
        self.seed_nodes_p2 = list(seed_nodes_p2)
        self.seed_nodes = self.seed_nodes_p1 + self.seed_nodes_p2
        self.m1 = len(self.seed_nodes_p1)
        self.m2 = len(self.seed_nodes_p2)
        self.reset_states_from_seeds()

    def get_inactive_neighbors(self, node: int) -> List[int]:
        return [n for n in self.G.neighbors(node) if self.states[n] == 0]

    def influence_neighbor(self, active_node: int) -> int:
        s = self.states[active_node]
        u = random.random()
        if s == 1:
            if u < self.gamma1:
                return 1
            elif u < self.gamma1 + self.alpha12:
                return 2
            return 0
        if s == 2:
            if u < self.gamma2:
                return 2
            elif u < self.gamma2 + self.alpha21:
                return 1
            return 0
        return 0

    def update_step(self):
        new_states = self.states.copy()
        active_nodes = np.where((self.states == 1) | (self.states == 2))[0]
        influence_attempts = {}
        for a in active_nodes:
            for nb in self.get_inactive_neighbors(a):
                influence_attempts.setdefault(nb, []).append(a)
        for x, attackers in influence_attempts.items():
            if self.states[x] == 0:
                chosen = random.choice(attackers)
                new_states[x] = self.influence_neighbor(chosen)
        for a in active_nodes:
            new_states[a] = -1 if self.states[a] == 1 else -2
        self.states = new_states

    def run_simulation(self):
        results = []
        self.history = []
        for t in range(1, self.T + 1):
            st = self.states.copy()
            self.history.append(st)
            a1 = int(np.sum(st == 1))
            a2 = int(np.sum(st == 2))
            s1 = int(np.sum(st == -1))
            s2 = int(np.sum(st == -2))
            results.append({
                'time': t,
                'active1': a1,
                'active2': a2,
                'cumulative1': a1 + s1,
                'cumulative2': a2 + s2,
            })
            if t < self.T:
                self.update_step()
        return results

    def compute_tree_model(self, fixed_k=None):
        k = float(fixed_k) if fixed_k is not None else (2 * self.G.number_of_edges() / max(1, self.G.number_of_nodes()))

        total_seeds = max(1, self.m1 + self.m2)
        share1 = self.m1 / total_seeds
        N1 = max(float(self.m1), round(self.N * share1))
        N2 = max(float(self.m2), self.N - N1)
        if N1 + N2 > self.N:
            excess = N1 + N2 - self.N
            if N2 - excess >= self.m2:
                N2 -= excess
            else:
                N1 -= excess
        N1 = float(max(self.m1, N1))
        N2 = float(max(self.m2, self.N - N1))

        A11, A21, C11, C21 = [float(self.m1)], [0.0], [float(self.m1)], [0.0]
        A12, A22, C12, C22 = [0.0], [float(self.m2)], [0.0], [float(self.m2)]
        used1, used2 = float(self.m1), float(self.m2)
        results = []

        for t in range(1, self.T + 1):
            new_X1_finite = A11[t - 1] + A12[t - 1]
            new_X2_finite = A21[t - 1] + A22[t - 1]
            cum_X1_finite = C11[t - 1] + C12[t - 1]
            cum_X2_finite = C21[t - 1] + C22[t - 1]
            results.append({
                'time': t,
                'new_X1_finite': new_X1_finite,
                'new_X2_finite': new_X2_finite,
                'cum_X1_finite': cum_X1_finite,
                'cum_X2_finite': cum_X2_finite,
            })
            if t == self.T:
                break

            def pool_step(a1, a2, used, Npool):
                active_total = a1 + a2
                remaining = max(0.0, Npool - used)
                contacts = min(remaining, k * active_total) if active_total > 0 else 0.0
                if contacts <= 0.0 or active_total <= 0.0:
                    return 0.0, 0.0, used
                c_from_p1 = contacts * (a1 / active_total)
                c_from_p2 = contacts - c_from_p1
                next_a1 = c_from_p1 * self.gamma1 + c_from_p2 * self.alpha21
                next_a2 = c_from_p1 * self.alpha12 + c_from_p2 * self.gamma2
                return next_a1, next_a2, used + contacts

            na11, na21, used1 = pool_step(A11[-1], A21[-1], used1, N1)
            A11.append(na11)
            A21.append(na21)
            C11.append(C11[-1] + na11)
            C21.append(C21[-1] + na21)

            na12, na22, used2 = pool_step(A12[-1], A22[-1], used2, N2)
            A12.append(na12)
            A22.append(na22)
            C12.append(C12[-1] + na12)
            C22.append(C22[-1] + na22)

        return results, k

    def compute_mean_field_model(self, fixed_k=None):
        k = float(fixed_k) if fixed_k is not None else (2 * self.G.number_of_edges() / max(1, self.G.number_of_nodes()))
        a1 = [self.m1 / self.N]
        a2 = [self.m2 / self.N]
        results = [{
            'time': 1,
            'new_A1': float(self.m1),
            'new_A2': float(self.m2),
            'cum_A1': float(self.m1),
            'cum_A2': float(self.m2),
        }]

        for t in range(2, self.T + 1):
            cum_a1 = sum(a1)
            cum_a2 = sum(a2)
            I = max(0.0, 1.0 - (cum_a1 + cum_a2))
            a1_t = a1[-1]
            a2_t = a2[-1]
            total_new = a1_t + a2_t
            contact_prob = 1 - (1 - total_new) ** k if total_new > 0 else 0.0

            if total_new > 0:
                p1 = (a1_t / total_new) * self.gamma1 + (a2_t / total_new) * self.alpha21
                p2 = (a1_t / total_new) * self.alpha12 + (a2_t / total_new) * self.gamma2
            else:
                p1 = 0.0
                p2 = 0.0

            new_a1 = I * contact_prob * p1
            new_a2 = I * contact_prob * p2
            a1.append(new_a1)
            a2.append(new_a2)

            results.append({
                'time': t,
                'new_A1': new_a1 * self.N,
                'new_A2': new_a2 * self.N,
                'cum_A1': sum(x * self.N for x in a1),
                'cum_A2': sum(x * self.N for x in a2),
            })
        return results, k

    def compute_nimfa_model(self):
        results, _ = self.compute_nimfa_full(return_histories=False)
        return results

    def compute_nimfa_full(self, return_histories: bool = True):
        N = self.N
        neigh = [list(self.G.neighbors(i)) for i in range(N)]

        M1 = np.zeros(N, dtype=float)
        M2 = np.zeros(N, dtype=float)
        M1[self.seed_nodes_p1] = 1.0
        M2[self.seed_nodes_p2] = 1.0
        cum1 = M1.copy()
        cum2 = M2.copy()

        results = [{
            'time': 1,
            'new_N1': float(M1.sum()),
            'new_N2': float(M2.sum()),
            'cum_N1': float(cum1.sum()),
            'cum_N2': float(cum2.sum()),
        }]

        hist = None
        if return_histories:
            hist = {
                'M1': [M1.copy()],
                'M2': [M2.copy()],
                'cum1': [cum1.copy()],
                'cum2': [cum2.copy()],
                'q': [np.zeros(N, dtype=float)],
                'r1': [np.zeros(N, dtype=float)],
                'r2': [np.zeros(N, dtype=float)],
            }

        for t in range(2, self.T + 1):
            active_tot = np.clip(M1 + M2, 0.0, 1.0)
            q = np.zeros(N, dtype=float)
            u1 = np.zeros(N, dtype=float)
            u2 = np.zeros(N, dtype=float)

            for i in range(N):
                Ni = neigh[i]
                if not Ni:
                    q[i] = 0.0
                    continue
                vals = np.clip(1.0 - active_tot[Ni], 0.0, 1.0)
                q[i] = 1.0 - (float(np.prod(vals)) if len(vals) > 0 else 1.0)
                u1[i] = float(np.sum(M1[Ni]))
                u2[i] = float(np.sum(M2[Ni]))

            u = u1 + u2
            r1 = np.divide(u1, u, out=np.zeros_like(u1), where=(u > 0))
            r2 = np.divide(u2, u, out=np.zeros_like(u2), where=(u > 0))
            I = np.clip(1.0 - cum1 - cum2, 0.0, 1.0)

            M1 = I * q * (r1 * self.gamma1 + r2 * self.alpha21)
            M2 = I * q * (r2 * self.gamma2 + r1 * self.alpha12)
            cum1 = np.clip(cum1 + M1, 0.0, 1.0)
            cum2 = np.clip(cum2 + M2, 0.0, 1.0)

            results.append({
                'time': t,
                'new_N1': float(M1.sum()),
                'new_N2': float(M2.sum()),
                'cum_N1': float(cum1.sum()),
                'cum_N2': float(cum2.sum()),
            })

            if return_histories:
                hist['M1'].append(M1.copy())
                hist['M2'].append(M2.copy())
                hist['cum1'].append(cum1.copy())
                hist['cum2'].append(cum2.copy())
                hist['q'].append(q.copy())
                hist['r1'].append(r1.copy())
                hist['r2'].append(r2.copy())

        if return_histories:
            hist = {k: np.vstack(v) for k, v in hist.items()}
        return results, hist


# ============================================================
# GENERIC MULTI-PROJECT APPROXIMATIONS
# ============================================================

def compute_tree_model_H(N: int, T: int, k: float, gamma: np.ndarray, alpha: np.ndarray,
                         m: np.ndarray) -> List[dict]:
    H = len(gamma)
    total_m = float(np.sum(m))
    shares = m / total_m
    pool_sizes = np.maximum(m.astype(float), np.round(N * shares).astype(float))
    diff = float(N - np.sum(pool_sizes))
    pool_sizes[-1] += diff
    pool_sizes = np.maximum(pool_sizes, m.astype(float))

    B = matrix_with_diag(gamma, alpha)

    active_by_pool = []
    cum_by_pool = []
    used = []
    for h in range(H):
        vec = np.zeros(H, dtype=float)
        vec[h] = float(m[h])
        active_by_pool.append([vec.copy()])
        cum_by_pool.append([vec.copy()])
        used.append(float(m[h]))

    results = []
    for t in range(1, T + 1):
        new_total = np.zeros(H, dtype=float)
        cum_total = np.zeros(H, dtype=float)
        for p in range(H):
            new_total += active_by_pool[p][t - 1]
            cum_total += cum_by_pool[p][t - 1]
        row = {'time': t}
        for h in range(H):
            row[f'new_{h + 1}'] = float(new_total[h])
            row[f'cum_{h + 1}'] = float(cum_total[h])
        results.append(row)
        if t == T:
            break

        for p in range(H):
            avec = active_by_pool[p][-1]
            active_total = float(np.sum(avec))
            remaining = max(0.0, float(pool_sizes[p] - used[p]))
            contacts = min(remaining, k * active_total) if active_total > 0 else 0.0
            if contacts <= 0.0 or active_total <= 0.0:
                next_vec = np.zeros(H, dtype=float)
            else:
                contacts_by_source = contacts * (avec / active_total)
                next_vec = contacts_by_source @ B
                used[p] += contacts
            active_by_pool[p].append(next_vec.copy())
            cum_by_pool[p].append(cum_by_pool[p][-1] + next_vec)

    return results


def compute_mf_model_H(N: int, T: int, k: float, gamma: np.ndarray, alpha: np.ndarray,
                       m: np.ndarray) -> List[dict]:
    H = len(gamma)
    B = matrix_with_diag(gamma, alpha)
    a_hist = [m.astype(float) / float(N)]
    cum = a_hist[0].copy()

    results = []
    row = {'time': 1}
    for h in range(H):
        row[f'new_{h + 1}'] = float(m[h])
        row[f'cum_{h + 1}'] = float(m[h])
    results.append(row)

    for t in range(2, T + 1):
        a = a_hist[-1]
        x = float(np.sum(a))
        I = max(0.0, 1.0 - float(np.sum(cum)))
        q = 1 - (1 - x) ** k if x > 0 else 0.0
        if x > 0:
            frac = a / x
            new_a = I * q * (frac @ B)
        else:
            new_a = np.zeros(H, dtype=float)
        a_hist.append(new_a.copy())
        cum = cum + new_a
        row = {'time': t}
        for h in range(H):
            row[f'new_{h + 1}'] = float(new_a[h] * N)
            row[f'cum_{h + 1}'] = float(cum[h] * N)
        results.append(row)

    return results


def compute_nimfa_model_H(G: nx.Graph, T: int, gamma: np.ndarray, alpha: np.ndarray,
                          seed_nodes: List[int]) -> List[dict]:
    H = len(gamma)
    N = G.number_of_nodes()
    B = matrix_with_diag(gamma, alpha)
    neigh = [list(G.neighbors(i)) for i in range(N)]

    M = np.zeros((N, H), dtype=float)
    for h, node in enumerate(seed_nodes):
        M[node, h] = 1.0
    cum = M.copy()

    results = []
    row = {'time': 1}
    for h in range(H):
        row[f'new_{h + 1}'] = float(np.sum(M[:, h]))
        row[f'cum_{h + 1}'] = float(np.sum(cum[:, h]))
    results.append(row)

    for t in range(2, T + 1):
        p = np.clip(np.sum(M, axis=1), 0.0, 1.0)
        q = np.zeros(N, dtype=float)
        u = np.zeros((N, H), dtype=float)
        for i in range(N):
            Ni = neigh[i]
            if not Ni:
                continue
            vals = np.clip(1.0 - p[Ni], 0.0, 1.0)
            q[i] = 1.0 - (float(np.prod(vals)) if len(vals) > 0 else 1.0)
            for h in range(H):
                u[i, h] = float(np.sum(M[Ni, h]))
        u_tot = np.sum(u, axis=1)
        r = np.divide(u, u_tot[:, None], out=np.zeros_like(u), where=(u_tot[:, None] > 0))
        I = np.clip(1.0 - np.sum(cum, axis=1), 0.0, 1.0)
        M = (I * q)[:, None] * (r @ B)
        cum = np.clip(cum + M, 0.0, 1.0)
        row = {'time': t}
        for h in range(H):
            row[f'new_{h + 1}'] = float(np.sum(M[:, h]))
            row[f'cum_{h + 1}'] = float(np.sum(cum[:, h]))
        results.append(row)

    return results


# ============================================================
# MULTI-PROJECT MODELS (SIMULATION)
# ============================================================

class MultiProjectInfluenceModel:
    def __init__(self, N, p, T, H, gamma, alpha, graph_model="ER", graph_params=None,
                 iteration=0, base_seed_graph=42):
        self.original_N = int(N)
        self.p = float(p)
        self.T = int(T)
        self.H = int(H)
        self.gamma = np.array(gamma, dtype=float)
        self.alpha = np.array(alpha, dtype=float)
        self.graph_model = graph_model
        self.graph_params = graph_params or {}
        self.iteration = int(iteration)
        self.base_seed_graph = int(base_seed_graph)

        self.G = generate_graph(self.graph_model, self.original_N, self.p,
                                self.base_seed_graph + self.iteration, self.graph_params)
        self.G = _simple_graph(self.G)
        self.N = self.G.number_of_nodes()
        self.states = np.zeros(self.N, dtype=int)
        self.history = []
        self.seed_nodes = []
        self.initialize_seeds()

    def initialize_seeds(self):
        comps = sorted(nx.connected_components(self.G), key=len, reverse=True)
        chosen_component = None
        for cc in comps:
            if len(cc) >= self.H:
                chosen_component = sorted(list(cc))
                break
        if chosen_component is None:
            raise ValueError(f"No connected component with at least H={self.H} nodes.")
        chosen = random.sample(chosen_component, self.H)
        self.seed_nodes = chosen
        for h, node in enumerate(chosen, start=1):
            self.states[node] = h

    def get_inactive_neighbors(self, node):
        return [n for n in self.G.neighbors(node) if self.states[n] == 0]

    def influence_from_project(self, project_idx):
        h = project_idx - 1
        u = random.random()
        if u < self.gamma[h]:
            return project_idx
        acc = self.gamma[h]
        for ell in range(self.H):
            if ell == h:
                continue
            acc += self.alpha[h, ell]
            if u < acc:
                return ell + 1
        return 0

    def update_step(self):
        new_states = self.states.copy()
        active_nodes = np.where(self.states > 0)[0]
        influence_attempts = {}
        for a in active_nodes:
            for nb in self.get_inactive_neighbors(a):
                influence_attempts.setdefault(nb, []).append(a)
        for x, attackers in influence_attempts.items():
            if self.states[x] == 0:
                chosen = random.choice(attackers)
                chosen_project = self.states[chosen]
                new_states[x] = self.influence_from_project(chosen_project)
        for a in active_nodes:
            new_states[a] = -self.states[a]
        self.states = new_states

    def run_simulation(self):
        results = []
        self.history = []
        for t in range(1, self.T + 1):
            st = self.states.copy()
            self.history.append(st)
            row = {'time': t}
            for h in range(1, self.H + 1):
                active_h = int(np.sum(st == h))
                silent_h = int(np.sum(st == -h))
                row[f'active_{h}'] = active_h
                row[f'cumulative_{h}'] = active_h + silent_h
            results.append(row)
            if t < self.T:
                self.update_step()
        return results


# ============================================================
# TABLES + ERRORS
# ============================================================

def print_activation_tables(sim_data_avg, tree_data, mf_data, nimfa_data_avg, num_iterations, k_value=None):
    print("" + "=" * 90)
    print(f"ACTIVATION TABLES FOR ALL MODELS (Average of {num_iterations} iterations)")
    if k_value is not None:
        print(f"Using fixed average degree k = {k_value:.2f} (Tree + MF baseline)")
    print("Including NIMFA (node-level mean-field, averaged over runs).")
    print("=" * 90)

    headers = ["t", "Simulation (A)", "Simulation (Cum)", "Tree (A)", "Tree (Cum)", "MF (A)", "MF (Cum)", "NIMFA (A)", "NIMFA (Cum)"]

    print("" + "-" * 90)
    print("LEADER 1 ACTIVATIONS")
    print("-" * 90)
    table_data = []
    for i in range(len(sim_data_avg)):
        t = sim_data_avg[i]['time']
        table_data.append([
            t,
            f"{sim_data_avg[i]['active1_avg']:.2f}",
            f"{sim_data_avg[i]['cumulative1_avg']:.2f}",
            f"{tree_data[i]['new_X1_finite']:.2f}",
            f"{tree_data[i]['cum_X1_finite']:.2f}",
            f"{mf_data[i]['new_A1']:.2f}",
            f"{mf_data[i]['cum_A1']:.2f}",
            f"{nimfa_data_avg[i]['new_N1_avg']:.2f}",
            f"{nimfa_data_avg[i]['cum_N1_avg']:.2f}",
        ])
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    print("" + "-" * 90)
    print("LEADER 2 ACTIVATIONS")
    print("-" * 90)
    table_data = []
    for i in range(len(sim_data_avg)):
        t = sim_data_avg[i]['time']
        table_data.append([
            t,
            f"{sim_data_avg[i]['active2_avg']:.2f}",
            f"{sim_data_avg[i]['cumulative2_avg']:.2f}",
            f"{tree_data[i]['new_X2_finite']:.2f}",
            f"{tree_data[i]['cum_X2_finite']:.2f}",
            f"{mf_data[i]['new_A2']:.2f}",
            f"{mf_data[i]['cum_A2']:.2f}",
            f"{nimfa_data_avg[i]['new_N2_avg']:.2f}",
            f"{nimfa_data_avg[i]['cum_N2_avg']:.2f}",
        ])
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def calculate_error_metrics(sim_data_avg, tree_data, mf_data, nimfa_data_avg, num_iterations):
    print("" + "=" * 90)
    print(f"ERROR METRICS (vs Monte Carlo average, {num_iterations} runs)")
    print("Including NIMFA errors vs simulation.")
    print("=" * 90)

    sim_cum1 = np.array([d['cumulative1_avg'] for d in sim_data_avg], float)
    sim_cum2 = np.array([d['cumulative2_avg'] for d in sim_data_avg], float)
    tree_cum1 = np.array([d['cum_X1_finite'] for d in tree_data], float)
    tree_cum2 = np.array([d['cum_X2_finite'] for d in tree_data], float)
    mf_cum1 = np.array([d['cum_A1'] for d in mf_data], float)
    mf_cum2 = np.array([d['cum_A2'] for d in mf_data], float)
    nimfa_cum1 = np.array([d['cum_N1_avg'] for d in nimfa_data_avg], float)
    nimfa_cum2 = np.array([d['cum_N2_avg'] for d in nimfa_data_avg], float)

    def bounded_rel_error(yhat, y):
        return np.abs(yhat - y) / np.maximum(np.maximum(np.abs(yhat), np.abs(y)), 1e-12)

    def mae_rmse_rel(yhat, y):
        err = np.abs(yhat - y)
        mae = float(err.mean())
        rmse = float(np.sqrt(((yhat - y) ** 2).mean()))
        rel = float(100.0 * bounded_rel_error(yhat, y).mean())
        return mae, rmse, rel

    def final_abs_rel(yhat, y):
        ae = float(abs(yhat[-1] - y[-1]))
        rel = float(100.0 * ae / y[-1])
        return ae, rel

    t_mae_tot, t_rmse_tot, t_rel_tot = mae_rmse_rel(tree_cum1 + tree_cum2, sim_cum1 + sim_cum2)
    mf_mae_tot, mf_rmse_tot, mf_rel_tot = mae_rmse_rel(mf_cum1 + mf_cum2, sim_cum1 + sim_cum2)
    n_mae_tot, n_rmse_tot, n_rel_tot = mae_rmse_rel(nimfa_cum1 + nimfa_cum2, sim_cum1 + sim_cum2)

    t_final_tot_abs, t_final_tot_rel = final_abs_rel(tree_cum1 + tree_cum2, sim_cum1 + sim_cum2)
    mf_final_tot_abs, mf_final_tot_rel = final_abs_rel(mf_cum1 + mf_cum2, sim_cum1 + sim_cum2)
    n_final_tot_abs, n_final_tot_rel = final_abs_rel(nimfa_cum1 + nimfa_cum2, sim_cum1 + sim_cum2)

    print("TOTAL (Leader 1 + Leader 2):")
    print(f"Tree  - MAE: {t_mae_tot:.2f}, RMSE: {t_rmse_tot:.2f}, RelErr: {t_rel_tot:.1f}%")
    print(f"MF    - MAE: {mf_mae_tot:.2f}, RMSE: {mf_rmse_tot:.2f}, RelErr: {mf_rel_tot:.1f}%")
    print(f"NIMFA - MAE: {n_mae_tot:.2f}, RMSE: {n_rmse_tot:.2f}, RelErr: {n_rel_tot:.1f}%")

    print("TOTAL FINAL-TIME ERRORS (t = T, relative to Monte Carlo final size):")
    print(f"Tree  - AbsErr(T): {t_final_tot_abs:.2f}, FinalRelErr(T): {t_final_tot_rel:.1f}%")
    print(f"MF    - AbsErr(T): {mf_final_tot_abs:.2f}, FinalRelErr(T): {mf_final_tot_rel:.1f}%")
    print(f"NIMFA - AbsErr(T): {n_final_tot_abs:.2f}, FinalRelErr(T): {n_final_tot_rel:.1f}%")

    cumulative_winner = min({"Tree": t_rel_tot, "MF": mf_rel_tot, "NIMFA": n_rel_tot}, key=lambda k: {"Tree": t_rel_tot, "MF": mf_rel_tot, "NIMFA": n_rel_tot}[k])
    final_winner = min({"Tree": t_final_tot_rel, "MF": mf_final_tot_rel, "NIMFA": n_final_tot_rel}, key=lambda k: {"Tree": t_final_tot_rel, "MF": mf_final_tot_rel, "NIMFA": n_final_tot_rel}[k])

    print("" + "-" * 90)
    print(f"BEST METHOD BY TOTAL CUMULATIVE RelErr: {cumulative_winner}")
    print(f"BEST METHOD BY TOTAL FINAL-TIME RelErr(T): {final_winner}")
    print("-" * 90)

    return {
        'total': {
            'tree': {'mae': t_mae_tot, 'rmse': t_rmse_tot, 'rel_mae': t_rel_tot, 'final_abs': t_final_tot_abs, 'final_rel': t_final_tot_rel},
            'mf': {'mae': mf_mae_tot, 'rmse': mf_rmse_tot, 'rel_mae': mf_rel_tot, 'final_abs': mf_final_tot_abs, 'final_rel': mf_final_tot_rel},
            'nimfa': {'mae': n_mae_tot, 'rmse': n_rmse_tot, 'rel_mae': n_rel_tot, 'final_abs': n_final_tot_abs, 'final_rel': n_final_tot_rel},
            'best_cumulative_method': cumulative_winner,
            'best_final_method': final_winner,
        }
    }


def calculate_error_metrics_multi(sim_avg, tree_data, mf_data, nimfa_avg, H: int) -> Dict[str, np.ndarray]:
    methods = ['Tree', 'MF', 'NIMFA']
    final_rel = np.zeros((3, H), dtype=float)
    time_rel = np.zeros((3, H), dtype=float)

    for h in range(H):
        sim_curve = np.array([d[f'cumulative_{h + 1}_avg'] for d in sim_avg], dtype=float)
        tree_curve = np.array([d[f'cum_{h + 1}'] for d in tree_data], dtype=float)
        mf_curve = np.array([d[f'cum_{h + 1}'] for d in mf_data], dtype=float)
        nimfa_curve = np.array([d[f'cum_{h + 1}_avg'] for d in nimfa_avg], dtype=float)
        curves = [tree_curve, mf_curve, nimfa_curve]
        for m_idx, curve in enumerate(curves):
            final_rel[m_idx, h] = 100.0 * abs(curve[-1] - sim_curve[-1]) / max(abs(curve[-1]), abs(sim_curve[-1]), 1e-12)
            time_rel[m_idx, h] = 100.0 * np.mean(np.abs(curve - sim_curve) / np.maximum(np.maximum(np.abs(curve), np.abs(sim_curve)), 1e-12))

    return {'methods': methods, 'final_rel': final_rel, 'time_rel': time_rel}


# ============================================================
# VISUALS
# ============================================================

def plot_network_states_separate(model, save_figures=False, output_dir="outputs_competitive_influence"):
    print("" + "=" * 80)
    print("VISUALIZING NETWORK STATES (START OF EACH STEP) | LAST ITERATION")
    print("=" * 80)

    pos = get_graph_positions(model.G, model.graph_model, seed=42)
    state_colors = {-2: 'darkred', -1: 'darkblue', 0: 'lightgray', 1: 'blue', 2: 'red'}
    state_labels = {-2: 'Silent (Leader 2)', -1: 'Silent (Leader 1)', 0: 'Inactive', 1: 'Active (Leader 1)', 2: 'Active (Leader 2)'}
    unique_states = sorted(state_colors.keys())
    nodes = list(model.G.nodes())

    if save_figures:
        ensure_dir(output_dir)

    for idx in range(len(model.history)):
        t = idx + 1
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111)
        states = model.history[idx]
        node_colors = [state_colors[int(states[n])] for n in nodes]
        nx.draw_networkx_nodes(model.G, pos, node_color=node_colors, node_size=100, ax=ax, alpha=0.9)
        nx.draw_networkx_edges(model.G, pos, alpha=0.2, width=0.7, ax=ax)

        if t == 1:
            nx.draw_networkx_nodes(
                model.G,
                pos,
                nodelist=model.seed_nodes,
                node_color=[('blue' if n in model.seed_nodes_p1 else 'red') for n in model.seed_nodes],
                node_size=240,
                edgecolors='black',
                linewidths=2,
                ax=ax,
            )

        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=state_colors[s], markersize=10, label=state_labels[s]) for s in unique_states]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, title="State Legend")
        ax.set_title(f"Network State at START of Step t = {t}", fontsize=12, pad=20)
        ax.set_axis_off()
        plt.tight_layout()
        if save_figures:
            plt.savefig(os.path.join(output_dir, f"network_state_time_{t}_start_step.png"), dpi=150, bbox_inches='tight')
        plt.show()


def plot_comparative_analysis(sim_avg, tree_data, mf_data, nimfa_avg, num_iterations, k_value=None,
                              graph_model="ER", save_figures=False, output_dir="outputs_competitive_influence"):
    print("" + "=" * 90)
    print(f"COMPARATIVE ANALYSIS (Average of {num_iterations} runs) | Graph={graph_model}")
    if k_value is not None:
        print(f"Using fixed k = {k_value:.2f} (Tree + MF baseline)")
    print("Including averaged NIMFA curve.")
    print("=" * 90)

    t = [d['time'] for d in sim_avg]
    sim1 = [d['cumulative1_avg'] for d in sim_avg]
    sim2 = [d['cumulative2_avg'] for d in sim_avg]
    tree1 = [d['cum_X1_finite'] for d in tree_data]
    tree2 = [d['cum_X2_finite'] for d in tree_data]
    mf1 = [d['cum_A1'] for d in mf_data]
    mf2 = [d['cum_A2'] for d in mf_data]
    nim1 = [d['cum_N1_avg'] for d in nimfa_avg]
    nim2 = [d['cum_N2_avg'] for d in nimfa_avg]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Graph model: {graph_model}", fontsize=15)

    ax1.plot(t, sim1, linewidth=3, label='Simulation (Avg)', marker='o')
    ax1.plot(t, mf1, linewidth=2, linestyle=':', label='MF')
    ax1.plot(t, tree1, linewidth=2, linestyle='--', label='Tree')
    ax1.plot(t, nim1, linewidth=2, linestyle='-.', label='NIMFA')
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Cumulative activations')
    ax1.set_title('Leader 1')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(t, sim2, linewidth=3, label='Simulation (Avg)', marker='s')
    ax2.plot(t, mf2, linewidth=2, linestyle=':', label='MF')
    ax2.plot(t, tree2, linewidth=2, linestyle='--', label='Tree')
    ax2.plot(t, nim2, linewidth=2, linestyle='-.', label='NIMFA')
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Cumulative activations')
    ax2.set_title('Leader 2')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_figures:
        ensure_dir(output_dir)
        plt.savefig(os.path.join(output_dir, f"comparison_{graph_model}.png"), dpi=150, bbox_inches='tight')
    plt.show()


def plot_batch_metric(xvals, curves: Dict[str, List[float]], xlabel: str, ylabel: str, title: str, save_path: Optional[str] = None):
    plt.figure(figsize=(10, 5))
    for label, y in curves.items():
        plt.plot(xvals, y, marker='o', linewidth=2, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_k_error_lollipop(k_values, mf_errors, nimfa_errors,
                          ylabel='Relative error (%)',
                          title='Model error vs average degree k',
                          save_path: Optional[str] = None):
    plt.figure(figsize=(11, 5))

    plt.plot(
        k_values,
        mf_errors,
        marker='o',
        markersize=7,
        linewidth=2.2,
        linestyle='-',
        label='MF vs simulation'
    )

    plt.plot(
        k_values,
        nimfa_errors,
        marker='s',
        markersize=7,
        linewidth=2.2,
        linestyle='-',
        label='NIMFA vs simulation'
    )

    plt.xlabel('Target average degree k')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path) or ".")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_multi_project_cumulative_panels(sim_avg, tree_data, mf_data, nimfa_avg, H: int,
                                         graph_model: str, save_figures=False, output_dir="outputs_competitive_influence"):
    t = [d['time'] for d in sim_avg]
    ncols = min(3, H)
    nrows = int(math.ceil(H / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.ravel()
    for h in range(H):
        ax = axes[h]
        sim_curve = [d[f'cumulative_{h + 1}_avg'] for d in sim_avg]
        tree_curve = [d[f'cum_{h + 1}'] for d in tree_data]
        mf_curve = [d[f'cum_{h + 1}'] for d in mf_data]
        nimfa_curve = [d[f'cum_{h + 1}_avg'] for d in nimfa_avg]
        ax.plot(t, sim_curve, linewidth=2.8, marker='o', label='Simulation')
        ax.plot(t, tree_curve, linewidth=2, linestyle='--', label='Tree')
        ax.plot(t, mf_curve, linewidth=2, linestyle=':', label='MF')
        ax.plot(t, nimfa_curve, linewidth=2, linestyle='-.', label='NIMFA')
        ax.set_title(f'Project {h + 1}')
        ax.set_xlabel('Time t')
        ax.set_ylabel('Cumulative activations')
        ax.grid(True, alpha=0.25)
    for ax in axes[H:]:
        ax.axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4)
    fig.suptitle(f'Multi-project cumulative comparison on {graph_model}', fontsize=14)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    if save_figures:
        ensure_dir(output_dir)
        plt.savefig(os.path.join(output_dir, f'multiproject_panels_{graph_model}.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_multi_project_final_bars(sim_avg, tree_data, mf_data, nimfa_avg, H: int,
                                  graph_model: str, save_figures=False, output_dir="outputs_competitive_influence"):
    sim_final = np.array([sim_avg[-1][f'cumulative_{h + 1}_avg'] for h in range(H)])
    tree_final = np.array([tree_data[-1][f'cum_{h + 1}'] for h in range(H)])
    mf_final = np.array([mf_data[-1][f'cum_{h + 1}'] for h in range(H)])
    nimfa_final = np.array([nimfa_avg[-1][f'cum_{h + 1}_avg'] for h in range(H)])

    x = np.arange(H)
    w = 0.2
    plt.figure(figsize=(max(10, 1.5 * H), 5))
    plt.bar(x - 1.5 * w, sim_final, width=w, label='Simulation')
    plt.bar(x - 0.5 * w, tree_final, width=w, label='Tree')
    plt.bar(x + 0.5 * w, mf_final, width=w, label='MF')
    plt.bar(x + 1.5 * w, nimfa_final, width=w, label='NIMFA')
    plt.xticks(x, [f'P{h + 1}' for h in range(H)])
    plt.xlabel('Project')
    plt.ylabel('Final cumulative activations')
    plt.title(f'Final cumulative comparison on {graph_model}')
    plt.grid(True, axis='y', alpha=0.25)
    plt.legend()
    plt.tight_layout()
    if save_figures:
        ensure_dir(output_dir)
        plt.savefig(os.path.join(output_dir, f'multiproject_final_bars_{graph_model}.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_multi_project_error_heatmap(error_payload: Dict[str, np.ndarray], H: int,
                                     graph_model: str, save_figures=False, output_dir="outputs_competitive_influence"):
    mat = error_payload['final_rel']
    methods = error_payload['methods']
    plt.figure(figsize=(max(8, H), 4))
    im = plt.imshow(mat, aspect='auto', cmap='YlOrRd')
    plt.colorbar(im, label='Final relative error (%)')
    plt.yticks(np.arange(len(methods)), methods)
    plt.xticks(np.arange(H), [f'P{h + 1}' for h in range(H)])
    plt.xlabel('Project')
    plt.ylabel('Method')
    plt.title(f'Final relative error heatmap on {graph_model}')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f'{mat[i, j]:.1f}', ha='center', va='center', color='black', fontsize=9)
    plt.tight_layout()
    if save_figures:
        ensure_dir(output_dir)
        plt.savefig(os.path.join(output_dir, f'multiproject_error_heatmap_{graph_model}.png'), dpi=150, bbox_inches='tight')
    plt.show()


def print_lattice_nimfa_summary(results, histories, output_dir: str):
    print("" + "=" * 90)
    print("LATTICE NIMFA SUMMARY")
    print("=" * 90)
    print("For the lattice demo, we show both true simulation snapshots and node-level NIMFA quantities.")
    print("  M_i^1(t) : approximate probability that node i is newly active for Leader 1 at time t")
    print("  M_i^2(t) : approximate probability that node i is newly active for Leader 2 at time t")
    print("and the aggregated curves")
    print("  ~M_1(t) = sum_i M_i^1(t),   ~M_2(t) = sum_i M_i^2(t)")
    print("which are the expected numbers of newly active nodes at time t.")
    print("We also keep cumulative probabilities bar{M}_i^h(t) in the histories.")

    rows = []
    for d in results:
        rows.append([
            d['time'],
            f"{d['new_N1']:.4f}",
            f"{d['new_N2']:.4f}",
            f"{d['cum_N1']:.4f}",
            f"{d['cum_N2']:.4f}",
        ])
    print(tabulate(rows, headers=["t", "~M1(t)", "~M2(t)", "cum1", "cum2"], tablefmt="grid"))

    ensure_dir(output_dir)
    write_matrix_csv(os.path.join(output_dir, "lattice_nimfa_M1_by_time_and_node.csv"), histories['M1'])
    write_matrix_csv(os.path.join(output_dir, "lattice_nimfa_M2_by_time_and_node.csv"), histories['M2'])
    write_matrix_csv(os.path.join(output_dir, "lattice_nimfa_cum1_by_time_and_node.csv"), histories['cum1'])
    write_matrix_csv(os.path.join(output_dir, "lattice_nimfa_cum2_by_time_and_node.csv"), histories['cum2'])
    print(f"Nodewise NIMFA matrices saved to: {output_dir}")


def plot_lattice_nimfa_totals(results, save_figures=False, output_dir="outputs_competitive_influence"):
    t = [d['time'] for d in results]
    m1 = [d['new_N1'] for d in results]
    m2 = [d['new_N2'] for d in results]
    c1 = [d['cum_N1'] for d in results]
    c2 = [d['cum_N2'] for d in results]

    plt.figure(figsize=(10, 5))
    plt.plot(t, m1, label=r'$\tilde M_1(t)$', linewidth=2)
    plt.plot(t, m2, label=r'$\tilde M_2(t)$', linewidth=2)
    plt.plot(t, c1, label='cum Leader 1', linestyle='--', linewidth=2)
    plt.plot(t, c2, label='cum Leader 2', linestyle='--', linewidth=2)
    plt.xlabel('Time t')
    plt.ylabel('Expected number of nodes')
    plt.title('Lattice NIMFA totals')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_figures:
        ensure_dir(output_dir)
        plt.savefig(os.path.join(output_dir, 'lattice_nimfa_totals.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_lattice_nimfa_heatmaps(histories, save_figures=False, output_dir="outputs_competitive_influence"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    mats = [
        (histories['M1'], r'$M_i^1(t)$', axes[0, 0]),
        (histories['M2'], r'$M_i^2(t)$', axes[0, 1]),
        (histories['cum1'], r'$\bar M_i^1(t)$', axes[1, 0]),
        (histories['cum2'], r'$\bar M_i^2(t)$', axes[1, 1]),
    ]
    for mat, title, ax in mats:
        im = ax.imshow(mat, aspect='auto', origin='lower')
        ax.set_title(title)
        ax.set_ylabel('time index')
        ax.set_xlabel('node index')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_figures:
        ensure_dir(output_dir)
        plt.savefig(os.path.join(output_dir, 'lattice_nimfa_heatmaps.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_lattice_nimfa_snapshots(G: nx.Graph, histories, times: List[int],
                                 save_figures=False, output_dir="outputs_competitive_influence"):
    pos = get_graph_positions(G, "LATTICE")
    nodes = list(G.nodes())
    ncols = min(3, len(times))
    nrows = int(math.ceil(len(times) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, t in zip(axes, times):
        idx = min(max(1, t), histories['cum1'].shape[0]) - 1
        cum1 = histories['cum1'][idx]
        cum2 = histories['cum2'][idx]
        node_colors = []
        for i in nodes:
            if cum1[i] < 1e-10 and cum2[i] < 1e-10:
                node_colors.append('lightgray')
            elif cum1[i] >= cum2[i]:
                node_colors.append('red')
            else:
                node_colors.append('blue')
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=260, ax=ax)
        nx.draw_networkx_edges(G, pos, width=0.8, alpha=0.35, ax=ax)
        ax.set_title(f'Lattice NIMFA snapshot at t={t}')
        ax.set_axis_off()

    for ax in axes[len(times):]:
        ax.axis('off')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Leader 1 dominant', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Leader 2 dominant', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Still negligible / inactive', markerfacecolor='lightgray', markersize=10),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if save_figures:
        ensure_dir(output_dir)
        plt.savefig(os.path.join(output_dir, 'lattice_nimfa_snapshots.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_lattice_simulation_snapshots(model, times: List[int],
                                      save_figures=False, output_dir="outputs_competitive_influence"):
    pos = get_graph_positions(model.G, "LATTICE")
    nodes = list(model.G.nodes())
    ncols = min(3, len(times))
    nrows = int(math.ceil(len(times) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    state_colors = {-2: 'darkred', -1: 'darkblue', 0: 'lightgray', 1: 'blue', 2: 'red'}

    for ax, t in zip(axes, times):
        idx = min(max(1, t), len(model.history)) - 1
        st = model.history[idx]
        node_colors = [state_colors[int(st[n])] for n in nodes]
        nx.draw_networkx_nodes(model.G, pos, node_color=node_colors, node_size=260, ax=ax)
        nx.draw_networkx_edges(model.G, pos, width=0.8, alpha=0.35, ax=ax)
        ax.set_title(f'Lattice simulation snapshot at t={t}')
        ax.set_axis_off()

    for ax in axes[len(times):]:
        ax.axis('off')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Leader 1 active', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Leader 1 silent', markerfacecolor='darkblue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Leader 2 active', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Leader 2 silent', markerfacecolor='darkred', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Inactive', markerfacecolor='lightgray', markersize=10),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if save_figures:
        ensure_dir(output_dir)
        plt.savefig(os.path.join(output_dir, 'lattice_simulation_snapshots.png'), dpi=150, bbox_inches='tight')
    plt.show()


def print_selected_lattice_node_prob_tables(histories, node_ids: List[int]):
    print("" + "=" * 90)
    print("SELECTED NODEWISE NIMFA PROBABILITIES")
    print("=" * 90)
    print(f"Selected agents: {node_ids}")
    print("M_i^1(t): approximate probability that agent i becomes newly active for Leader 1 at time t")
    print("M_i^2(t): approximate probability that agent i becomes newly active for Leader 2 at time t")

    headers_m1 = ['t'] + [f'M1[node {i}]' for i in node_ids]
    table_m1 = []
    T = histories['M1'].shape[0]
    for t in range(T):
        row = [t + 1] + [f"{histories['M1'][t, i]:.4f}" for i in node_ids]
        table_m1.append(row)
    print("\nNodewise table for M_i^1(t)")
    print(tabulate(table_m1, headers=headers_m1, tablefmt='grid'))

    headers_m2 = ['t'] + [f'M2[node {i}]' for i in node_ids]
    table_m2 = []
    for t in range(T):
        row = [t + 1] + [f"{histories['M2'][t, i]:.4f}" for i in node_ids]
        table_m2.append(row)
    print("\nNodewise table for M_i^2(t)")
    print(tabulate(table_m2, headers=headers_m2, tablefmt='grid'))


def plot_selected_lattice_node_traces(histories, node_ids: List[int],
                                      save_figures=False, output_dir="outputs_competitive_influence"):
    t = np.arange(1, histories['M1'].shape[0] + 1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for node_id in node_ids:
        axes[0].plot(t, histories['M1'][:, node_id], linewidth=2, label=f'node {node_id}')
    axes[0].set_ylabel(r'$M_i^1(t)$')
    axes[0].set_title('Selected nodewise probabilities for Leader 1')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=3, fontsize=9)

    for node_id in node_ids:
        axes[1].plot(t, histories['M2'][:, node_id], linewidth=2, label=f'node {node_id}')
    axes[1].set_xlabel('Time t')
    axes[1].set_ylabel(r'$M_i^2(t)$')
    axes[1].set_title('Selected nodewise probabilities for Leader 2')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=3, fontsize=9)

    plt.tight_layout()
    if save_figures:
        ensure_dir(output_dir)
        plt.savefig(os.path.join(output_dir, 'lattice_selected_node_traces.png'), dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# RUN TWO-LEADER ANALYSIS
# ============================================================

def run_complete_analysis(cfg: ExperimentConfig, plot_summary: Optional[bool] = None,
                          plot_network: Optional[bool] = None,
                          print_tables: Optional[bool] = None,
                          save_csv_name: Optional[str] = None):
    gm = cfg.graph_model.upper()
    gp = cfg.graph_params or {}

    if plot_summary is None:
        plot_summary = cfg.plot_single_run_summary
    if plot_network is None:
        plot_network = cfg.plot_network_states
    if print_tables is None:
        print_tables = cfg.verbose_tables

    p = float(cfg.k_target) / max(1, (cfg.N - 1))

    print("=" * 90)
    print("COMPETITIVE INFLUENCE MODEL ANALYSIS")
    print("=" * 90)
    print(f"Graph model: {gm} | graph_params={gp}")
    print(f"Parameters: N={cfg.N}, T={cfg.T}, runs={cfg.num_iterations}")
    print(f"gamma1={cfg.gamma1}, gamma2={cfg.gamma2}, alpha12={cfg.alpha12}, alpha21={cfg.alpha21}")
    print(f"Initially active nodes: m1={cfg.m1}, m2={cfg.m2}")
    print("Seed Selection: Random")
    print("k for Tree + MF: fixed expected degree")

    k_expected = expected_degree_for_models(gm, cfg.N, p, gp)
    print(f"Using fixed k (baseline for Tree + MF): k = {k_expected:.4f}")

    all_sim_results = []
    all_nimfa_results = []
    last_model = None

    t0 = time.perf_counter()
    print("Running stochastic simulation + NIMFA each iteration...")
    for i in range(cfg.num_iterations):
        random.seed(cfg.base_seed_rng + i)
        np.random.seed(cfg.base_seed_rng + i)

        model = StochasticInfluenceModel(
            N=cfg.N,
            p=p,
            T=cfg.T,
            gamma1=cfg.gamma1,
            gamma2=cfg.gamma2,
            alpha12=cfg.alpha12,
            alpha21=cfg.alpha21,
            m1=cfg.m1,
            m2=cfg.m2,
            random_seeds=cfg.random_seeds,
            iteration=i,
            graph_model=gm,
            graph_params=gp,
            base_seed_graph=cfg.base_seed_graph,
        )

        sim_res = model.run_simulation()
        nimfa_res = model.compute_nimfa_model()
        all_sim_results.append(sim_res)
        all_nimfa_results.append(nimfa_res)
        if i == cfg.num_iterations - 1:
            last_model = model
        if i % max(1, min(100, cfg.num_iterations // 10)) == 0 or i == cfg.num_iterations - 1:
            print(f"  Completed {i + 1}/{cfg.num_iterations} runs...")

    elapsed = time.perf_counter() - t0
    print_graph_stats(last_model.G, gm, gp)

    sim_data_avg = []
    nimfa_data_avg = []
    for idx in range(cfg.T):
        tt = all_sim_results[0][idx]['time']
        sim_data_avg.append({
            'time': tt,
            'active1_avg': safe_mean([all_sim_results[r][idx]['active1'] for r in range(cfg.num_iterations)]),
            'active2_avg': safe_mean([all_sim_results[r][idx]['active2'] for r in range(cfg.num_iterations)]),
            'cumulative1_avg': safe_mean([all_sim_results[r][idx]['cumulative1'] for r in range(cfg.num_iterations)]),
            'cumulative2_avg': safe_mean([all_sim_results[r][idx]['cumulative2'] for r in range(cfg.num_iterations)]),
        })
        nimfa_data_avg.append({
            'time': tt,
            'new_N1_avg': safe_mean([all_nimfa_results[r][idx]['new_N1'] for r in range(cfg.num_iterations)]),
            'new_N2_avg': safe_mean([all_nimfa_results[r][idx]['new_N2'] for r in range(cfg.num_iterations)]),
            'cum_N1_avg': safe_mean([all_nimfa_results[r][idx]['cum_N1'] for r in range(cfg.num_iterations)]),
            'cum_N2_avg': safe_mean([all_nimfa_results[r][idx]['cum_N2'] for r in range(cfg.num_iterations)]),
        })

    tree_results, _ = last_model.compute_tree_model(fixed_k=k_expected)
    mf_results, _ = last_model.compute_mean_field_model(fixed_k=k_expected)
    k_used = k_expected

    if print_tables:
        print_activation_tables(sim_data_avg, tree_results, mf_results, nimfa_data_avg, cfg.num_iterations, k_used)

    error_metrics = calculate_error_metrics(sim_data_avg, tree_results, mf_results, nimfa_data_avg, cfg.num_iterations)

    if plot_network:
        plot_network_states_separate(last_model, cfg.save_figures, cfg.output_dir)
    if plot_summary:
        plot_comparative_analysis(sim_data_avg, tree_results, mf_results, nimfa_data_avg,
                                  cfg.num_iterations, k_used, gm, cfg.save_figures, cfg.output_dir)

    payload = {
        'graph_model': gm,
        'graph_params': gp,
        'N': cfg.N,
        'T': cfg.T,
        'num_iterations': cfg.num_iterations,
        'k_target': cfg.k_target,
        'k_used_tree_mf': k_used,
        'runtime_seconds': elapsed,
        'final_sim_p1': sim_data_avg[-1]['cumulative1_avg'],
        'final_sim_p2': sim_data_avg[-1]['cumulative2_avg'],
        'final_tree_p1': tree_results[-1]['cum_X1_finite'],
        'final_tree_p2': tree_results[-1]['cum_X2_finite'],
        'final_mf_p1': mf_results[-1]['cum_A1'],
        'final_mf_p2': mf_results[-1]['cum_A2'],
        'final_nimfa_p1': nimfa_data_avg[-1]['cum_N1_avg'],
        'final_nimfa_p2': nimfa_data_avg[-1]['cum_N2_avg'],
        'tree_relmae_total': error_metrics['total']['tree']['rel_mae'],
        'mf_relmae_total': error_metrics['total']['mf']['rel_mae'],
        'nimfa_relmae_total': error_metrics['total']['nimfa']['rel_mae'],
        'tree_finalrel_total': error_metrics['total']['tree']['final_rel'],
        'mf_finalrel_total': error_metrics['total']['mf']['final_rel'],
        'nimfa_finalrel_total': error_metrics['total']['nimfa']['final_rel'],
        'best_cumulative_method': error_metrics['total']['best_cumulative_method'],
        'best_final_method': error_metrics['total']['best_final_method'],
    }

    if save_csv_name:
        ensure_dir(cfg.output_dir)
        rows = []
        for i in range(cfg.T):
            rows.append({
                'time': sim_data_avg[i]['time'],
                'sim_active1_avg': sim_data_avg[i]['active1_avg'],
                'sim_active2_avg': sim_data_avg[i]['active2_avg'],
                'sim_cum1_avg': sim_data_avg[i]['cumulative1_avg'],
                'sim_cum2_avg': sim_data_avg[i]['cumulative2_avg'],
                'tree_new1': tree_results[i]['new_X1_finite'],
                'tree_new2': tree_results[i]['new_X2_finite'],
                'tree_cum1': tree_results[i]['cum_X1_finite'],
                'tree_cum2': tree_results[i]['cum_X2_finite'],
                'mf_new1': mf_results[i]['new_A1'],
                'mf_new2': mf_results[i]['new_A2'],
                'mf_cum1': mf_results[i]['cum_A1'],
                'mf_cum2': mf_results[i]['cum_A2'],
                'nimfa_new1_avg': nimfa_data_avg[i]['new_N1_avg'],
                'nimfa_new2_avg': nimfa_data_avg[i]['new_N2_avg'],
                'nimfa_cum1_avg': nimfa_data_avg[i]['cum_N1_avg'],
                'nimfa_cum2_avg': nimfa_data_avg[i]['cum_N2_avg'],
            })
        write_csv(os.path.join(cfg.output_dir, save_csv_name), rows)

    return payload, error_metrics


# ============================================================
# SPECIAL MODES
# ============================================================

def run_multi_project_mode(cfg: ExperimentConfig):
    print("MULTI-PROJECT MODE")
    H = int(input("Enter number of projects H: ").strip())
    gamma, alpha = prompt_multi_project_parameters(H, cfg.base_seed_rng)
    ok, msg = validate_multi_params(gamma, alpha)
    if not ok:
        raise ValueError(msg)
    pretty_print_multi_params(gamma, alpha)

    p = float(cfg.k_target) / max(1, (cfg.N - 1))
    k_expected = expected_degree_for_models(cfg.graph_model, cfg.N, p, cfg.graph_params)

    print("" + "=" * 90)
    print("MULTI-PROJECT STOCHASTIC + APPROXIMATION ANALYSIS")
    print("=" * 90)
    print(f"Graph model: {cfg.graph_model}")
    print(f"N={cfg.N}, T={cfg.T}, H={H}, runs={cfg.num_iterations}")
    print("One initiator per project at t=1")
    print(f"Using fixed k for Tree + MF baseline: {k_expected:.4f}")

    all_sim_results = []
    all_nimfa_results = []
    last_model = None
    t0 = time.perf_counter()

    for i in range(cfg.num_iterations):
        random.seed(cfg.base_seed_rng + i)
        np.random.seed(cfg.base_seed_rng + i)
        model = MultiProjectInfluenceModel(cfg.N, p, cfg.T, H, gamma, alpha,
                                           cfg.graph_model, cfg.graph_params, i, cfg.base_seed_graph)
        sim_res = model.run_simulation()
        nimfa_res = compute_nimfa_model_H(model.G, cfg.T, gamma, alpha, model.seed_nodes)
        all_sim_results.append(sim_res)
        all_nimfa_results.append(nimfa_res)
        if i == cfg.num_iterations - 1:
            last_model = model
        if i % max(1, min(100, cfg.num_iterations // 10)) == 0 or i == cfg.num_iterations - 1:
            print(f"  Completed {i + 1}/{cfg.num_iterations} runs...")

    elapsed = time.perf_counter() - t0
    print_graph_stats(last_model.G, cfg.graph_model, cfg.graph_params)

    sim_avg = []
    nimfa_avg = []
    for t_idx in range(cfg.T):
        row_sim = {'time': all_sim_results[0][t_idx]['time']}
        row_nimfa = {'time': all_nimfa_results[0][t_idx]['time']}
        for h in range(H):
            row_sim[f'active_{h + 1}_avg'] = safe_mean([all_sim_results[r][t_idx][f'active_{h + 1}'] for r in range(cfg.num_iterations)])
            row_sim[f'cumulative_{h + 1}_avg'] = safe_mean([all_sim_results[r][t_idx][f'cumulative_{h + 1}'] for r in range(cfg.num_iterations)])
            row_nimfa[f'new_{h + 1}_avg'] = safe_mean([all_nimfa_results[r][t_idx][f'new_{h + 1}'] for r in range(cfg.num_iterations)])
            row_nimfa[f'cum_{h + 1}_avg'] = safe_mean([all_nimfa_results[r][t_idx][f'cum_{h + 1}'] for r in range(cfg.num_iterations)])
        sim_avg.append(row_sim)
        nimfa_avg.append(row_nimfa)

    m = np.ones(H, dtype=float)
    tree_data = compute_tree_model_H(cfg.N, cfg.T, k_expected, gamma, alpha, m)
    mf_data = compute_mf_model_H(cfg.N, cfg.T, k_expected, gamma, alpha, m)
    error_payload = calculate_error_metrics_multi(sim_avg, tree_data, mf_data, nimfa_avg, H)

    print("\nFINAL AVERAGE CUMULATIVE ACTIVATIONS")
    final_table = []
    for h in range(H):
        final_table.append([
            f'P{h + 1}',
            sim_avg[-1][f'cumulative_{h + 1}_avg'],
            tree_data[-1][f'cum_{h + 1}'],
            mf_data[-1][f'cum_{h + 1}'],
            nimfa_avg[-1][f'cum_{h + 1}_avg'],
        ])
    print(tabulate(final_table,
                   headers=["Project", "Simulation", "Tree", "MF", "NIMFA"],
                   tablefmt="grid", floatfmt=".2f"))

    print("\nFINAL RELATIVE ERRORS (%) PER PROJECT")
    err_table = []
    for h in range(H):
        err_table.append([
            f'P{h + 1}',
            error_payload['final_rel'][0, h],
            error_payload['final_rel'][1, h],
            error_payload['final_rel'][2, h],
        ])
    print(tabulate(err_table,
                   headers=["Project", "Tree", "MF", "NIMFA"],
                   tablefmt="grid", floatfmt=".2f"))

    plot_multi_project_cumulative_panels(sim_avg, tree_data, mf_data, nimfa_avg, H,
                                         cfg.graph_model, cfg.save_figures, cfg.output_dir)
    plot_multi_project_final_bars(sim_avg, tree_data, mf_data, nimfa_avg, H,
                                  cfg.graph_model, cfg.save_figures, cfg.output_dir)
    plot_multi_project_error_heatmap(error_payload, H,
                                     cfg.graph_model, cfg.save_figures, cfg.output_dir)

    rows = []
    for t_idx in range(cfg.T):
        row = {'time': sim_avg[t_idx]['time']}
        for h in range(H):
            row[f'sim_active_{h + 1}_avg'] = sim_avg[t_idx][f'active_{h + 1}_avg']
            row[f'sim_cum_{h + 1}_avg'] = sim_avg[t_idx][f'cumulative_{h + 1}_avg']
            row[f'tree_new_{h + 1}'] = tree_data[t_idx][f'new_{h + 1}']
            row[f'tree_cum_{h + 1}'] = tree_data[t_idx][f'cum_{h + 1}']
            row[f'mf_new_{h + 1}'] = mf_data[t_idx][f'new_{h + 1}']
            row[f'mf_cum_{h + 1}'] = mf_data[t_idx][f'cum_{h + 1}']
            row[f'nimfa_new_{h + 1}_avg'] = nimfa_avg[t_idx][f'new_{h + 1}_avg']
            row[f'nimfa_cum_{h + 1}_avg'] = nimfa_avg[t_idx][f'cum_{h + 1}_avg']
        rows.append(row)
    ensure_dir(cfg.output_dir)
    write_csv(os.path.join(cfg.output_dir, f"multiproject_{cfg.graph_model}_H{H}.csv"), rows)

    payload = {
        'graph_model': cfg.graph_model,
        'H': H,
        'runtime_seconds': elapsed,
        'gamma': gamma.tolist(),
        'alpha': alpha.tolist(),
        'final_rel_error_tree_mean': float(np.mean(error_payload['final_rel'][0])),
        'final_rel_error_mf_mean': float(np.mean(error_payload['final_rel'][1])),
        'final_rel_error_nimfa_mean': float(np.mean(error_payload['final_rel'][2])),
    }
    return payload, error_payload


def run_lattice_nimfa_demo(cfg: ExperimentConfig):
    local = clone_cfg(cfg)
    local.graph_model = "LATTICE"
    local.N = int(cfg.lattice_demo_N)
    local.T = int(cfg.lattice_demo_T)
    local.num_iterations = 1
    local.graph_params = {
        'rows': int(cfg.lattice_demo_rows),
        'cols': int(cfg.lattice_demo_cols),
        'periodic': False,
    }

    p = float(local.k_target) / max(1, (local.N - 1))
    model = StochasticInfluenceModel(
        N=local.N,
        p=p,
        T=local.T,
        gamma1=local.gamma1,
        gamma2=local.gamma2,
        alpha12=local.alpha12,
        alpha21=local.alpha21,
        m1=1,
        m2=1,
        random_seeds=True,
        iteration=0,
        graph_model="LATTICE",
        graph_params=local.graph_params,
        base_seed_graph=local.base_seed_graph,
    )

    # Use random initial seeds so repeated runs are not identical.
    sys_rng = random.SystemRandom()
    chosen = sys_rng.sample(list(model.G.nodes()), 2)
    model.set_manual_seeds([chosen[0]], [chosen[1]])
    print("" + "=" * 90)
    print("LATTICE DEMO")
    print("=" * 90)
    print(f"Random initial seed for Leader 1 (blue): node {chosen[0]}")
    print(f"Random initial seed for Leader 2 (red): node {chosen[1]}")
    print("Simulation snapshots use the true stochastic states, so once a node adopts a leader")
    print("it cannot switch later; it only changes from active to silent in the next time step.")

    sim_results = model.run_simulation()
    results, histories = model.compute_nimfa_full(return_histories=True)

    print_lattice_nimfa_summary(results, histories, local.output_dir)
    selected_nodes = sorted(set(list(range(min(3, local.N))) + list(range(max(0, local.N - 3), local.N))))
    print_selected_lattice_node_prob_tables(histories, selected_nodes)

    snap_times = sorted(set([1, 2, 3, 4, max(5, local.T // 2), local.T]))
    plot_lattice_simulation_snapshots(model, snap_times,
                                      save_figures=local.save_figures, output_dir=local.output_dir)
    plot_lattice_nimfa_totals(results, save_figures=local.save_figures, output_dir=local.output_dir)
    plot_lattice_nimfa_heatmaps(histories, save_figures=local.save_figures, output_dir=local.output_dir)
    plot_selected_lattice_node_traces(histories, selected_nodes,
                                      save_figures=local.save_figures, output_dir=local.output_dir)

    ensure_dir(local.output_dir)
    write_csv(os.path.join(local.output_dir, 'lattice_simulation_summary.csv'), sim_results)
    return results, histories


def run_sbm_tuned_experiment(cfg: ExperimentConfig):
    local = clone_cfg(cfg)
    local.graph_model = "SBM"
    N_input = input("Enter N for tuned SBM [default 1000]: ").strip()
    T_input = input("Enter T for tuned SBM [default 8]: ").strip()
    local.N = int(N_input) if N_input else 1000
    local.T = int(T_input) if T_input else 8
    local.k_target = 15
    local.plot_network_states = False
    local.plot_single_run_summary = True
    local.verbose_tables = True

    sizes = [local.N // 2, local.N - local.N // 2]
    p_out = 0.001
    s1, s2 = sizes
    avg_in = (max(0, s1 - 1) + max(0, s2 - 1)) / 2.0
    avg_out = (s1 + s2) / 2.0
    p_in = (local.k_target - p_out * avg_out) / max(1e-12, avg_in)
    local.graph_params = {
        'sizes': sizes,
        'p_in': float(np.clip(p_in, 0.0, 1.0)),
        'p_out': p_out,
    }

    print("" + "=" * 90)
    print("TUNED SBM EXPERIMENT")
    print("=" * 90)
    print("Goal: test whether stronger communities and larger expected degree help NIMFA.")
    print(f"Chosen parameters: N={local.N}, T={local.T}, k_target≈{local.k_target}, p_in={local.graph_params['p_in']:.5f}, p_out={local.graph_params['p_out']:.5f}")
    print("Interpretation: smaller p_out keeps communities more separated, while k≈15 reduces sparse-noise effects.")

    return run_complete_analysis(local, save_csv_name="sbm_tuned_k15.csv")


def experiment_k_sweep_errors(cfg: ExperimentConfig,
                              k_values: Optional[List[int]] = None,
                              use_final_error: bool = False):
    k_values = k_values or list(range(3, 101, 3))
    rows = []

    for k in k_values:
        local = clone_cfg(cfg)
        local.k_target = float(k)
        local.graph_params = _default_graph_params_for_model(local, local.graph_model)
        local.plot_network_states = False
        local.plot_single_run_summary = False
        local.verbose_tables = False
        print(f"### K-SWEEP EXPERIMENT: k={k} | graph={local.graph_model} ###")
        payload, _ = run_complete_analysis(
            local,
            plot_summary=False,
            plot_network=False,
            print_tables=False,
            save_csv_name=f"k_sweep_{local.graph_model}_k_{k}.csv"
        )
        row = {
            'graph_model': local.graph_model,
            'k_target': k,
            'mf_relmae_total': payload['mf_relmae_total'],
            'nimfa_relmae_total': payload['nimfa_relmae_total'],
            'mf_finalrel_total': payload['mf_finalrel_total'],
            'nimfa_finalrel_total': payload['nimfa_finalrel_total'],
            'final_sim_total': payload['final_sim_p1'] + payload['final_sim_p2'],
            'final_mf_total': payload['final_mf_p1'] + payload['final_mf_p2'],
            'final_nimfa_total': payload['final_nimfa_p1'] + payload['final_nimfa_p2'],
        }
        rows.append(row)

    write_csv(os.path.join(cfg.output_dir, f"k_sweep_summary_{cfg.graph_model}.csv"), rows)

    if use_final_error:
        mf_y = [r['mf_finalrel_total'] for r in rows]
        nimfa_y = [r['nimfa_finalrel_total'] for r in rows]
        ylabel = 'Final-time relative error (%)'
        title = f'Final-time error vs k on {cfg.graph_model.upper()}'
        save_name = f'k_sweep_final_error_{cfg.graph_model}.png'
    else:
        mf_y = [r['mf_relmae_total'] for r in rows]
        nimfa_y = [r['nimfa_relmae_total'] for r in rows]
        ylabel = 'Relative error (%)'
        title = f'Cumulative error vs k on {cfg.graph_model.upper()}'
        save_name = f'k_sweep_cumulative_error_{cfg.graph_model}.png'

    plot_k_error_lollipop(
        k_values=[r['k_target'] for r in rows],
        mf_errors=mf_y,
        nimfa_errors=nimfa_y,
        ylabel=ylabel,
        title=title,
        save_path=os.path.join(cfg.output_dir, save_name) if cfg.save_figures else None
    )
    return rows


# ============================================================
# BATCH EXPERIMENTS
# ============================================================

def _default_graph_params_for_model(cfg: ExperimentConfig, model_name: str) -> dict:
    gm = model_name.upper()
    if gm == "ER":
        return {}
    if gm == "WS":
        k_ws = int(round(cfg.k_target))
        k_ws = max(2, min(k_ws, cfg.N - 1))
        if k_ws % 2 == 1:
            k_ws += 1
            if k_ws >= cfg.N:
                k_ws = max(2, cfg.N - 2)
        return {"k_ws": k_ws, "beta": 0.2}
    if gm == "BA":
        return {"m": max(1, int(round(cfg.k_target / 2)))}
    if gm == "RR":
        d = int(round(cfg.k_target))
        d = max(0, min(d, cfg.N - 1))
        if (cfg.N * d) % 2 == 1:
            d = max(0, d - 1)
        return {"d": d}
    if gm == "SBM":
        sizes = [cfg.N // 2, cfg.N - cfg.N // 2]
        p_out = 0.001
        s1, s2 = sizes
        avg_in = (max(0, s1 - 1) + max(0, s2 - 1)) / 2.0
        avg_out = (s1 + s2) / 2.0
        p_in = (cfg.k_target - p_out * avg_out) / max(1e-12, avg_in)
        return {"sizes": sizes, "p_in": float(np.clip(p_in, 0.0, 1.0)), "p_out": p_out}
    if gm == "HK":
        return {"m": max(1, int(round(cfg.k_target / 2))), "triad_p": 0.35}
    if gm == "RCAV":
        clique_size = _best_divisor_near_target(cfg.N, max(3, int(round(cfg.k_target)) + 1))
        return {"clique_size": clique_size, "rewire_p": 0.08}
    if gm == "LATTICE":
        rows, cols = _grid_dims(cfg.N)
        return {"rows": rows, "cols": cols, "periodic": False}
    raise ValueError(f"Unknown graph model {model_name}")


def experiment_graph_families(cfg: ExperimentConfig, graph_models: Optional[List[str]] = None):
    graph_models = graph_models or ["ER", "WS", "BA", "RR", "SBM", "HK", "RCAV", "LATTICE"]
    rows = []

    for gm in graph_models:
        local = clone_cfg(cfg)
        local.graph_model = gm
        local.graph_params = _default_graph_params_for_model(local, gm)
        local.plot_network_states = False
        local.plot_single_run_summary = False
        local.verbose_tables = False
        print(f"### GRAPH FAMILY EXPERIMENT: {gm} ###")
        payload, _ = run_complete_analysis(local, plot_summary=True, plot_network=False, print_tables=False,
                                           save_csv_name=f"graph_family_{gm}.csv")
        rows.append(payload)

    print("" + "=" * 90)
    print("GRAPH FAMILY SUMMARY")
    print("=" * 90)
    table_rows = [[
        r['graph_model'], r['final_sim_p1'], r['final_sim_p2'],
        r['tree_relmae_total'], r['mf_relmae_total'], r['nimfa_relmae_total'],
        r['best_cumulative_method'], r['best_final_method']
    ] for r in rows]
    print(tabulate(table_rows,
                   headers=["Graph", "Final Sim P1", "Final Sim P2", "Tree RelErr %", "MF RelErr %", "NIMFA RelErr %", "Best Cum", "Best Final"],
                   tablefmt="grid", floatfmt=".2f"))

    write_csv(os.path.join(cfg.output_dir, "graph_family_summary.csv"), rows)
    plot_batch_metric([r['graph_model'] for r in rows],
                      {
                          'Tree RelErr %': [r['tree_relmae_total'] for r in rows],
                          'MF RelErr %': [r['mf_relmae_total'] for r in rows],
                          'NIMFA RelErr %': [r['nimfa_relmae_total'] for r in rows],
                      },
                      xlabel='Graph model', ylabel='Relative error (%)',
                      title='Approximation error across graph families (cumulative curves)',
                      save_path=os.path.join(cfg.output_dir, 'graph_family_errors_cumulative.png') if cfg.save_figures else None)
    plot_batch_metric([r['graph_model'] for r in rows],
                      {
                          'Tree Final RelErr(T) %': [r['tree_finalrel_total'] for r in rows],
                          'MF Final RelErr(T) %': [r['mf_finalrel_total'] for r in rows],
                          'NIMFA Final RelErr(T) %': [r['nimfa_finalrel_total'] for r in rows],
                      },
                      xlabel='Graph model', ylabel='Final-time relative error (%)',
                      title='Approximation error across graph families at final time',
                      save_path=os.path.join(cfg.output_dir, 'graph_family_errors_final.png') if cfg.save_figures else None)
    return rows


def experiment_parameter_sweep(cfg: ExperimentConfig, parameter_sets: Optional[List[dict]] = None):
    if parameter_sets is None:
        parameter_sets = [
            {'gamma1': 0.20, 'gamma2': 0.20, 'alpha12': 0.05, 'alpha21': 0.05},
            {'gamma1': 0.25, 'gamma2': 0.20, 'alpha12': 0.10, 'alpha21': 0.15},
            {'gamma1': 0.30, 'gamma2': 0.25, 'alpha12': 0.10, 'alpha21': 0.10},
            {'gamma1': 0.35, 'gamma2': 0.30, 'alpha12': 0.15, 'alpha21': 0.15},
        ]
    rows = []
    for idx, pars in enumerate(parameter_sets):
        local = clone_cfg(cfg)
        for k, v in pars.items():
            setattr(local, k, v)
        local.plot_network_states = False
        local.plot_single_run_summary = False
        local.verbose_tables = False
        print(f"### PARAMETER SWEEP EXPERIMENT {idx + 1}/{len(parameter_sets)} ###")
        payload, _ = run_complete_analysis(local, plot_summary=False, plot_network=False, print_tables=False,
                                           save_csv_name=f"parameter_sweep_{idx + 1}.csv")
        rows.append(payload)
    write_csv(os.path.join(cfg.output_dir, "parameter_sweep_summary.csv"), rows)
    return rows


def experiment_multiple_leaders(cfg: ExperimentConfig, leader_values: Optional[List[int]] = None, symmetric_only: bool = False):
    leader_values = leader_values or [1, 5, 10, 20]
    rows = []
    pairs = [(m, m) for m in leader_values] if symmetric_only else [(m1, m2) for m1 in leader_values for m2 in leader_values]

    for m1, m2 in pairs:
        local = clone_cfg(cfg)
        local.m1 = int(m1)
        local.m2 = int(m2)
        local.plot_network_states = False
        local.plot_single_run_summary = False
        local.verbose_tables = False
        print(f"### LEADER SWEEP: (m1, m2)=({m1}, {m2}) ###")
        payload, _ = run_complete_analysis(local, plot_summary=False, plot_network=False, print_tables=False,
                                           save_csv_name=f"leaders_m1_{m1}_m2_{m2}.csv")
        rows.append(payload)

    write_csv(os.path.join(cfg.output_dir, "multiple_leaders_summary.csv"), rows)
    return rows


def experiment_size_scaling(cfg: ExperimentConfig, sizes: Optional[List[int]] = None):
    sizes = sizes or [200, 500, 1000, 2000]
    rows = []
    for N in sizes:
        local = clone_cfg(cfg)
        local.N = int(N)
        local.graph_params = _default_graph_params_for_model(local, local.graph_model)
        local.plot_network_states = False
        local.plot_single_run_summary = False
        local.verbose_tables = False
        if N >= 2000:
            local.num_iterations = min(local.num_iterations, 1000)
        elif N >= 1000:
            local.num_iterations = min(local.num_iterations, 2000)
        print(f"### SIZE SCALING: N={N} ###")
        payload, _ = run_complete_analysis(local, plot_summary=False, plot_network=False, print_tables=False,
                                           save_csv_name=f"size_N_{N}.csv")
        rows.append(payload)

    write_csv(os.path.join(cfg.output_dir, "size_scaling_summary.csv"), rows)
    xvals = [r['N'] for r in rows]
    plot_batch_metric(xvals,
                      {
                          'Tree RelErr %': [r['tree_relmae_total'] for r in rows],
                          'MF RelErr %': [r['mf_relmae_total'] for r in rows],
                          'NIMFA RelErr %': [r['nimfa_relmae_total'] for r in rows],
                      },
                      xlabel='Network size N', ylabel='Relative error (%)',
                      title=f'Accuracy vs network size on {cfg.graph_model.upper()} (cumulative curves)',
                      save_path=os.path.join(cfg.output_dir, 'size_scaling_errors_cumulative.png') if cfg.save_figures else None)
    plot_batch_metric(xvals,
                      {'Runtime (s)': [r['runtime_seconds'] for r in rows]},
                      xlabel='Network size N', ylabel='Runtime (seconds)',
                      title=f'Runtime vs network size on {cfg.graph_model.upper()}',
                      save_path=os.path.join(cfg.output_dir, 'size_scaling_runtime.png') if cfg.save_figures else None)
    return rows



# ============================================================
# COMPETITION WHAT-IF ANALYSIS 
# ============================================================

def _mean_ci95(values):
    """Return mean and a two-sided 95% CI for Monte Carlo observations.

    Student-t is used when SciPy is available; otherwise the large-sample
    1.96 critical value is used. With the intended R=1000 runs the difference
    is negligible.
    """
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    if n == 1:
        return mean, mean, mean
    sd = float(np.std(arr, ddof=1))
    se = sd / math.sqrt(n)
    try:
        from scipy.stats import t as student_t
        crit = float(student_t.ppf(0.975, n - 1))
    except Exception:
        crit = 1.96
    return mean, mean - crit * se, mean + crit * se


def _largest_component_nodes(G: nx.Graph) -> List[int]:
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    return sorted(list(comps[0])) if comps else list(G.nodes())


def _pick_node_near_degree(G: nx.Graph, target_degree: int,
                           excluded: Optional[set] = None,
                           avoid_neighbors_of: Optional[int] = None) -> int:
    """Pick a node in the LCC whose realized degree is closest to target_degree."""
    excluded = excluded or set()
    candidates = _largest_component_nodes(G)
    if avoid_neighbors_of is not None:
        preferred = [n for n in candidates
                     if n not in excluded and n != avoid_neighbors_of
                     and not G.has_edge(n, avoid_neighbors_of)]
        if preferred:
            candidates = preferred
        else:
            candidates = [n for n in candidates if n not in excluded]
    else:
        candidates = [n for n in candidates if n not in excluded]
    if not candidates:
        raise ValueError("No candidate node available for the requested seed selection.")
    return min(candidates, key=lambda n: (abs(G.degree(n) - target_degree), n))


def _pick_two_balanced_seed_nodes(G: nx.Graph, target_degree: int) -> Tuple[int, int]:
    """Pick two non-adjacent nodes with degrees as close as possible to target_degree."""
    first = _pick_node_near_degree(G, target_degree)
    second = _pick_node_near_degree(G, target_degree, excluded={first}, avoid_neighbors_of=first)
    return first, second


def _validate_two_project_parameters(gamma1: float, gamma2: float,
                                     alpha12: float, alpha21: float):
    vals = [gamma1, gamma2, alpha12, alpha21]
    if any(v < 0 or v > 1 for v in vals):
        raise ValueError("All follow/switch probabilities must lie in [0,1].")
    if gamma1 + alpha12 > 1 + 1e-12:
        raise ValueError("Need gamma1 + alpha12 <= 1.")
    if gamma2 + alpha21 > 1 + 1e-12:
        raise ValueError("Need gamma2 + alpha21 <= 1.")


def _run_fixed_graph_competition_scenario(
        cfg: ExperimentConfig,
        G: nx.Graph,
        seed1: int,
        seed2: int,
        gamma1: float,
        gamma2: float,
        alpha12: float,
        alpha21: float,
        num_iterations: int,
        scenario_seed_offset: int = 0) -> dict:
    """Run one controlled scenario on the same realized graph and fixed seeds."""
    _validate_two_project_parameters(gamma1, gamma2, alpha12, alpha21)

    p = float(cfg.k_target) / max(1, (cfg.N - 1))
    final1, final2, deltas = [], [], []

    for r in range(num_iterations):
        # Common random-number structure across scenarios improves comparability.
        rng_seed = int(cfg.base_seed_rng + scenario_seed_offset + r)
        random.seed(rng_seed)
        np.random.seed(rng_seed)

        model = StochasticInfluenceModel(
            N=cfg.N,
            p=p,
            T=cfg.T,
            gamma1=gamma1,
            gamma2=gamma2,
            alpha12=alpha12,
            alpha21=alpha21,
            m1=1,
            m2=1,
            random_seeds=False,
            iteration=0,
            graph_model="ER",
            graph_params={},
            base_seed_graph=cfg.base_seed_graph,
            fixed_graph=G,
        )
        model.set_manual_seeds([seed1], [seed2])
        res = model.run_simulation()
        z1 = float(res[-1]['cumulative1'])
        z2 = float(res[-1]['cumulative2'])
        final1.append(z1)
        final2.append(z2)
        deltas.append(z1 - z2)

    z1_mean, z1_lo, z1_hi = _mean_ci95(final1)
    z2_mean, z2_lo, z2_hi = _mean_ci95(final2)
    d_mean, d_lo, d_hi = _mean_ci95(deltas)

    # NIMFA is deterministic once the graph, seeds and parameters are fixed.
    model_n = StochasticInfluenceModel(
        N=cfg.N,
        p=p,
        T=cfg.T,
        gamma1=gamma1,
        gamma2=gamma2,
        alpha12=alpha12,
        alpha21=alpha21,
        m1=1,
        m2=1,
        random_seeds=False,
        iteration=0,
        graph_model="ER",
        graph_params={},
        base_seed_graph=cfg.base_seed_graph,
        fixed_graph=G,
    )
    model_n.set_manual_seeds([seed1], [seed2])
    nimfa = model_n.compute_nimfa_model()
    n1 = float(nimfa[-1]['cum_N1'])
    n2 = float(nimfa[-1]['cum_N2'])

    if d_mean > 1e-12:
        winner = "P1"
    elif d_mean < -1e-12:
        winner = "P2"
    else:
        winner = "Tie"

    return {
        'seed1': int(seed1),
        'seed2': int(seed2),
        'degree1': int(G.degree(seed1)),
        'degree2': int(G.degree(seed2)),
        'gamma1': float(gamma1),
        'gamma2': float(gamma2),
        'alpha12': float(alpha12),
        'alpha21': float(alpha21),
        'sim_Z1_mean': z1_mean,
        'sim_Z1_ci_low': z1_lo,
        'sim_Z1_ci_high': z1_hi,
        'sim_Z2_mean': z2_mean,
        'sim_Z2_ci_low': z2_lo,
        'sim_Z2_ci_high': z2_hi,
        'sim_Delta_mean': d_mean,
        'sim_Delta_ci_low': d_lo,
        'sim_Delta_ci_high': d_hi,
        'winner': winner,
        'nimfa_Z1': n1,
        'nimfa_Z2': n2,
        'nimfa_Delta': n1 - n2,
    }


def _write_competition_whatif_latex(path: str, seed_rows: List[dict], param_rows: List[dict]):
    """Write compact LaTeX tables that can be pasted into the revised paper."""
    ensure_dir(os.path.dirname(path) or ".")

    def fmt(x):
        return f"{float(x):.2f}"

    lines = []
    lines.append(r"% Auto-generated by experiment_competition_what_if")
    lines.append(r"% Table A: effect of the local connectivity of the initial active users")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Competition outcome on the same ER graph when the initial positions of the two projects are exchanged.}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Setting & $d_1$ & $d_2$ & $\mathbb{E}[Z_1^T]$ & $\mathbb{E}[Z_2^T]$ & $\Delta(T)$ \\")
    lines.append(r"\midrule")
    for r in seed_rows:
        label = str(r['scenario']).replace('_', r'\_')
        lines.append(f"{label} & {r['degree1']} & {r['degree2']} & {fmt(r['sim_Z1_mean'])} & {fmt(r['sim_Z2_mean'])} & {fmt(r['sim_Delta_mean'])} " + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    lines.append(r"% Table B: one-at-a-time parameter changes")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{One-at-a-time competition analysis on a fixed ER graph. All non-varied parameters are kept at their baseline values.}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Parameter & Value & $\mathbb{E}[Z_1^T]$ & $\mathbb{E}[Z_2^T]$ & $\Delta(T)$ \\")
    lines.append(r"\midrule")
    for r in param_rows:
        pname = {'gamma1': r'$\gamma_1$', 'gamma2': r'$\gamma_2$',
                 'alpha12': r'$\alpha_{12}$', 'alpha21': r'$\alpha_{21}$'}[r['varied_parameter']]
        lines.append(f"{pname} & {float(r['parameter_value']):.2f} & {fmt(r['sim_Z1_mean'])} & {fmt(r['sim_Z2_mean'])} & {fmt(r['sim_Delta_mean'])} " + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


def experiment_competition_what_if(cfg: ExperimentConfig,
                                   num_iterations: int = 1000,
                                   weak_target_degree: int = 3,
                                   strong_target_degree: int = 15):
    """Reviewer-oriented use case on ONE fixed ER graph.

    Part A isolates the impact of the local network position of the initial
    active users by exchanging a low-degree and a high-degree seed.

    Part B fixes the graph and uses two similarly connected seeds, then changes
    exactly one follow/switch parameter at a time. The competition outcome is
    measured by Delta(T) = E[Z_1^T] - E[Z_2^T].
    """
    local = clone_cfg(cfg)
    local.graph_model = "ER"
    local.graph_params = {}
    local.N = 2000
    local.T = 5
    local.k_target = 10.0
    local.num_iterations = int(num_iterations)
    local.gamma1 = 0.35
    local.gamma2 = 0.25
    local.alpha12 = 0.20
    local.alpha21 = 0.15

    ensure_dir(local.output_dir)
    p = local.k_target / (local.N - 1)
    G = nx.erdos_renyi_graph(local.N, p, seed=local.base_seed_graph)
    G = _simple_graph(G)

    print("" + "=" * 100)
    print("COMPETITION WHAT-IF ANALYSIS ON ONE FIXED ER GRAPH")
    print("=" * 100)
    print(f"N={local.N}, T={local.T}, R={local.num_iterations}, target k={local.k_target:.0f}")
    print("Baseline parameters:")
    print(f"  gamma1={local.gamma1}, gamma2={local.gamma2}, alpha12={local.alpha12}, alpha21={local.alpha21}")
    print(f"Realized average degree = {2 * G.number_of_edges() / G.number_of_nodes():.3f}")

    # ------------------------------------------------------------------
    # A. Local connectivity / seed position
    # ------------------------------------------------------------------
    weak = _pick_node_near_degree(G, weak_target_degree)
    strong = _pick_node_near_degree(G, strong_target_degree,
                                    excluded={weak}, avoid_neighbors_of=weak)

    print("\nA) EFFECT OF INITIAL LOCAL CONNECTIVITY")
    print(f"Weak seed candidate:   node {weak}, degree={G.degree(weak)}")
    print(f"Strong seed candidate: node {strong}, degree={G.degree(strong)}")

    seed_rows = []
    seed_settings = [
        ("P1 weak / P2 strong", weak, strong),
        ("P1 strong / P2 weak", strong, weak),
    ]
    for idx, (label, s1, s2) in enumerate(seed_settings):
        r = _run_fixed_graph_competition_scenario(
            local, G, s1, s2,
            local.gamma1, local.gamma2, local.alpha12, local.alpha21,
            local.num_iterations,
            scenario_seed_offset=0,  # same MC random streams for the two placements
        )
        r['experiment'] = 'seed_position'
        r['scenario'] = label
        seed_rows.append(r)

    seed_table = [[
        r['scenario'], r['seed1'], r['degree1'], r['seed2'], r['degree2'],
        r['sim_Z1_mean'], r['sim_Z2_mean'], r['sim_Delta_mean'],
        f"[{r['sim_Delta_ci_low']:.2f}, {r['sim_Delta_ci_high']:.2f}]", r['winner']
    ] for r in seed_rows]
    print(tabulate(seed_table,
                   headers=["Setting", "P1 seed", "d1", "P2 seed", "d2",
                            "E[Z1^T]", "E[Z2^T]", "Delta(T)", "95% CI Delta", "Winner"],
                   tablefmt="grid", floatfmt=".2f"))

    # ------------------------------------------------------------------
    # B. One-at-a-time follow/switch parameter changes
    # ------------------------------------------------------------------
    balanced1, balanced2 = _pick_two_balanced_seed_nodes(G, int(local.k_target))
    print("\nB) ONE-AT-A-TIME PARAMETER CHANGES")
    print(f"Balanced fixed seeds: P1 node {balanced1} (degree={G.degree(balanced1)}), "
          f"P2 node {balanced2} (degree={G.degree(balanced2)})")

    baseline = {
        'gamma1': local.gamma1,
        'gamma2': local.gamma2,
        'alpha12': local.alpha12,
        'alpha21': local.alpha21,
    }
    sweep_values = {
        'gamma1': [0.20, 0.35, 0.50],
        'gamma2': [0.10, 0.25, 0.40],
        'alpha12': [0.05, 0.20, 0.35],
        'alpha21': [0.00, 0.15, 0.30],
    }

    param_rows = []
    scenario_idx = 0
    for parameter_name, values in sweep_values.items():
        for value in values:
            pars = dict(baseline)
            pars[parameter_name] = float(value)
            r = _run_fixed_graph_competition_scenario(
                local, G, balanced1, balanced2,
                pars['gamma1'], pars['gamma2'], pars['alpha12'], pars['alpha21'],
                local.num_iterations,
                scenario_seed_offset=0,  # same MC random streams across parameter settings
            )
            r['experiment'] = 'parameter_sweep'
            r['scenario'] = f"vary_{parameter_name}"
            r['varied_parameter'] = parameter_name
            r['parameter_value'] = float(value)
            param_rows.append(r)
            scenario_idx += 1

    param_table = [[
        r['varied_parameter'], r['parameter_value'],
        r['sim_Z1_mean'], r['sim_Z2_mean'], r['sim_Delta_mean'],
        f"[{r['sim_Delta_ci_low']:.2f}, {r['sim_Delta_ci_high']:.2f}]", r['winner']
    ] for r in param_rows]
    print(tabulate(param_table,
                   headers=["Varied parameter", "Value", "E[Z1^T]", "E[Z2^T]",
                            "Delta(T)", "95% CI Delta", "Winner"],
                   tablefmt="grid", floatfmt=".2f"))

    # Save machine-readable results.
    seed_csv = os.path.join(local.output_dir, "competition_what_if_seed_position.csv")
    param_csv = os.path.join(local.output_dir, "competition_what_if_parameter_sweep.csv")
    write_csv(seed_csv, seed_rows)
    write_csv(param_csv, param_rows)

    # Combined CSV: normalize keys because the two experiment blocks have
    # a few block-specific columns (scenario versus varied_parameter/value).
    all_rows_raw = seed_rows + param_rows
    all_keys = []
    for rr in all_rows_raw:
        for key in rr.keys():
            if key not in all_keys:
                all_keys.append(key)
    all_rows = [{key: rr.get(key, "") for key in all_keys} for rr in all_rows_raw]
    write_csv(os.path.join(local.output_dir, "competition_what_if_all_results.csv"), all_rows)

    # Ready-to-paste LaTeX tables.
    latex_path = os.path.join(local.output_dir, "competition_what_if_tables.tex")
    _write_competition_whatif_latex(latex_path, seed_rows, param_rows)

    # ------------------------------------------------------------------
    # Figures for the paper / reviewer response
    # ------------------------------------------------------------------
    x = np.arange(len(seed_rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, [r['sim_Z1_mean'] for r in seed_rows], width, label='Project 1')
    ax.bar(x + width / 2, [r['sim_Z2_mean'] for r in seed_rows], width, label='Project 2')
    ax.set_xticks(x)
    ax.set_xticklabels([r['scenario'] for r in seed_rows])
    ax.set_ylabel('Expected cumulative influence at T')
    ax.set_title('Effect of initial local connectivity on competition')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    seed_fig = os.path.join(local.output_dir, "competition_seed_position.png")
    fig.savefig(seed_fig, dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    labels = {
        'gamma1': r'$\gamma_1$',
        'gamma2': r'$\gamma_2$',
        'alpha12': r'$\alpha_{12}$',
        'alpha21': r'$\alpha_{21}$',
    }
    for ax, pname in zip(axes, ['gamma1', 'gamma2', 'alpha12', 'alpha21']):
        rr = [r for r in param_rows if r['varied_parameter'] == pname]
        vals = [r['parameter_value'] for r in rr]
        means = [r['sim_Delta_mean'] for r in rr]
        lows = [r['sim_Delta_mean'] - r['sim_Delta_ci_low'] for r in rr]
        highs = [r['sim_Delta_ci_high'] - r['sim_Delta_mean'] for r in rr]
        ax.errorbar(vals, means, yerr=[lows, highs], marker='o', capsize=4)
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xlabel(labels[pname])
        ax.set_ylabel(r'$\Delta(T)$')
        ax.set_title(f'Vary {pname}; all other parameters fixed')
        ax.grid(alpha=0.25)
    fig.suptitle('One-at-a-time effect of model parameters on competition')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    param_fig = os.path.join(local.output_dir, "competition_parameter_effects.png")
    fig.savefig(param_fig, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print("\nSaved reviewer-oriented what-if outputs:")
    print(f"  {seed_csv}")
    print(f"  {param_csv}")
    print(f"  {latex_path}")
    print(f"  {seed_fig}")
    print(f"  {param_fig}")

    return {
        'graph': G,
        'seed_rows': seed_rows,
        'parameter_rows': param_rows,
        'weak_seed': weak,
        'strong_seed': strong,
        'balanced_seed1': balanced1,
        'balanced_seed2': balanced2,
        'latex_table_path': latex_path,
    }


# ============================================================
# MENUS
# ============================================================

def menu_choose_graph_and_options(cfg: ExperimentConfig) -> ExperimentConfig:
    print("Select graph model:")
    print("1. ER       (Erdos-Renyi)")
    print("2. WS       (Watts-Strogatz small-world)")
    print("3. BA       (Barabasi-Albert scale-free)")
    print("4. RR       (Random Regular)")
    print("5. SBM      (Stochastic Block Model)")
    print("6. HK       (Holme-Kim power-law + clustering)")
    print("7. RCAV     (Relaxed Caveman / strong communities)")
    print("8. LATTICE  (2D grid lattice)")

    g_choice = input("Enter choice (1/2/3/4/5/6/7/8): ").strip()
    mapping = {'1': 'ER', '2': 'WS', '3': 'BA', '4': 'RR', '5': 'SBM', '6': 'HK', '7': 'RCAV', '8': 'LATTICE'}
    cfg.graph_model = mapping.get(g_choice, 'ER')
    cfg.graph_params = _default_graph_params_for_model(cfg, cfg.graph_model)

    print("Seed Selection: Random")
    print("k for Tree + MF: fixed expected degree")
    print(f"Initially active nodes currently set to m1={cfg.m1}, m2={cfg.m2}")
    return cfg


def menu_choose_mode() -> str:
    print("Select run mode:")
    print("1. Single scenario")
    print("2. Compare graph families (ER/WS/BA/RR/SBM/HK/RCAV/LATTICE)")
    print("3. Parameter sweep")
    print("4. Multiple-initial-node sweep (two leaders only: vary m1 and m2)")
    print("5. Size scaling")
    print("6. Multiple projects (heterogeneous parameters + Simulation/Tree/MF/NIMFA)")
    print("7. Tuned SBM experiment (stronger communities, k≈15)")
    print("8. Lattice demo (random seeds, simulation snapshots, NIMFA nodewise plots)")
    print("9. Error vs k plot (MF and NIMFA against simulation)")
    print("10. Competition what-if analysis on one fixed ER graph ")
    return input("Enter choice (1/2/3/4/5/6/7/8/9/10): ").strip()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    ensure_dir(CONFIG.output_dir)
    cfg = clone_cfg(CONFIG)
    mode = menu_choose_mode()

    if mode == '1':
        cfg = menu_choose_graph_and_options(cfg)
        run_complete_analysis(cfg, save_csv_name=f"single_{cfg.graph_model.upper()}.csv")

    elif mode == '2':
        cfg.plot_network_states = False
        cfg.plot_single_run_summary = False
        cfg.verbose_tables = False
        experiment_graph_families(cfg)

    elif mode == '3':
        cfg = menu_choose_graph_and_options(cfg)
        cfg.plot_network_states = False
        cfg.plot_single_run_summary = False
        cfg.verbose_tables = False
        experiment_parameter_sweep(cfg)

    elif mode == '4':
        cfg = menu_choose_graph_and_options(cfg)
        cfg.plot_network_states = False
        cfg.plot_single_run_summary = False
        cfg.verbose_tables = False
        print("This sweep is for TWO leaders only.")
        print("m1 = number of initially active nodes for Leader 1")
        print("m2 = number of initially active nodes for Leader 2")
        sym = input("Run only symmetric cases m1=m2? (y/n): ").strip().lower() == 'y'
        experiment_multiple_leaders(cfg, leader_values=[1, 5, 10, 20], symmetric_only=sym)

    elif mode == '5':
        cfg = menu_choose_graph_and_options(cfg)
        cfg.plot_network_states = False
        cfg.plot_single_run_summary = False
        cfg.verbose_tables = False
        experiment_size_scaling(cfg)

    elif mode == '6':
        cfg = menu_choose_graph_and_options(cfg)
        cfg.plot_network_states = False
        cfg.plot_single_run_summary = False
        cfg.verbose_tables = False
        run_multi_project_mode(cfg)

    elif mode == '7':
        cfg.plot_network_states = False
        run_sbm_tuned_experiment(cfg)

    elif mode == '8':
        cfg.plot_network_states = False
        cfg.plot_single_run_summary = False
        cfg.verbose_tables = False
        run_lattice_nimfa_demo(cfg)

    elif mode == '9':
        cfg = menu_choose_graph_and_options(cfg)
        cfg.plot_network_states = False
        cfg.plot_single_run_summary = False
        cfg.verbose_tables = False
        print("Running k-sweep for error plot.")
        print("Default values use k = 3, 6, ..., 99.")
        experiment_k_sweep_errors(cfg, k_values=list(range(3, 101, 3)), use_final_error=True)

    elif mode == '10':
        # Reviewer-oriented use case: keep ONE ER graph fixed and study competition.
        # The paper baseline is used: N=2000, T=5, k=10 and
        # (gamma1,gamma2,alpha12,alpha21)=(0.35,0.25,0.20,0.15).
        raw_runs = input("Monte Carlo runs for the what-if analysis [1000]: ").strip()
        runs = int(raw_runs) if raw_runs else 1000
        experiment_competition_what_if(cfg, num_iterations=runs,
                                       weak_target_degree=3, strong_target_degree=15)
    else:
        print("Invalid choice. Running single scenario with default settings.")
        cfg = menu_choose_graph_and_options(cfg)
        run_complete_analysis(cfg, save_csv_name=f"single_{cfg.graph_model.upper()}.csv")
