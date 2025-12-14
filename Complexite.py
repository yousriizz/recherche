#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import math
import random
import time
import sys
from io import StringIO
from collections import deque
from typing import List, Tuple, Set, Dict, Optional

# =========================
# Affichage / utilitaires
# =========================

def print_matrix(mat: List[List[float]], title: str, fmt: str = "{:>10}"):
    print(f"\n{title}")
    for row in mat:
        print(" ".join(fmt.format(x) for x in row))

def print_vector(vec: List[float], title: str, fmt: str = "{:>10}"):
    print(f"\n{title}")
    print(" ".join(fmt.format(x) for x in vec))

def total_cost(costs: List[List[float]], x: List[List[float]]) -> float:
    return sum(costs[i][j] * x[i][j] for i in range(len(costs)) for j in range(len(costs[0])))

def almost_zero(v: float, tol: float = 1e-9) -> bool:
    return abs(v) < tol

# =========================
# Lecture fichier .txt
# =========================

def read_transport_problem_txt(path: str) -> Tuple[List[List[float]], List[float], List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.strip().startswith("#")]

    m, n = map(int, lines[0].split())
    if len(lines) != 1 + m + 1:
        raise ValueError(f"Format invalide : attendu {1+m+1} lignes, obtenu {len(lines)}")

    costs: List[List[float]] = []
    supply: List[float] = []

    for k in range(1, 1 + m):
        parts = list(map(float, lines[k].split()))
        if len(parts) != n + 1:
            raise ValueError(f"Ligne {k+1} : attendu {n+1} valeurs")
        costs.append(parts[:n])
        supply.append(parts[n])

    demand = list(map(float, lines[1 + m].split()))
    if len(demand) != n:
        raise ValueError(f"Dernière ligne : attendu {n} valeurs")

    return costs, supply, demand

# =========================
# Équilibrage
# =========================

def balance_problem(costs: List[List[float]], supply: List[float], demand: List[float]) -> Tuple[List[List[float]], List[float], List[float]]:
    s_sum = sum(supply)
    d_sum = sum(demand)
    m, n = len(costs), len(costs[0])

    if abs(s_sum - d_sum) < 1e-9:
        return costs, supply, demand

    if s_sum > d_sum:
        diff = s_sum - d_sum
        for i in range(m):
            costs[i].append(0.0)
        demand = demand + [diff]
    else:
        diff = d_sum - s_sum
        costs.append([0.0]*n)
        supply = supply + [diff]

    return costs, supply, demand

# =========================
# Génération aléatoire
# =========================

def generate_random_transport_problem(n: int):
    costs = [[random.randint(1, 100) for _ in range(n)] for _ in range(n)]
    temp = [[random.randint(1, 100) for _ in range(n)] for _ in range(n)]
    supply = [sum(temp[i]) for i in range(n)]
    demand = [sum(temp[i][j] for i in range(n)) for j in range(n)]
    return costs, supply, demand

# =========================
# Proposition initiale
# =========================

def northwest_corner(costs, supply, demand, verbose=True):
    m, n = len(costs), len(costs[0])
    a, b = supply[:], demand[:]
    x = [[0.0]*n for _ in range(m)]
    i = j = 0

    while i < m and j < n:
        q = min(a[i], b[j])
        x[i][j] = q
        a[i] -= q
        b[j] -= q
        if verbose:
            print(f"NO : x[{i},{j}] = {q}")
        if almost_zero(a[i]): i += 1
        if almost_zero(b[j]): j += 1

    return x

def balas_hammer(costs, supply, demand, verbose=True):
    m, n = len(costs), len(costs[0])
    a, b = supply[:], demand[:]
    x = [[0.0]*n for _ in range(m)]
    active_r = [True]*m
    active_c = [True]*n

    def penalty_row(i):
        vals = [costs[i][j] for j in range(n) if active_c[j]]
        if len(vals) < 2: return math.inf
        v1, v2 = sorted(vals)[:2]
        return v2 - v1

    def penalty_col(j):
        vals = [costs[i][j] for i in range(m) if active_r[i]]
        if len(vals) < 2: return math.inf
        v1, v2 = sorted(vals)[:2]
        return v2 - v1

    while any(active_r) and any(active_c):
        pr = [penalty_row(i) if active_r[i] else -1 for i in range(m)]
        pc = [penalty_col(j) if active_c[j] else -1 for j in range(n)]

        if max(pr) >= max(pc):
            i = pr.index(max(pr))
            j = min((j for j in range(n) if active_c[j]), key=lambda c: costs[i][c])
        else:
            j = pc.index(max(pc))
            i = min((i for i in range(m) if active_r[i]), key=lambda r: costs[r][j])

        q = min(a[i], b[j])
        x[i][j] = q
        a[i] -= q
        b[j] -= q
        if verbose:
            print(f"BH : x[{i},{j}] = {q}")
        if almost_zero(a[i]): active_r[i] = False
        if almost_zero(b[j]): active_c[j] = False

    return x

# =========================
# Graphe / base / potentiels
# =========================

Node = Tuple[str, int]   # ('r', i) ou ('c', j)
Cell = Tuple[int, int]   # (i, j)

def basis_positive(x: List[List[float]], tol: float = 1e-9) -> Set[Cell]:
    return {(i, j) for i in range(len(x)) for j in range(len(x[0])) if x[i][j] > tol}

def adjacency_from_edges(m: int, n: int, edges: Set[Cell]) -> Dict[Node, List[Node]]:
    adj: Dict[Node, List[Node]] = {}
    for (i, j) in edges:
        adj.setdefault(('r', i), []).append(('c', j))
        adj.setdefault(('c', j), []).append(('r', i))
    for i in range(m): adj.setdefault(('r', i), [])
    for j in range(n): adj.setdefault(('c', j), [])
    return adj

def count_components(adj: Dict[Node, List[Node]], m: int, n: int) -> int:
    seen: Set[Node] = set()
    comps = 0
    for node in adj:
        if node in seen: continue
        comps += 1
        q = deque([node])
        seen.add(node)
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
    return comps

def make_connected_by_zero_edges(costs: List[List[float]], x: List[List[float]], edges: Set[Cell]) -> Tuple[Set[Cell], List[Cell]]:
    m, n = len(costs), len(costs[0])
    added: List[Cell] = []

    while True:
        adj = adjacency_from_edges(m, n, edges)
        comps = count_components(adj, m, n)
        if comps <= 1:
            break

        comp_id: Dict[Node,int] = {}
        cid = 0
        for node in adj.keys():
            if node in comp_id: continue
            q = deque([node])
            comp_id[node] = cid
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in comp_id:
                        comp_id[v] = cid
                        q.append(v)
            cid += 1

        best = None
        best_cost = math.inf
        for i in range(m):
            for j in range(n):
                if (i,j) in edges: continue
                if comp_id[('r',i)] != comp_id[('c',j)] and costs[i][j] < best_cost:
                    best_cost = costs[i][j]
                    best = (i,j)
        if best is None: break
        edges.add(best)
        if almost_zero(x[best[0]][best[1]]):
            added.append(best)
    return edges, added

def spanning_tree_from_edges(costs: List[List[float]], edges: Set[Cell], m: int, n: int) -> Set[Cell]:
    total_nodes = m + n
    need = total_nodes - 1
    parent = list(range(total_nodes))
    rank = [0]*total_nodes

    def idx(node: Node) -> int:
        return node[1] if node[0]=='r' else m + node[1]

    def find(u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u: int, v: int) -> bool:
        ru, rv = find(u), find(v)
        if ru == rv: return False
        if rank[ru]<rank[rv]: parent[ru]=rv
        elif rank[ru]>rank[rv]: parent[rv]=ru
        else: parent[rv]=ru; rank[ru]+=1
        return True

    sorted_edges = sorted(list(edges), key=lambda e: costs[e[0]][e[1]])
    tree: Set[Cell] = set()
    for (i,j) in sorted_edges:
        if len(tree)>=need: break
        if union(idx(('r',i)), idx(('c',j))):
            tree.add((i,j))
    return tree

def compute_potentials(costs: List[List[float]], tree_edges: Set[Cell]) -> Tuple[List[float], List[float]]:
    m, n = len(costs), len(costs[0])
    adj = adjacency_from_edges(m,n,tree_edges)
    u: List[Optional[float]]=[None]*m
    v: List[Optional[float]]=[None]*n
    u[0]=0.0
    q=deque([('r',0)])
    while q:
        node=q.popleft()
        if node[0]=='r':
            i=node[1]
            for nb in adj[node]:
                j=nb[1]
                if v[j] is None:
                    v[j]=costs[i][j]-u[i]
                    q.append(('c',j))
        else:
            j=node[1]
            for nb in adj[node]:
                i=nb[1]
                if u[i] is None:
                    u[i]=costs[i][j]-v[j]
                    q.append(('r',i))
    for i in range(m):
        if u[i] is None: u[i]=0.0
    for j in range(n):
        if v[j] is None: v[j]=0.0
    return u,v

def potential_cost_table(u: List[float], v: List[float]) -> List[List[float]]:
    return [[u[i]+v[j] for j in range(len(v))] for i in range(len(u))]

def marginal_costs(costs: List[List[float]], u: List[float], v: List[float]) -> List[List[float]]:
    return [[costs[i][j]-(u[i]+v[j]) for j in range(len(v))] for i in range(len(u))]

def is_degenerate(basis_pos: Set[Cell], m:int, n:int) -> bool:
    return len(basis_pos)<m+n-1

def nodes_to_cell(u: Node, v: Node) -> Cell:
    if u[0]=='r' and v[0]=='c': return (u[1],v[1])
    if u[0]=='c' and v[0]=='r': return (v[1],u[1])
    raise ValueError("Arête invalide")

def find_path_in_tree(adj: Dict[Node,List[Node]], start: Node, goal: Node) -> List[Node]:
    q=deque([start])
    parent={start:None}
    while q:
        u=q.popleft()
        if u==goal: break
        for v in adj[u]:
            if v not in parent:
                parent[v]=u
                q.append(v)
    if goal not in parent: raise RuntimeError("Chemin introuvable")
    path=[]
    cur=goal
    while cur is not None:
        path.append(cur)
        cur=parent[cur]
    path.reverse()
    return path

def choose_entering_edge(d: List[List[float]], basis_for_test: Set[Cell]) -> Optional[Cell]:
    best=None; best_val=0.0
    m,n=len(d),len(d[0])
    for i in range(m):
        for j in range(n):
            if (i,j) in basis_for_test: continue
            if d[i][j]<best_val: best_val=d[i][j]; best=(i,j)
    return best

def improve_solution_on_cycle(x: List[List[float]], costs: List[List[float]], tree_edges: Set[Cell], entering: Cell) -> List[List[float]]:
    m,n=len(costs),len(costs[0])
    i0,j0=entering
    adj=adjacency_from_edges(m,n,tree_edges)
    path_nodes=find_path_in_tree(adj,('r',i0),('c',j0))
    cycle_cells=[entering]+[nodes_to_cell(path_nodes[k],path_nodes[k+1]) for k in range(len(path_nodes)-1)]
    plus_cells=[cycle_cells[i] for i in range(len(cycle_cells)) if i%2==0]
    minus_cells=[cycle_cells[i] for i in range(len(cycle_cells)) if i%2==1]
    theta=min(x[i][j] for (i,j) in minus_cells)
    for (i,j) in plus_cells: x[i][j]+=theta
    for (i,j) in minus_cells: x[i][j]-=theta; x[i][j]=0.0 if almost_zero(x[i][j]) else x[i][j]
    return x

# =========================
# MODI / Marche-pied
# =========================

def modi_silent(costs,x0):
    old=sys.stdout; sys.stdout=StringIO()
    try: return modi_stepping_stone(costs,x0)
    finally: sys.stdout=old

def modi_stepping_stone(costs: List[List[float]], x0: List[List[float]]) -> List[List[float]]:
    m,n=len(costs),len(costs[0])
    x=[row[:] for row in x0]
    it=1
    while True:
        print(f"\n================= Itération MODI {it} =================")
        print_matrix(x,"Proposition de transport X",fmt="{:>10.2f}")
        print(f"\nCoût total de transport = {total_cost(costs,x):.2f}")
        pos_basis=basis_positive(x)
        print(f"\nTest dégénérescence : {'DÉGÉNÉRÉE' if is_degenerate(pos_basis,m,n) else 'NON dégénérée'}")
        base_edges, added0 = make_connected_by_zero_edges(costs,x,set(pos_basis))
        if added0:
            print("\nAjouts d'arêtes 0 :", added0)
        tree_edges=spanning_tree_from_edges(costs,base_edges,m,n)
        print(f"Arbre utilisé (|E|={len(tree_edges)}) :",sorted(tree_edges))
        u,v=compute_potentials(costs,tree_edges)
        print_vector(u,"Potentiels u",fmt="{:>10.2f}")
        print_vector(v,"Potentiels v",fmt="{:>10.2f}")
        cp=potential_cost_table(u,v)
        d=marginal_costs(costs,u,v)
        print_matrix(cp,"Coûts potentiels",fmt="{:>10.2f}")
        print_matrix(d,"Coûts marginaux",fmt="{:>10.2f}")
        entering=choose_entering_edge(d,tree_edges)
        if entering is None:
            print("\nProposition OPTIMALE")
            break
        print(f"\nArête à ajouter = {entering} avec d_ij = {d[entering[0]][entering[1]]:.2f}")
        x=improve_solution_on_cycle(x,costs,tree_edges,entering)
        it+=1
    print("\n================= SOLUTION OPTIMALE =================")
    print_matrix(x,"Proposition optimale X*",fmt="{:>10.2f}")
    print(f"\nCoût total optimal = {total_cost(costs,x):.2f}")
    return x

#Complexité
# =========================
# Benchmark
# =========================

import matplotlib
matplotlib.use('TkAgg')  # backend interactif pour PyCharm
import matplotlib.pyplot as plt
import time

def benchmark_transport_graphic(N_values=[10,40, 100], NB_RUNS=100):
    """
    Benchmark des méthodes Northwest Corner et Balas-Hammer.
    Retourne toutes les données et trace 6 sous-graphes pour :
    θNO, θBH, tNO, tBH, θNO+tNO, θBH+tBH
    """
    # Stockage complet
    all_data = {k: [] for k in ['n','thetaNO','thetaBH','tNO','tBH','thetaNO_tNO','thetaBH_tBH']}

    for n in N_values:
        for run in range(NB_RUNS):
            # Génération du problème
            costs, supply, demand = generate_random_transport_problem(n)
            costs, supply, demand = balance_problem(costs, supply, demand)

            # Northwest Corner
            start = time.time()
            xNO = northwest_corner(costs, supply, demand, verbose=False)
            elapsed_NO = time.time() - start
            cost_NO = total_cost(costs, xNO)

            # Balas-Hammer
            start = time.time()
            xBH = balas_hammer(costs, supply, demand, verbose=False)
            elapsed_BH = time.time() - start
            cost_BH = total_cost(costs, xBH)

            # Stockage
            all_data['n'].append(n)
            all_data['thetaNO'].append(cost_NO)
            all_data['thetaBH'].append(cost_BH)
            all_data['tNO'].append(elapsed_NO)
            all_data['tBH'].append(elapsed_BH)
            all_data['thetaNO_tNO'].append(cost_NO + elapsed_NO)
            all_data['thetaBH_tBH'].append(cost_BH + elapsed_BH)

    # --- Tracé graphique avec 6 sous-graphes ---
    metrics = ['thetaNO', 'thetaBH', 'tNO', 'tBH', 'thetaNO_tNO', 'thetaBH_tBH']
    titles = ['θNO', 'θBH', 'tNO', 'tBH', 'θNO+tNO', 'θBH+tBH']

    plt.figure(figsize=(15,10))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        plt.scatter(all_data['n'], all_data[metric], alpha=0.5, s=10)
        plt.xlabel('n')
        plt.ylabel(titles[i-1])
        plt.title(titles[i-1])
        plt.grid(True)

    plt.tight_layout()
    plt.show(block=True)

    return all_data

# Exemple d'utilisation
data = benchmark_transport_graphic(N_values=[10, 40, 100], NB_RUNS=100)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1️⃣ Calcul des maxima pour chaque n
# =========================
def compute_max_per_n(all_data, N_values):
    """
    Pour chaque n, retourne la valeur maximale de chaque métrique
    sur les NB_RUNS réalisations.
    """
    max_data = {k: [] for k in ['thetaNO','thetaBH','tNO','tBH','thetaNO_tNO','thetaBH_tBH']}

    for n in N_values:
        # Indices correspondant à ce n
        indices = [i for i, val in enumerate(all_data['n']) if val == n]
        for key in max_data.keys():
            max_val = max(all_data[key][i] for i in indices)
            max_data[key].append(max_val)

    return max_data

# =========================
# 2️⃣ Tracé log-log des maxima
# =========================
def plot_worst_case(N_values, max_data):
    """
    Trace les maxima pour chaque n sur un graphique log-log
    Toutes les courbes sont visibles avec couleurs et marqueurs distincts.
    """
    plt.figure(figsize=(12,6))

    # Couleurs et marqueurs distincts pour chaque métrique
    colors = ['blue', 'red', 'green', 'orange', 'cyan', 'magenta']
    markers = ['o', 's', '^', 'D', 'v', '*']
    labels = ['θNO','θBH','tNO','tBH','θNO+tNO','θBH+tBH']

    # Calculer la valeur maximale pour ajuster l'axe y
    global_max = max(max(v) for v in max_data.values())
    plt.ylim(bottom=1, top=global_max*1.2)

    # Tracé des courbes
    for i, key in enumerate(max_data.keys()):
        plt.plot(
            N_values,
            max_data[key],
            marker=markers[i],
            linestyle='-',
            color=colors[i],
            label=labels[i],
            markersize=6,
            linewidth=1.5
        )

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('n (log scale)', fontsize=12)
    plt.ylabel('Valeur maximale (log scale)', fontsize=12)
    plt.title('Complexité dans le pire cas', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show(block=True)

# =========================
# 3️⃣ Estimation automatique des complexités
# =========================
def estimate_complexity(N_values, max_data):
    """
    Estime la complexité à partir de la pente du log-log plot.
    Affiche la pente et une qualification standard (linéaire, quadratique, etc.)
    """
    print("Estimation de la complexité (pente log-log) :")
    log_n = np.log(N_values)
    for key, values in max_data.items():
        log_val = np.log(values)
        slope, _ = np.polyfit(log_n, log_val, 1)

        # Classification selon la pente
        if abs(slope) < 0.1:
            complexity = "Constant O(1)"
        elif slope < 0.9:
            complexity = "Logarithmic O(log n) ou quasi-linéaire O(n log n)"
        elif slope < 1.5:
            complexity = "Linear O(n)"
        elif slope < 2.5:
            complexity = "Quadratic O(n²)"
        elif slope < 5:
            complexity = f"Polynomial O(n^{slope:.2f})"
        else:
            complexity = f"Exponential O({slope:.2f}^n)"

        print(f"{key}: pente ≈ {slope:.2f} → {complexity}")


import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def compare_algorithms_worst_case(N_values, max_data):
    """
    Compare θNO+tNO et θBH+tBH pour chaque valeur de n.
    Trace les maxima dans le pire cas sur un log-log plot.
    """
    # Extraire les maxima
    thetaNO_tNO_max = max_data['thetaNO_tNO']
    thetaBH_tBH_max = max_data['thetaBH_tBH']

    plt.figure(figsize=(10, 6))

    # Tracé des deux courbes
    plt.plot(
        N_values,
        thetaNO_tNO_max,
        marker='o',
        linestyle='-',
        color='blue',
        markersize=6,
        linewidth=1.5,
        label='θNO + tNO'
    )
    plt.plot(
        N_values,
        thetaBH_tBH_max,
        marker='s',
        linestyle='-',
        color='red',
        markersize=6,
        linewidth=1.5,
        label='θBH + tBH'
    )

    # Log-log scale pour visualiser la complexité
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('n (log scale)', fontsize=12)
    plt.ylabel('Valeur maximale (θ + t)', fontsize=12)
    plt.title('Comparaison de la complexité dans le pire des cas', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show(block=True)

    # Affichage des valeurs maximales pour discussion
    print("\nValeurs maximales pour chaque n :")
    for i, n in enumerate(N_values):
        print(f"n={n} : θNO+tNO={thetaNO_tNO_max[i]:.2f}, θBH+tBH={thetaBH_tBH_max[i]:.2f}")


# =========================
# 4️⃣ Exemple d'utilisation
# =========================
if __name__ == "__main__":
    # Suppose que 'data' est ton dictionnaire retourné par benchmark_transport_graphic()
    # et que N_values contient les différentes tailles de n
    N_values = sorted(list(set(data['n'])))  # extraire les valeurs uniques de n
    max_data = compute_max_per_n(data, N_values)
    plot_worst_case(N_values, max_data)
    estimate_complexity(N_values, max_data)
    # N_values : liste des tailles de problème
    # max_data : dictionnaire des maxima calculés avec compute_max_per_n()
    compare_algorithms_worst_case(N_values, max_data)

