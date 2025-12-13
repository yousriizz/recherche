#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import math
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
    s = 0.0
    for i in range(len(costs)):
        for j in range(len(costs[0])):
            s += costs[i][j] * x[i][j]
    return s

def almost_zero(v: float, tol: float = 1e-9) -> bool:
    return abs(v) < tol


# =========================
# Lecture fichier .txt
# =========================

def read_transport_problem_txt(path: str) -> Tuple[List[List[float]], List[float], List[float]]:
    """
    Format (comme ton exemple) :
    ligne1 : m n
    puis m lignes : n coûts + 1 provision (dernière colonne)
    dernière ligne : n commandes (demandes)
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.strip().startswith("#")]

    m, n = map(int, lines[0].split())
    if len(lines) != 1 + m + 1:
        raise ValueError(f"Format invalide: attendu {1+m+1} lignes (hors vides/commentaires), obtenu {len(lines)}.")

    costs: List[List[float]] = []
    supply: List[float] = []

    for k in range(1, 1 + m):
        parts = list(map(float, lines[k].split()))
        if len(parts) != n + 1:
            raise ValueError(f"Ligne {k+1}: attendu {n+1} valeurs (n coûts + 1 provision).")
        costs.append(parts[:n])
        supply.append(parts[n])

    demand = list(map(float, lines[1 + m].split()))
    if len(demand) != n:
        raise ValueError(f"Dernière ligne: attendu {n} valeurs (commandes).")

    return costs, supply, demand


# =========================
# Équilibrage (si besoin)
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
        costs.append([0.0] * n)
        supply = supply + [diff]

    return costs, supply, demand


# =========================
# Proposition initiale : Coin Nord-Ouest
# =========================

def northwest_corner(costs: List[List[float]], supply: List[float], demand: List[float], verbose: bool = True) -> List[List[float]]:
    m, n = len(costs), len(costs[0])
    a = supply[:]
    b = demand[:]
    x = [[0.0] * n for _ in range(m)]

    i, j = 0, 0
    step = 1
    while i < m and j < n:
        q = min(a[i], b[j])
        x[i][j] = q
        a[i] -= q
        b[j] -= q

        if verbose:
            print(f"\n[Coin Nord-Ouest] Étape {step} : remplir ({i},{j}) avec {q}")
        step += 1

        if almost_zero(a[i]): i += 1
        if almost_zero(b[j]): j += 1

    return x


# =========================
# Proposition initiale : Balas-Hammer (Vogel)
# =========================

def balas_hammer(costs: List[List[float]], supply: List[float], demand: List[float], verbose: bool = True) -> List[List[float]]:
    m, n = len(costs), len(costs[0])
    a = supply[:]
    b = demand[:]
    x = [[0.0] * n for _ in range(m)]

    active_r = [True] * m
    active_c = [True] * n

    def row_penalty(i: int) -> float:
        vals = [costs[i][j] for j in range(n) if active_c[j]]
        if len(vals) == 0: return -math.inf
        if len(vals) == 1: return math.inf
        v1, v2 = sorted(vals)[:2]
        return v2 - v1

    def col_penalty(j: int) -> float:
        vals = [costs[i][j] for i in range(m) if active_r[i]]
        if len(vals) == 0: return -math.inf
        if len(vals) == 1: return math.inf
        v1, v2 = sorted(vals)[:2]
        return v2 - v1

    step = 1
    while any(active_r) and any(active_c):
        if verbose:
            print(f"\n[Balas-Hammer] Itération {step}")

        pr = [row_penalty(i) if active_r[i] else None for i in range(m)]
        pc = [col_penalty(j) if active_c[j] else None for j in range(n)]

        max_pr = max([v for v in pr if v is not None], default=-math.inf)
        max_pc = max([v for v in pc if v is not None], default=-math.inf)
        pmax = max(max_pr, max_pc)

        rows_max = [i for i in range(m) if pr[i] == pmax]
        cols_max = [j for j in range(n) if pc[j] == pmax]

        if verbose:
            print("  Pénalités lignes   :", pr)
            print("  Pénalités colonnes :", pc)
            print("  Pénalité maximale  :", pmax)
            print("  Lignes max         :", rows_max)
            print("  Colonnes max       :", cols_max)

        # Choix ligne/colonne : départage par plus petit coût minimal
        choose_row = False
        if rows_max and cols_max:
            best_row_min = min(min(costs[i][j] for j in range(n) if active_c[j]) for i in rows_max)
            best_col_min = min(min(costs[i][j] for i in range(m) if active_r[i]) for j in cols_max)
            choose_row = best_row_min <= best_col_min
        elif rows_max:
            choose_row = True

        if choose_row:
            i = min(rows_max, key=lambda r: min(costs[r][j] for j in range(n) if active_c[j]))
            j = min([j for j in range(n) if active_c[j]], key=lambda c: costs[i][c])
            if verbose:
                print(f"  → Choix de la ligne {i}, arête minimale en colonne {j} (coût {costs[i][j]})")
        else:
            j = min(cols_max, key=lambda c: min(costs[i][c] for i in range(m) if active_r[i]))
            i = min([i for i in range(m) if active_r[i]], key=lambda r: costs[r][j])
            if verbose:
                print(f"  → Choix de la colonne {j}, arête minimale en ligne {i} (coût {costs[i][j]})")

        q = min(a[i], b[j])
        x[i][j] = q
        a[i] -= q
        b[j] -= q

        if verbose:
            print(f"  Remplissage : x[{i},{j}] = {q}")

        if almost_zero(a[i]): active_r[i] = False
        if almost_zero(b[j]): active_c[j] = False

        step += 1

    return x


# =========================
# Graphe / base / potentiels
# =========================

Node = Tuple[str, int]   # ('r', i) ou ('c', j)
Cell = Tuple[int, int]   # (i, j)

def basis_positive(x: List[List[float]], tol: float = 1e-9) -> Set[Cell]:
    m, n = len(x), len(x[0])
    return {(i, j) for i in range(m) for j in range(n) if x[i][j] > tol}

def adjacency_from_edges(m: int, n: int, edges: Set[Cell]) -> Dict[Node, List[Node]]:
    adj: Dict[Node, List[Node]] = {}
    def add(u: Node, v: Node):
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    for (i, j) in edges:
        add(('r', i), ('c', j))
    for i in range(m):
        adj.setdefault(('r', i), [])
    for j in range(n):
        adj.setdefault(('c', j), [])
    return adj

def count_components(adj: Dict[Node, List[Node]], m: int, n: int) -> int:
    seen: Set[Node] = set()
    comps = 0
    for i in range(m):
        start = ('r', i)
        if start in seen:
            continue
        comps += 1
        q = deque([start])
        seen.add(start)
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
    # si des colonnes isolées non touchées (rare si tout ok), les compter aussi
    for j in range(n):
        start = ('c', j)
        if start not in seen:
            comps += 1
            q = deque([start])
            seen.add(start)
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        q.append(v)
    return comps

def make_connected_by_zero_edges(costs: List[List[float]], x: List[List[float]], edges: Set[Cell]) -> Tuple[Set[Cell], List[Cell]]:
    """
    Si le graphe des arêtes de base est non connexe, on ajoute des arêtes à 0 (non utilisées)
    (de coût minimal) pour connecter les composantes.
    """
    m, n = len(costs), len(costs[0])
    added: List[Cell] = []

    while True:
        adj = adjacency_from_edges(m, n, edges)
        comps = count_components(adj, m, n)
        if comps <= 1:
            break

        # Trouver composantes via BFS
        comp_id: Dict[Node, int] = {}
        cid = 0
        for node in adj.keys():
            if node in comp_id:
                continue
            q = deque([node])
            comp_id[node] = cid
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in comp_id:
                        comp_id[v] = cid
                        q.append(v)
            cid += 1

        # Choisir une arête (i,j) non présente qui relie 2 composantes différentes, coût minimal
        best = None
        best_cost = math.inf
        for i in range(m):
            for j in range(n):
                if (i, j) in edges:
                    continue
                a = comp_id[('r', i)]
                b = comp_id[('c', j)]
                if a != b and costs[i][j] < best_cost:
                    best_cost = costs[i][j]
                    best = (i, j)

        if best is None:
            break  # sécurité
        edges.add(best)
        if almost_zero(x[best[0]][best[1]]):
            added.append(best)

    return edges, added

def spanning_tree_from_edges(costs: List[List[float]], edges: Set[Cell], m: int, n: int) -> Set[Cell]:
    """
    Extrait un arbre couvrant (m+n-1 arêtes) depuis un ensemble d'arêtes (possiblement cyclique).
    Priorité : conserver d'abord les arêtes de base existantes au coût faible.
    """
    total_nodes = m + n
    need = total_nodes - 1
    parent = list(range(total_nodes))
    rank = [0] * total_nodes

    def idx(node: Node) -> int:
        t, k = node
        return k if t == 'r' else m + k

    def find(u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u: int, v: int) -> bool:
        ru, rv = find(u), find(v)
        if ru == rv:
            return False
        if rank[ru] < rank[rv]:
            parent[ru] = rv
        elif rank[ru] > rank[rv]:
            parent[rv] = ru
        else:
            parent[rv] = ru
            rank[ru] += 1
        return True

    # Trier les arêtes par coût croissant (Kruskal)
    sorted_edges = sorted(list(edges), key=lambda e: costs[e[0]][e[1]])
    tree: Set[Cell] = set()
    for (i, j) in sorted_edges:
        if len(tree) >= need:
            break
        if union(idx(('r', i)), idx(('c', j))):
            tree.add((i, j))
    return tree

def compute_potentials(costs: List[List[float]], tree_edges: Set[Cell]) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """
    Calcule u,v sur un ARBRE (unique). Fixe u0=0 et propage.
    """
    m, n = len(costs), len(costs[0])
    adj = adjacency_from_edges(m, n, tree_edges)

    u: List[Optional[float]] = [None] * m
    v: List[Optional[float]] = [None] * n

    u[0] = 0.0
    q = deque([('r', 0)])

    while q:
        node = q.popleft()
        if node[0] == 'r':
            i = node[1]
            for nb in adj[node]:
                j = nb[1]
                if v[j] is None:
                    v[j] = costs[i][j] - u[i]  # type: ignore
                    q.append(('c', j))
        else:
            j = node[1]
            for nb in adj[node]:
                i = nb[1]
                if u[i] is None:
                    u[i] = costs[i][j] - v[j]  # type: ignore
                    q.append(('r', i))

    # Si jamais non connexe (devrait être rare), on complète
    for i in range(m):
        if u[i] is None:
            u[i] = 0.0
    for j in range(n):
        if v[j] is None:
            v[j] = 0.0

    return u, v

def potential_cost_table(u: List[float], v: List[float]) -> List[List[float]]:
    m, n = len(u), len(v)
    return [[u[i] + v[j] for j in range(n)] for i in range(m)]

def marginal_costs(costs: List[List[float]], u: List[float], v: List[float]) -> List[List[float]]:
    m, n = len(costs), len(costs[0])
    return [[costs[i][j] - (u[i] + v[j]) for j in range(n)] for i in range(m)]

def is_degenerate(basis_pos: Set[Cell], m: int, n: int) -> bool:
    return len(basis_pos) < (m + n - 1)


# =========================
# Cycle et amélioration
# =========================

def nodes_to_cell(u: Node, v: Node) -> Cell:
    if u[0] == 'r' and v[0] == 'c':
        return (u[1], v[1])
    if u[0] == 'c' and v[0] == 'r':
        return (v[1], u[1])
    raise ValueError("Arête invalide (doit relier une ligne à une colonne).")

def find_path_in_tree(adj: Dict[Node, List[Node]], start: Node, goal: Node) -> List[Node]:
    q = deque([start])
    parent: Dict[Node, Optional[Node]] = {start: None}
    while q:
        u = q.popleft()
        if u == goal:
            break
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                q.append(v)
    if goal not in parent:
        raise RuntimeError("Chemin introuvable (arbre non connexe).")
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

def choose_entering_edge(d: List[List[float]], basis_for_test: Set[Cell]) -> Optional[Cell]:
    """
    Arête entrante = d_ij minimal (le plus négatif) hors base (ici, hors l'arbre).
    """
    m, n = len(d), len(d[0])
    best = None
    best_val = 0.0
    for i in range(m):
        for j in range(n):
            if (i, j) in basis_for_test:
                continue
            if d[i][j] < best_val:
                best_val = d[i][j]
                best = (i, j)
    return best

def improve_solution_on_cycle(
    x: List[List[float]],
    costs: List[List[float]],
    tree_edges: Set[Cell],
    entering: Cell
) -> List[List[float]]:
    """
    Ajoute entering, forme un cycle avec l'arbre, fait l'ajustement (+/-) avec theta.
    """
    m, n = len(costs), len(costs[0])
    i0, j0 = entering

    adj = adjacency_from_edges(m, n, tree_edges)
    path_nodes = find_path_in_tree(adj, ('r', i0), ('c', j0))
    path_nodes_rev = list(reversed(path_nodes))  # ('c',j0) -> ... -> ('r',i0)

    cycle_cells: List[Cell] = [entering]
    for k in range(len(path_nodes_rev) - 1):
        cycle_cells.append(nodes_to_cell(path_nodes_rev[k], path_nodes_rev[k + 1]))

    plus_cells, minus_cells = [], []
    for idx, cell in enumerate(cycle_cells):
        (plus_cells if idx % 2 == 0 else minus_cells).append(cell)

    theta = min(x[i][j] for (i, j) in minus_cells)

    print("\n  Cycle formé (ordre) :", cycle_cells)
    print("  + cellules :", plus_cells)
    print("  - cellules :", minus_cells)
    print("  Theta (maximisation sur cycle) =", theta)

    for (i, j) in plus_cells:
        x[i][j] += theta
    for (i, j) in minus_cells:
        x[i][j] -= theta
        if almost_zero(x[i][j]):
            x[i][j] = 0.0

    return x


# =========================
# MODI / Marche-pied
# =========================

def modi_stepping_stone(costs: List[List[float]], x0: List[List[float]]) -> List[List[float]]:
    m, n = len(costs), len(costs[0])
    x = [row[:] for row in x0]

    it = 1
    while True:
        print(f"\n================= Itération MODI {it} =================")

        # Affichage proposition + coût
        print_matrix(x, "Proposition de transport X", fmt="{:>10.2f}")
        z = total_cost(costs, x)
        print(f"\nCoût total de transport = {z:.2f}")

        # Test dégénérescence
        pos_basis = basis_positive(x)
        deg = is_degenerate(pos_basis, m, n)
        print(f"\nTest dégénérescence : {'DÉGÉNÉRÉE' if deg else 'NON dégénérée'} "
              f"(nb arêtes positives={len(pos_basis)} ; attendu={m+n-1})")

        # Modifications graphe : connecter avec arêtes 0 si non connexe
        base_edges = set(pos_basis)
        base_edges, added0 = make_connected_by_zero_edges(costs, x, base_edges)

        if added0:
            print("\nModifications du graphe (non connexe) : ajouts d'arêtes 0 pour connecter :")
            for e in added0:
                print(f"  + ajout 0 : {e} coût={costs[e[0]][e[1]]}")
        else:
            print("\nGraphe : pas d'ajout 0 nécessaire pour la connexion.")

        # Cas cyclique : on extrait un arbre couvrant (affichage demandé “obtenir un arbre”)
        tree_edges = spanning_tree_from_edges(costs, base_edges, m, n)
        print(f"\nArbre utilisé (|E|={len(tree_edges)} attendu={m+n-1}) :", sorted(tree_edges))

        # Potentiels
        u_opt, v_opt = compute_potentials(costs, tree_edges)
        u = [float(v) for v in u_opt]
        v = [float(vv) for vv in v_opt]

        print_vector(u, "Potentiels u (lignes)", fmt="{:>10.2f}")
        print_vector(v, "Potentiels v (colonnes)", fmt="{:>10.2f}")

        # Tables
        cp = potential_cost_table(u, v)
        d = marginal_costs(costs, u, v)

        print_matrix(cp, "Table des coûts potentiels (u_i + v_j)", fmt="{:>10.2f}")
        print_matrix(d, "Table des coûts marginaux d_ij = c_ij - (u_i + v_j)", fmt="{:>10.2f}")

        # Optimalité
        entering = choose_entering_edge(d, tree_edges)
        if entering is None:
            print("\nProposition OPTIMALE (aucun d_ij négatif hors arbre).")
            break

        print(f"\nProposition NON optimale : arête à ajouter = {entering} avec d_ij = {d[entering[0]][entering[1]]:.2f}")

        # Maximisation sur cycle + nouvelle itération
        x = improve_solution_on_cycle(x, costs, tree_edges, entering)
        it += 1

    print("\n================= SOLUTION OPTIMALE =================")
    print_matrix(x, "Proposition optimale X*", fmt="{:>10.2f}")
    z = total_cost(costs, x)
    print(f"\nCoût total optimal = {z:.2f}")
    return x


# =========================
# Programme interactif
# =========================

def main():
    print("=== Transport : Proposition initiale + Marche-pied (MODI) ===")

    while True:
        # Choisir le numéro du problème
        num = input("\nChoisir le numéro du problème à traiter (ex: 1) : ").strip()
        if not num:
            print("Numéro vide. Fin.")
            return

        # Lire le tableau de contraintes sur fichier
        filename = f"probleme{num}.txt"
        print(f"\nLecture du fichier : {filename}")

        try:
            costs, supply, demand = read_transport_problem_txt(filename)
        except Exception as e:
            print(f"Erreur lecture fichier : {e}")
            again = input("Changer de problème ? (o/n) : ").strip().lower()
            if again != "o":
                break
            continue

        # Stocker en mémoire + équilibrage
        costs, supply, demand = balance_problem(costs, supply, demand)

        # Créer matrices correspondantes et afficher
        print_matrix(costs, "Matrice des coûts C", fmt="{:>10.2f}")
        print_vector(supply, "Provisions (offres) P_i", fmt="{:>10.2f}")
        print_vector(demand, "Commandes (demandes) C_j", fmt="{:>10.2f}")
        print(f"\nSomme provisions = {sum(supply):.2f} ; Somme commandes = {sum(demand):.2f}")

        # Choix proposition initiale
        print("\nChoisir l’algorithme pour fixer la proposition initiale :")
        print("  1) Coin Nord-Ouest")
        print("  2) Balas-Hammer (Vogel)")
        choice = input("Votre choix (1/2) : ").strip()

        if choice == "1":
            x0 = northwest_corner(costs, supply, demand, verbose=True)
        elif choice == "2":
            x0 = balas_hammer(costs, supply, demand, verbose=True)
        else:
            print("Choix invalide → Balas-Hammer par défaut.")
            x0 = balas_hammer(costs, supply, demand, verbose=True)

        # Affichages demandés après exécution
        print_matrix(x0, "\nProposition initiale X0", fmt="{:>10.2f}")
        print(f"\nCoût total de X0 = {total_cost(costs, x0):.2f}")

        # Marche-pied avec potentiels
        modi_stepping_stone(costs, x0)

        # Proposer de changer de problème
        again = input("\nChanger de problème de transport ? (o/n) : ").strip().lower()
        if again != "o":
            print("Fin.")
            break


if __name__ == "__main__":
    main()
