#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solveur du problème de transport (projet Recherche Opérationnelle).
Fonctions :
 - lecture .txt (format décrit dans le sujet)
 - affichages soignés (matrices, tables)
 - solutions initiales : Nord-Ouest et Balas-Hammer (Vogel)
 - méthode du marche-pied avec potentiels (u,v) et amélioration itérative
 - gestion dégénérescence (ajout d'allocations 0 pour atteindre m+n-1 basics)
 - sauvegarde des traces d'exécution
Usage :
    python3 transport_solver.py probleme.txt
Ou importer et appeler functions depuis un autre script.
"""
import sys
import math
import itertools
from collections import deque, defaultdict
from copy import deepcopy
from typing import List, Tuple, Optional

EPS = 1e-9  # tolérance

# ---------------------------
# Lecture du fichier .txt
# ---------------------------
def read_problem_from_file(path: str):
    """
    Lit le fichier .txt et renvoie (n, m, costs, supplies P, demands C)
    Format robuste :
    - première ligne : n m
    - ensuite n lignes contenant m coûts suivis (optionnellement) de la provision Pi
      si la ligne contient m+1 nombres : les m premiers sont ai,j et le dernier est Pi
      sinon si la ligne contient m nombres : on attend ensuite une ligne 'Provisions' ou P séparée
    - la dernière ligne après les n lignes contient les demandes C1..Cm (m valeurs)
    """
    with open(path, 'r', encoding='utf-8') as f:
        tokens = []
        for line in f:
            # ignorer commentaires/vides
            line = line.strip()
            if not line:
                continue
            # remplacer virgules par points si nécessaire
            line = line.replace(',', '.')
            # extraire tous les nombres de la ligne
            parts = line.split()
            # garder uniquement éléments qui ressemblent à des nombres
            nums = []
            for p in parts:
                # tenter conversion
                try:
                    if '.' in p:
                        nums.append(float(p))
                    else:
                        nums.append(int(p))
                except:
                    # ignorer mots non numériques
                    continue
            tokens.extend(nums)

    if len(tokens) < 2:
        raise ValueError("Fichier mal formé : pas assez d'informations.")

    # première lecture : n m
    n = int(tokens[0]); m = int(tokens[1])
    # maintenant on attend n*(m or m+1) + m chiffres au total, mais faire robuste.
    rest = tokens[2:]
    # On essaie de lire n lignes avec m coûts + 1 provision
    costs = [[0]*m for _ in range(n)]
    supplies = []
    idx = 0
    success = True
    try:
        for i in range(n):
            row = []
            for j in range(m):
                row.append(float(rest[idx])); idx += 1
            costs[i] = row
            # provision
            supplies.append(float(rest[idx])); idx += 1
    except Exception:
        # échec : essayer variante où les P sont générées plus tard
        success = False

    if not success:
        # réinitialiser et réessayer : lire n*m coûts, puis m demandes, puis extraire Pi et Cj de "temp" matrix if present
        rest = tokens[2:]
        if len(rest) < n*m + m:
            raise ValueError("Fichier non conforme : impossible d'extraire n*m coûts et m demandes.")
        idx = 0
        costs = []
        for i in range(n):
            row = []
            for j in range(m):
                row.append(float(rest[idx])); idx += 1
            costs.append(row)
        supplies = []
        # Suppose qu'après les n lignes de coûts, il y a une ligne de m demandes, puis une ligne 'CCC...' ou Pi elsewhere
        # Ici on suppose qu'après coûts il y a la ligne demandes (m)
        demands = []
        for j in range(m):
            demands.append(float(rest[idx])); idx += 1
        # if there are extra tokens, try to parse provisions from tail of file (rare)
        # fallback: generate supplies by summing random tmp matrix approach (but here we error)
        raise ValueError("Format non standard détecté. Utilisez le format : n m, n lignes (m coûts + provision), 1 ligne demandes.")

    # demandes
    demands = []
    try:
        for j in range(m):
            demands.append(float(rest[idx])); idx += 1
    except Exception:
        raise ValueError("Impossible de lire les demandes Cj. Vérifiez le format (dernière ligne : C1 .. Cm).")

    # vérifier équilibré
    totalP = sum(supplies)
    totalC = sum(demands)
    if abs(totalP - totalC) > 1e-6:
        raise ValueError(f"Le problème n'est pas équilibré : sum(P)={totalP} != sum(C)={totalC}.")

    return n, m, costs, supplies, demands

# ---------------------------
# Affichage soigné
# ---------------------------
def format_table_costs(costs: List[List[float]], supplies: List[float], demands: List[float]) -> str:
    n = len(costs); m = len(costs[0]) if n>0 else 0
    # largeur des colonnes
    colw = max(6, max(len(str(int(c))) for row in costs for c in row) + 1)
    s = ""
    # header
    header = "       " + "".join(f"{'C'+str(j+1):>{colw}}" for j in range(m)) + f"{'   Provision':>12}\n"
    s += header
    for i in range(n):
        row = f"{'P'+str(i+1):<6} "
        for j in range(m):
            row += f"{int(costs[i][j]):>{colw}}"
        row += f"{int(supplies[i]):>12}\n"
        s += row
    # demands
    dem_row = "Demand "
    for j in range(m):
        dem_row += f"{int(demands[j]):>{colw}}"
    s += dem_row + "\n"
    return s

def format_table_allocation(alloc: List[List[Optional[float]]], costs: List[List[float]]) -> str:
    n = len(alloc); m = len(alloc[0]) if n>0 else 0
    # determine column width
    def cellstr(i,j):
        a = alloc[i][j]
        if a is None:
            return "-"
        else:
            if abs(a - round(a)) < 1e-9:
                return str(int(round(a)))
            else:
                return f"{a:.2f}"
    colw = max(6, max(len(cellstr(i,j)) for i in range(n) for j in range(m)) + 1, max(len(str(int(costs[i][j]))) for i in range(n) for j in range(m)) +1)
    s = ""
    header = "       " + "".join(f"{'C'+str(j+1):>{colw}}" for j in range(m)) + "\n"
    s += header
    for i in range(n):
        row = f"{'P'+str(i+1):<6} "
        for j in range(m):
            cell = cellstr(i,j)
            row += f"{cell:>{colw}}"
        s += row + "\n"
    return s

def pretty_print(title: str, text: str):
    border = "="*8
    print(f"\n{border} {title} {border}\n")
    print(text)

# ---------------------------
# Solutions initiales
# ---------------------------
def northwest_corner(supplies: List[float], demands: List[float]) -> List[List[Optional[float]]]:
    """North-West corner method"""
    n = len(supplies); m = len(demands)
    alloc = [[None]*m for _ in range(n)]
    P = supplies.copy()
    C = demands.copy()
    i = 0; j = 0
    while i < n and j < m:
        q = min(P[i], C[j])
        alloc[i][j] = q
        P[i] -= q
        C[j] -= q
        if abs(P[i]) < EPS:
            i += 1
        if abs(C[j]) < EPS:
            j += 1
    return alloc

def vogel_approximation(costs: List[List[float]], supplies: List[float], demands: List[float]) -> List[List[Optional[float]]]:
    """Balas-Hammer ~ Vogel's approximation method"""
    n = len(supplies); m = len(demands)
    P = supplies.copy(); C = demands.copy()
    alloc = [[None]*m for _ in range(n)]
    rows_avail = set(range(n))
    cols_avail = set(range(m))

    while rows_avail and cols_avail:
        # compute penalties for rows and columns
        row_pen = {}
        for i in rows_avail:
            costs_row = sorted([costs[i][j] for j in cols_avail])
            if len(costs_row) >= 2:
                row_pen[i] = costs_row[1] - costs_row[0]
            elif len(costs_row) == 1:
                row_pen[i] = costs_row[0]
            else:
                row_pen[i] = -1
        col_pen = {}
        for j in cols_avail:
            costs_col = sorted([costs[i][j] for i in rows_avail])
            if len(costs_col) >= 2:
                col_pen[j] = costs_col[1] - costs_col[0]
            elif len(costs_col) == 1:
                col_pen[j] = costs_col[0]
            else:
                col_pen[j] = -1

        # find max penalty
        max_row = max(row_pen.items(), key=lambda x: (x[1], -x[0])) if row_pen else (None, None)
        max_col = max(col_pen.items(), key=lambda x: (x[1], -x[0])) if col_pen else (None, None)
        if max_row[1] is None and max_col[1] is None:
            break
        if max_row[1] >= max_col[1]:
            i = max_row[0]
            # choose cheapest column in available cols
            j = min(cols_avail, key=lambda jj: costs[i][jj])
        else:
            j = max_col[0]
            i = min(rows_avail, key=lambda ii: costs[ii][j])
        qty = min(P[i], C[j])
        alloc[i][j] = qty
        P[i] -= qty
        C[j] -= qty
        if abs(P[i]) < EPS:
            rows_avail.remove(i)
        if abs(C[j]) < EPS:
            cols_avail.remove(j)
    return alloc

# ---------------------------
# Utilitaires sur allocations
# ---------------------------
def basic_cells(alloc: List[List[Optional[float]]]) -> List[Tuple[int,int]]:
    n = len(alloc); m = len(alloc[0]) if n>0 else 0
    cells = []
    for i in range(n):
        for j in range(m):
            if alloc[i][j] is not None:
                cells.append((i,j))
    return cells

def is_degenerate(alloc: List[List[Optional[float]]]) -> bool:
    n = len(alloc); m = len(alloc[0]) if n>0 else 0
    return len(basic_cells(alloc)) < (n + m - 1)

def make_nondegenerate(alloc: List[List[Optional[float]]], costs: List[List[float]]):
    """
    Ajoute des allocations de 0 (fictives) sur les arêtes de plus faibles coûts
    jusqu'à atteindre n+m-1 basic cells.
    """
    n = len(alloc); m = len(alloc[0])
    needed = (n + m - 1) - len(basic_cells(alloc))
    if needed <= 0:
        return alloc
    # lister toutes les positions vides triées par coût croissant
    empties = [(costs[i][j], i, j) for i in range(n) for j in range(m) if alloc[i][j] is None]
    empties.sort()
    k = 0
    for _, i, j in empties:
        # ajouter 0 si cela ne crée pas un cycle (on peut toutefois accepter cycle car on gère cycles plus tard)
        alloc[i][j] = 0.0
        k += 1
        if k >= needed:
            break
    return alloc

def compute_total_cost(alloc: List[List[Optional[float]]], costs: List[List[float]]) -> float:
    total = 0.0
    n = len(alloc); m = len(alloc[0])
    for i in range(n):
        for j in range(m):
            if alloc[i][j] is not None and abs(alloc[i][j])>EPS:
                total += alloc[i][j] * costs[i][j]
    return total

# ---------------------------
# Graph and cycle detection (pour les basics)
# ---------------------------
def build_graph_from_alloc(alloc: List[List[Optional[float]]]) -> dict:
    """
    Construire un graphe bipartite à partir des cases basiques.
    Sommets : P1..Pn (rows) et C1..Cm (cols) ; arête entre Pi et Cj si alloc[i][j] is not None
    Représenté par dictionnaire adjacency list mapping vertex->set(neighbors)
    vs: utiliser 'r0','r1',... pour rows et 'c0','c1',... pour columns
    """
    n = len(alloc); m = len(alloc[0]) if n>0 else 0
    G = defaultdict(set)
    for i in range(n):
        for j in range(m):
            if alloc[i][j] is not None:
                ri = f"r{i}"
                cj = f"c{j}"
                G[ri].add(cj)
                G[cj].add(ri)
    return G

def is_connected_alloc(alloc: List[List[Optional[float]]]) -> bool:
    G = build_graph_from_alloc(alloc)
    if not G:
        return False
    start = next(iter(G))
    visited = set()
    q = deque([start])
    while q:
        u = q.popleft()
        if u in visited: continue
        visited.add(u)
        for v in G[u]:
            if v not in visited:
                q.append(v)
    # we expect all rows and columns that appear in basics to be connected
    return len(visited) == len(G)

def find_cycle_in_alloc(alloc: List[List[Optional[float]]], start_cell: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    """
    Trouver un cycle alternatif (entrelacé) commençant/terminant sur start_cell (i,j).
    Cycle must alternate between row-moves and col-moves and include only basic cells.
    Retourne liste ordonnée des cellules du cycle (i,j) ... si trouvé, sinon None.
    On cherche cycle simple (sans répétition).
    """
    n = len(alloc); m = len(alloc[0])
    basics = set(basic_cells(alloc))
    if start_cell not in basics:
        return None

    # backtracking DFS: state = current cell, visited cells, must alternate direction (0 row->col, 1 col->row)
    def neighbors(cell, dir):
        # dir=0 means move in same row to another basic cell (change col)
        i,j = cell
        res = []
        if dir == 0:
            for jj in range(m):
                if jj != j and (i,jj) in basics:
                    res.append((i,jj))
        else:
            for ii in range(n):
                if ii != i and (ii,j) in basics:
                    res.append((ii,j))
        return res

    visited = set()
    path = []

    # we need cycles of even length >=4
    def dfs(current, dir):
        path.append(current)
        visited.add(current)
        for nb in neighbors(current, dir):
            if nb == start_cell and len(path) >= 4 and len(path)%2==0:
                # cycle closed
                return path.copy()
            if nb not in visited:
                res = dfs(nb, 1-dir)
                if res:
                    return res
        # backtrack
        visited.remove(current)
        path.pop()
        return None

    return dfs(start_cell, 0)

# ---------------------------
# Potentiels u,v et coûts réduits
# ---------------------------
def compute_potentials(alloc: List[List[Optional[float]]], costs: List[List[float]]):
    """
    Calcule u[i], v[j] tels que pour chaque arête basique (i,j): u[i] + v[j] = cost[i][j]
    On fixe u[0] = 0 puis solving by propagation on básicos.
    Retourne (u, v)
    """
    n = len(alloc); m = len(alloc[0])
    u = [None]*n
    v = [None]*m
    basics = basic_cells(alloc)
    if not basics:
        return u, v
    u[0] = 0.0
    # iteratively propagate
    changed = True
    while changed:
        changed = False
        for i,j in basics:
            if u[i] is not None and v[j] is None:
                v[j] = costs[i][j] - u[i]; changed = True
            if v[j] is not None and u[i] is None:
                u[i] = costs[i][j] - v[j]; changed = True
    return u, v

def compute_reduced_costs(alloc: List[List[Optional[float]]], costs: List[List[float]]):
    """Pour toutes les cases non basiques, calculer reduced_cost = cost - (u+v)"""
    n = len(alloc); m = len(alloc[0])
    u, v = compute_potentials(alloc, costs)
    reduced = [[None]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if alloc[i][j] is None:
                # si u ou v est None -> on ne peut pas calculer -> laisser None
                if u[i] is None or v[j] is None:
                    reduced[i][j] = None
                else:
                    reduced[i][j] = costs[i][j] - (u[i] + v[j])
    return reduced, u, v

def find_entering_edge(reduced_costs: List[List[Optional[float]]]) -> Optional[Tuple[int,int,float]]:
    n = len(reduced_costs); m = len(reduced_costs[0]) if n>0 else 0
    best = None
    for i in range(n):
        for j in range(m):
            val = reduced_costs[i][j]
            if val is None:
                continue
            # recherche coût réduit strictement négatif (améliorant)
            if val < -EPS:
                if best is None or val < best[2]:
                    best = (i,j,val)
    return best

# ---------------------------
# Maximisation sur cycle (ajout d'arête)
# ---------------------------
def apply_cycle_improvement(alloc: List[List[Optional[float]]], entering: Tuple[int,int], costs: List[List[float]], trace_print=None):
    """
    Ajoute l'arête entering (i,j) à la base (temporarily), trouve cycle, détermine delta, effectue ajustement:
    - marque + sur entering, ensuite -,+,-,... le long du cycle
    - delta = min valeur sur positions avec '-' (sauf celles with zero)
    - retirer les arêtes qui deviennent nulles (ou mettre None) ; conserver les zeros nécessaires à non-dégénérescence
    """
    i0,j0 = entering
    # assure qu'on a une valeur None à entering
    if alloc[i0][j0] is not None:
        raise ValueError("Entering cell is already basic.")
    # ajouter temporairement la cellule à basics
    alloc[i0][j0] = 0.0
    cyc = find_cycle_in_alloc(alloc, (i0,j0))
    if cyc is None:
        # pas de cycle trouvé (peut arriver si structure non connexe)
        # on retire cellule temporaire et échoue
        alloc[i0][j0] = None
        return False, "No cycle found"
    # ordonner cycle en alternant signes + - + - ... avec + at entering
    # find index of entering in cyc
    idx0 = cyc.index((i0,j0))
    ordered = cyc[idx0:] + cyc[:idx0]  # start at entering
    signs = []
    for k in range(len(ordered)):
        signs.append(1 if k%2==0 else -1)
    # trouver delta = min allocation on '-' positions
    minus_positions = [ordered[k] for k in range(len(ordered)) if signs[k]==-1]
    delta = min( (alloc[i][j] for (i,j) in minus_positions) )
    # effectue ajustement
    for k, (i,j) in enumerate(ordered):
        if signs[k] == 1:
            alloc[i][j] = (alloc[i][j] if alloc[i][j] is not None else 0.0) + delta
        else:
            alloc[i][j] = alloc[i][j] - delta
    # retirer arêtes devenues nulles (set None)
    removed = []
    for (i,j) in minus_positions:
        if abs(alloc[i][j]) < EPS:
            # pour éviter degenerescence complète, on peut les mettre à 0.0 ou None.
            # enlever et revenir à None (mais attention à degenerescence globale)
            alloc[i][j] = None
            removed.append((i,j))
    # si l'opération a retiré plusieurs arêtes et provoque dégénérescence, caller devra appeler make_nondegenerate si besoin
    if trace_print:
        trace_print("Cycle trouvé: " + " -> ".join(f"({i+1},{j+1})" for (i,j) in ordered))
        trace_print(f"Delta = {delta}. Arêtes supprimées: {removed}")
    return True, {"cycle":ordered, "delta":delta, "removed":removed}

# ---------------------------
# Méthode du marche-pied (Stepping-stone / potentiel)
# ---------------------------
def stepping_stone_method(costs: List[List[float]], alloc_init: List[List[Optional[float]]], max_iters=1000, verbose=False, trace_writer=None):
    """
    Applique la méthode du marche-pied (potentiels) à partir d'une allocation initiale (alloc_init),
    jusqu'à optimalité (toutes reduced_costs >= 0) ou itérations max.
    Renvoie allocation finale, coût, nombre d'itérations, and trace as list of strings if trace_writer omitted.
    """
    alloc = deepcopy(alloc_init)
    n = len(alloc); m = len(alloc[0])
    # s'assurer non dégénérée initialement
    if is_degenerate(alloc):
        alloc = make_nondegenerate(alloc, costs)
        if trace_writer:
            trace_writer("Proposition initiale dégénérée: ajout d'allocations fictives 0 pour non-dégénérescence.")
    it = 0
    trace = []
    def tprint(s):
        trace.append(s)
        if trace_writer:
            trace_writer(s)
        if verbose:
            print(s)
    tprint(f"--- Début méthode marche-pied (itération max = {max_iters}) ---")
    while it < max_iters:
        it += 1
        # calcul potentiels et reduced costs
        reduced, u, v = compute_reduced_costs(alloc, costs)
        tprint(f"\nItération {it}")
        # affichage allocation et coût
        tprint("Allocation courante:")
        tprint(format_table_allocation(alloc, costs))
        total = compute_total_cost(alloc, costs)
        tprint(f"Coût total actuel = {total}")
        # afficher potentiels
        tprint(f"Potentiels u (rows): {['{:.2f}'.format(x) if x is not None else None for x in u]}")
        tprint(f"Potentiels v (cols): {['{:.2f}'.format(x) if x is not None else None for x in v]}")
        # afficher coûts marginaux (reduced)
        tprint("Coûts marginaux (réduits) pour cases non basiques (None signifie non calculable):")
        s = ""
        for i in range(n):
            row = ""
            for j in range(m):
                val = reduced[i][j]
                row += f"{val:8.2f}" if val is not None else f"{'  None':>8}"
            s += f"r{i+1}: {row}\n"
        tprint(s)
        # trouver entering edge
        entering = find_entering_edge(reduced)
        if entering is None:
            tprint("Aucune case améliorante (tous coûts réduits >= 0) => solution optimale atteinte.")
            break
        (ei,ej,val) = entering
        tprint(f"Arête améliorante détectée en ({ei+1},{ej+1}) avec coût réduit {val:.4f}")
        ok, info = apply_cycle_improvement(alloc, (ei,ej), costs, trace_print=tprint)
        if not ok:
            tprint("Amélioration impossible (pas de cycle) -> arrêt.")
            break
        # après modification, vérifier connexité et dégénérescence, corriger si nécessaire
        if not is_connected_alloc(alloc):
            tprint("La proposition devient non connexe après amélioration. Tentative de réparation : ajout d'arêtes de plus faible coût.")
            # ajouter arêtes (0 allocations) par coût croissant jusqu'à connexe
            empties = [(costs[i][j], i, j) for i in range(n) for j in range(m) if alloc[i][j] is None]
            empties.sort()
            for _, i, j in empties:
                alloc[i][j] = 0.0
                if is_connected_alloc(alloc):
                    break
            tprint("Réparation effectuée.")
        # si dégénérescence survenue, rendre non dégénérée
        if is_degenerate(alloc):
            make_nondegenerate(alloc, costs)
            tprint("Dégénérescence détectée après amélioration -> ajout d'allocations fictives 0.")
    else:
        tprint("Nombre maximum d'itérations atteint sans convergence garantie.")
    final_cost = compute_total_cost(alloc, costs)
    tprint(f"--- Fin méthode (itérations = {it}). Coût final = {final_cost} ---")
    return alloc, final_cost, it, trace

# ---------------------------
# Fonctions utilitaires I/O trace fichiers
# ---------------------------
def write_trace_to_file(trace_lines: List[str], outpath: str):
    with open(outpath, 'w', encoding='utf-8') as f:
        for line in trace_lines:
            f.write(line + "\n")

# ---------------------------
# Exemple d'utilisation et CLI
# ---------------------------
def solve_file_and_save_traces(path: str, save_prefix="trace"):
    n, m, costs, supplies, demands = read_problem_from_file(path)
    print("Problème lu :", n, "fournisseurs,", m, "clients.")
    pretty_print("Matrice des coûts", format_table_costs(costs, supplies, demands))

    # Nord-Ouest
    alloc_no = northwest_corner(supplies, demands)
    if is_degenerate(alloc_no):
        alloc_no = make_nondegenerate(alloc_no, costs)
    trace_no = []
    def writer_no(s): trace_no.append(s)
    alloc_no_final, cost_no_final, it_no, trace_steps_no = stepping_stone_method(costs, alloc_no, verbose=False, trace_writer=writer_no)
    # save trace
    write_trace_to_file(trace_no + trace_steps_no, f"{save_prefix}-no.txt")
    print(f"Nord-Ouest: coût final = {cost_no_final}, trace sauvegardée {save_prefix}-no.txt")

    # Balas-Hammer (Vogel)
    alloc_bh = vogel_approximation(costs, supplies, demands)
    if is_degenerate(alloc_bh):
        alloc_bh = make_nondegenerate(alloc_bh, costs)
    trace_bh = []
    def writer_bh(s): trace_bh.append(s)
    alloc_bh_final, cost_bh_final, it_bh, trace_steps_bh = stepping_stone_method(costs, alloc_bh, verbose=False, trace_writer=writer_bh)
    write_trace_to_file(trace_bh + trace_steps_bh, f"{save_prefix}-bh.txt")
    print(f"Balas-Hammer: coût final = {cost_bh_final}, trace sauvegardée {save_prefix}-bh.txt")

    # Résumé
    print("\n--- Résumé ---")
    print(f"Nord-Ouest coût = {cost_no_final}")
    print(f"Balas-Hammer coût = {cost_bh_final}")

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        path = sys.argv[1]
        prefix = "trace"
        if len(sys.argv) >= 3:
            prefix = sys.argv[2]
        solve_file_and_save_traces(path, save_prefix=prefix)
    else:
        # si pas de fichier fourni, on affiche un petit exemple embarqué
        print("Aucun fichier fourni. Exécution d'un petit exemple embarqué.")
        # exemple simple (extrait du PDF)
        costs = [
            [30,20,20],
            [10,50,20],
            [50,40,30],
            [30,20,30]
        ]
        supplies = [450,250,250,450]
        demands = [500,600,300]
        # vérifier équilibre
        if abs(sum(supplies)-sum(demands))>EPS:
            print("Exemple non équilibré - ajustement pour l'exemple.")
        pretty_print("Matrice coûts (exemple)", format_table_costs(costs, supplies, demands))
        alloc_no = northwest_corner(supplies, demands)
        pretty_print("Proposition initiale Nord-Ouest", format_table_allocation(alloc_no, costs))
        alloc_bh = vogel_approximation(costs, supplies, demands)
        pretty_print("Proposition initiale Balas-Hammer", format_table_allocation(alloc_bh, costs))
        # lancer méthode marche-pied sur Nord-Ouest et afficher coût final
        alloc_no_final, cost_no_final, it_no, trace_no = stepping_stone_method(costs, alloc_no, verbose=True)
        pretty_print("Solution finale (Nord-Ouest départ)", format_table_allocation(alloc_no_final, costs))
        print("Coût final =", cost_no_final)
