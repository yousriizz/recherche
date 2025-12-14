import os
from collections import defaultdict
from collections import deque

# ================================================================
# AFFICHAGE GENERIQUE D'UN TABLEAU (BORDURES FINES)
# ================================================================
def afficher_tableau(headers, donnees):
    col_widths = [
        max(len(str(x)) for x in [headers[i]] + [ligne[i] for ligne in donnees])
        for i in range(len(headers))
    ]

    def ligne_haut():
        return "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"

    def ligne_sep():
        return "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"

    def ligne_bas():
        return "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"

    def afficher_ligne(ligne):
        return "│" + "│".join(
            f" {str(ligne[i]).center(col_widths[i])} "
            for i in range(len(ligne))
        ) + "│"

    print(ligne_haut())
    print(afficher_ligne(headers))
    print(ligne_sep())

    for idx, ligne in enumerate(donnees):
        print(afficher_ligne(ligne))
        if idx < len(donnees) - 1:
            print(ligne_sep())

    print(ligne_bas())


# ================================================================
# LECTURE DU PROBLEME DE TRANSPORT DEPUIS /Fichier/
# ================================================================
def lire_probleme_transport(nom_fichier):
    chemin = os.path.join("Fichier", nom_fichier)

    with open(chemin, "r") as f:
        lignes = [ligne.strip() for ligne in f if ligne.strip()]

    # n et m
    n, m = map(int, lignes[0].split())

    A = []
    P = []

    # Coûts + Provisions
    for i in range(1, n + 1):
        valeurs = lignes[i].split()
        A.append([int(x) for x in valeurs[:m]])
        P.append(int(valeurs[m]))

    # Commandes
    C = [int(x) for x in lignes[n + 1].split()]

    return n, m, A, P, C


# ================================================================
# TABLEAU DES COUTS (A) + PROVISIONS (P) + COMMANDES (C)
# ================================================================
def construire_tableau_couts(A, P, C):
    headers = [""] + [f"C{j+1}" for j in range(len(C))] + ["P"]

    donnees = []
    for i, ligne in enumerate(A):
        donnees.append([f"P{i+1}"] + ligne + [P[i]])

    donnees.append(["Commandes"] + C + [""])

    return headers, donnees


# ================================================================
# PROPOSITION DE TRANSPORT (MATRICE B)
# ================================================================
def initialiser_proposition(n, m):
    return [[0 for _ in range(m)] for _ in range(n)]


def construire_tableau_proposition(B):
    n = len(B)
    m = len(B[0]) if n > 0 else 0

    headers = [""] + [f"C{j+1}" for j in range(m)]
    donnees = []

    for i, ligne in enumerate(B):
        donnees.append([f"P{i+1}"] + ligne)

    return headers, donnees


# ================================================================
# TABLE DES POTENTIELS (u POUR P_i, v POUR C_j)
# ================================================================
def construire_tableau_potentiels(u, v):

    donnees = []

    for i, val in enumerate(u):
        donnees.append([f"P{i+1}", val])

    for j, val in enumerate(v):
        donnees.append([f"C{j+1}", val])

    headers = ["Sommet", "Potentiel"]
    return headers, donnees


# ================================================================
# TABLE GENERIQUE POUR UNE MATRICE (ex : COUTS MARGINAUX)
# ================================================================
def construire_tableau_matrice(M):
    n = len(M)
    m = len(M[0]) if n > 0 else 0

    headers = [""] + [f"C{j+1}" for j in range(m)]
    donnees = []

    for i, ligne in enumerate(M):
        donnees.append([f"P{i+1}"] + ligne)

    return headers, donnees


# ================================================================
# ALGORITHME NORD-OUEST : PROPOSITION INITIALE
# ================================================================
def nord_ouest(P, C):

    n = len(P)
    m = len(C)

    B = [[0 for _ in range(m)] for _ in range(n)]

    restP = P.copy()
    restC = C.copy()

    i = 0  
    j = 0  

    while i < n and j < m:
        x = min(restP[i], restC[j])
        B[i][j] = x

        restP[i] -= x
        restC[j] -= x

        if restP[i] == 0 and i < n - 1:
            i += 1
        elif restC[j] == 0 and j < m - 1:
            j += 1
        else:
            break

    return B


# ================================================================
# ALGORITHME BALAS-HAMMER 
# ================================================================

def balas_hammer(A, P, C):
    n = len(P)
    m = len(C)

    B = [[0 for _ in range(m)] for _ in range(n)]
    restP = P.copy()
    restC = C.copy()

    lignes_actives = set(range(n))
    colonnes_actives = set(range(m))

    def deux_min(L):
        L = sorted(L)
        if len(L) == 1:
            return L[0], L[0]
        return L[0], L[1]

    while lignes_actives and colonnes_actives:
        penal_lignes = {}
        penal_colonnes = {}

        for i in lignes_actives:
            couts = [A[i][j] for j in colonnes_actives]
            c1, c2 = deux_min(couts)
            penal_lignes[i] = c2 - c1

        for j in colonnes_actives:
            couts = [A[i][j] for i in lignes_actives]
            c1, c2 = deux_min(couts)
            penal_colonnes[j] = c2 - c1

        headers_pen = [""] + [f"C{j+1}" for j in colonnes_actives] + ["Pénalité"]
        donnees_pen = []

        for i in lignes_actives:
            row = [f"P{i+1}"] + [A[i][j] for j in colonnes_actives] + [penal_lignes[i]]
            donnees_pen.append(row)

        row_col = ["Pénalité"]
        for j in colonnes_actives:
            row_col.append(penal_colonnes[j])
        row_col.append("")
        donnees_pen.append(row_col)

        afficher_tableau(headers_pen, donnees_pen)

        best_pen = -1
        choix = None
        choix_type = None

        for i, p in penal_lignes.items():
            if p > best_pen:
                best_pen = p
                choix = i
                choix_type = "L"

        for j, p in penal_colonnes.items():
            if p > best_pen:
                best_pen = p
                choix = j
                choix_type = "C"

        if choix_type == "L":
            i = choix
            min_cout = min(A[i][j] for j in colonnes_actives)
            candidats = [j for j in colonnes_actives if A[i][j] == min_cout]
            j = max(candidats, key=lambda jj: min(restP[i], restC[jj]))
        else:
            j = choix
            min_cout = min(A[i][j] for i in lignes_actives)
            candidats = [i for i in lignes_actives if A[i][j] == min_cout]
            i = max(candidats, key=lambda ii: min(restP[ii], restC[j]))

        x = min(restP[i], restC[j])
        B[i][j] = x
        restP[i] -= x
        restC[j] -= x

        headers_B, donnees_B = construire_tableau_proposition(B)
        afficher_tableau(headers_B, donnees_B)

        if restP[i] == 0:
            lignes_actives.remove(i)
        if restC[j] == 0:
            colonnes_actives.remove(j)

    return B


# ================================================================
# ALGORITHME MARCHE PIED
# ================================================================

def construire_graphe(B):
    n = len(B)
    m = len(B[0]) if n > 0 else 0
    graphe = defaultdict(list)

    for i in range(n):
        for j in range(m):
            if B[i][j] > 0:
                p = ('P', i)
                c = ('C', j)
                graphe[p].append(c)
                graphe[c].append(p)
    return graphe



def detecter_cycle_bfs(graphe):
    visited = set()
    parent = {}

    for sommet_depart in graphe:
        if sommet_depart not in visited:
            queue = deque([sommet_depart])
            visited.add(sommet_depart)
            parent[sommet_depart] = None

            while queue:
                u = queue.popleft()

                for v in graphe[u]:
                    if v not in visited:
                        visited.add(v)
                        parent[v] = u
                        queue.append(v)

                    elif parent[u] != v:
                        cycle = reconstruire_cycle(parent, u, v)
                        return True, cycle

    return False, []


def reconstruire_cycle(parent, u, v):
    chemin_u = []
    x = u
    while x is not None:
        chemin_u.append(x)
        x = parent[x]

    chemin_v = []
    x = v
    while x is not None:
        chemin_v.append(x)
        x = parent[x]

    for s in chemin_u:
        if s in chemin_v:
            idx_u = chemin_u.index(s)
            idx_v = chemin_v.index(s)
            return chemin_u[:idx_u+1] + chemin_v[:idx_v][::-1]

    return []

from collections import deque

def composantes_connexes(graphe):
    visited = set()
    composantes = []

    for sommet in graphe:
        if sommet not in visited:
            comp = []
            queue = deque([sommet])
            visited.add(sommet)

            while queue:
                u = queue.popleft()
                comp.append(u)
                for v in graphe[u]:
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)

            composantes.append(comp)

    return composantes

def maximiser_sur_cycle(B, cycle):
    edges = []

    for k in range(len(cycle)):
        a = cycle[k]
        b = cycle[(k + 1) % len(cycle)]
        if a[0] == 'P':
            edges.append((a[1], b[1]))
        else:
            edges.append((b[1], a[1]))

    moins = edges[1::2]

    theta = min(B[i][j] for i, j in moins)

    for idx, (i, j) in enumerate(edges):
        if idx % 2 == 0:
            B[i][j] += theta
        else:
            B[i][j] -= theta

    return B

def corriger_degenerescence(B):
    n = len(B)
    m = len(B[0])
    arêtes = sum(1 for i in range(n) for j in range(m) if B[i][j] > 0)

    for i in range(n):
        for j in range(m):
            if arêtes >= n + m - 1:
                break
            if B[i][j] == 0:
                B[i][j] = 1  
                arêtes += 1

    return B


def calculer_potentiels(A, B):
    n = len(B)
    m = len(B[0])
    u = [None] * n
    v = [None] * m


    u[0] = 0
    queue = [0]  

    while queue:
        i = queue.pop(0)
        for j in range(m):
            if B[i][j] > 0:
                if v[j] is None:
                    v[j] = A[i][j] - u[i]
                    for k in range(n):
                        if B[k][j] > 0 and u[k] is None:
                            u[k] = A[k][j] - v[j]
                            queue.append(k)

    for idx in range(n):
        if u[idx] is None:
            u[idx] = 0
    for idx in range(m):
        if v[idx] is None:
            v[idx] = 0

    return u, v


def calculer_couts_marginaux(A, B, u, v):
    n = len(B)
    m = len(B[0])
    Delta = [[0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            if B[i][j] == 0:
                Delta[i][j] = A[i][j] - (u[i] + v[j])
            else:
                Delta[i][j] = ""  

    return Delta

def meilleure_arete_ameliore(Delta):
    best = 0
    pos = None

    for i in range(len(Delta)):
        for j in range(len(Delta[0])):
            if Delta[i][j] != "" and Delta[i][j] < best:
                best = Delta[i][j]
                pos = (i, j)

    return pos, best

def methode_marche_pied(A, P, C):
    n = len(P)
    m = len(C)

    # -------------------------------
    # Proposition initiale : Nord-Ouest
    # -------------------------------
    B = nord_ouest(P, C)

    # -------------------------------
    # Corriger dégénérescence pour connexité
    # -------------------------------
    B = corriger_degenerescence(B)

    # -------------------------------
    # Construire graphe à partir de B
    # -------------------------------
    graphe = construire_graphe(B)

    # -------------------------------
    # Détection de cycle
    # -------------------------------
    has_cycle, cycle = detecter_cycle_bfs(graphe)
    print("\n--- Méthode du marche-pied ---")
    if has_cycle:
        print("Cycle détecté :", cycle)

        edges = []
        for k in range(len(cycle)):
            curr_type, curr_idx = cycle[k]
            next_type, next_idx = cycle[(k+1)%len(cycle)]
            if curr_type == 'P' and next_type == 'C':
                edges.append((curr_idx, next_idx))
            elif curr_type == 'C' and next_type == 'P':
                edges.append((next_idx, curr_idx))

        # -------------------------------
        # Maximisation du transport sur le cycle
        # -------------------------------
        min_val = float('inf')
        for k, (i,j) in enumerate(edges):
            if k % 2 == 1:  
                if B[i][j] < min_val:
                    min_val = B[i][j]
        for k, (i,j) in enumerate(edges):
            if k % 2 == 0:
                B[i][j] += min_val
            else:
                B[i][j] -= min_val
        print("Maximisation effectuée sur le cycle avec quantité =", min_val)
    else:
        print("Aucun cycle détecté")

    # -------------------------------
    # Test connexité
    # -------------------------------
    composantes = composantes_connexes(graphe)
    if len(composantes) == 1:
        print("Graphe connexe")
    else:
        print("Graphe non connexe")
        print("Composantes :", composantes)
        B = corriger_degenerescence(B)
        print("Proposition corrigée pour connexité")

    # -------------------------------
    # Calcul des potentiels u et v (robuste)
    # -------------------------------
    u, v = calculer_potentiels(A, B)

    headers_pot, donnees_pot = construire_tableau_potentiels(u, v)
    print("\n--- Table des potentiels (u,v) ---")
    afficher_tableau(headers_pot, donnees_pot)

    # -------------------------------
    # Calcul des coûts marginaux Δ
    # -------------------------------
    Delta = calculer_couts_marginaux(A, B, u, v)
    headers_delta, donnees_delta = construire_tableau_matrice(Delta)
    print("\n--- Table des coûts marginaux Δ ---")
    afficher_tableau(headers_delta, donnees_delta)

    # -------------------------------
    # Détection et ajout de la meilleure arête améliorante
    # -------------------------------
    pos, valeur = meilleure_arete_ameliore(Delta)
    if pos is not None:
        i,j = pos
        print(f"\nMeilleure arête améliorante : B[{i+1},{j+1}] avec Δ = {valeur}")
        B[i][j] = 1 
        print(f"Arête B[{i+1},{j+1}] ajoutée à la proposition")
    else:
        print("\nAucune arête améliorante détectée")

    return B



# ================================================================
# MAIN - INTERFACE UTILISATEUR
# ================================================================
if __name__ == "__main__":
    print("\n=== PROJET DE RECHERCHE OPERATIONNELLE ===\n")

    while True:
        # Choix du problème de transport
        print("Choix du problème de transport (1 à 12)")
        print("0 - Quitter")
        choix = input("Entrez un numéro de problème (0 pour quitter) : ")

        if choix.strip() in ("0", "q", "Q", "quit", "exit"):
            print("\n=== FIN DU PROGRAMME ===\n")
            break

        if not choix.isdigit() or not (1 <= int(choix) <= 12):
            print("Entrée invalide, merci de saisir un entier entre 1 et 12.\n")
            continue

        num_probleme = int(choix)
        nom_fichier = f"tableau_{num_probleme}.txt"

        try:
            n, m, A, P, C = lire_probleme_transport(nom_fichier)
        except FileNotFoundError:
            print(f"\n[ERREUR] Fichier {nom_fichier} introuvable dans le dossier 'Fichier/'.\n")
            continue

        print(f"\n=== Problème {num_probleme} chargé depuis {nom_fichier} ===")
        print(f"Dimensions : n = {n}, m = {m}\n")

        # Structures associées au problème
        B = initialiser_proposition(n, m)   # proposition de transport
        u = [0] * n                         # potentiels fournisseurs
        v = [0] * m                         # potentiels clients
        Delta = [[0 for _ in range(m)] for _ in range(n)]  # coûts marginaux

        while True:
            print("\n--- MENU POUR LE PROBLEME", num_probleme, "---")
            print("1 - Afficher le tableau des coûts / provisions / commandes")
            print("2 - Afficher la proposition de transport actuelle")
            print("3 - Afficher la table des potentiels (u, v)")
            print("4 - Afficher la table des coûts marginaux")
            print("5 - Appliquer l'algorithme Nord-Ouest à la proposition de transport")
            print("6 - Appliquer l'algorithme Balas-Hammer à la proposition de transport")
            print("7 - Appliquer la méthode du marche-pied")
            print("0 - Revenir au choix du problème")

            choix_action = input("Votre choix : ").strip()

            if choix_action == "0":
                print("\nRetour au menu de choix du problème.\n")
                break

            elif choix_action == "1":
                headers_couts, donnees_couts = construire_tableau_couts(A, P, C)
                print("\n--- Tableau des coûts / provisions / commandes ---")
                afficher_tableau(headers_couts, donnees_couts)

            elif choix_action == "2":
                headers_B, donnees_B = construire_tableau_proposition(B)
                print("\n--- Proposition de transport actuelle ---")
                afficher_tableau(headers_B, donnees_B)

            elif choix_action == "3":
                headers_pot, donnees_pot = construire_tableau_potentiels(u, v)
                print("\n--- Table des potentiels (u pour P_i, v pour C_j) ---")
                afficher_tableau(headers_pot, donnees_pot)

            elif choix_action == "4":
                headers_delta, donnees_delta = construire_tableau_matrice(Delta)
                print("\n--- Table des coûts marginaux ---")
                afficher_tableau(headers_delta, donnees_delta)

            elif choix_action == "5":
                B = nord_ouest(P, C)
                headers_B, donnees_B = construire_tableau_proposition(B)
                print("\n--- Proposition de transport (Nord-Ouest) ---")
                afficher_tableau(headers_B, donnees_B)


            elif choix_action == "6":
                B = balas_hammer(A, P, C)
                headers_B, donnees_B = construire_tableau_proposition(B)
                print("\n--- Proposition de transport (Balas-Hammer) ---")
                afficher_tableau(headers_B, donnees_B)

            elif choix_action == "7":
                B = methode_marche_pied(A, P, C)


            else:
                print("Choix invalide, merci de saisir un numéro du menu.")



