import os

# ================================================================
# AFFICHAGE GENERIQUE D'UN TABLEAU (BORDURES FINES)
# ================================================================
def afficher_tableau(headers, donnees):
    """
    Affiche un tableau ASCII avec bordures fines (Unicode),
    colonnes centrées, et lignes séparatrices.
    """
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
    """
    Format attendu :
        n m
        a11 a12 ... a1m P1
        ...
        an1 an2 ... anm Pn
        C1 C2 ... Cm
    """
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
    """
    Construit headers + donnees pour représenter :
          C1  C2 ... Cm   P
    P1   a11 a12 ... a1m P1
    ...
    Pn   an1 ...     anm Pn
    Commandes C1 C2 ... Cm
    """
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
    """
    Crée une matrice B de taille n x m initialisée à 0.
    B[i][j] = quantité de Pi vers Cj.
    """
    return [[0 for _ in range(m)] for _ in range(n)]


def construire_tableau_proposition(B):
    """
    Tableau pour afficher la proposition B :
          C1  C2 ... Cm
    P1   b11 b12 ... b1m
    ...
    Pn   bn1 ...     bnm
    """
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
    """
    Tableau pour afficher :
        Sommet   Potentiel
        P1       u1
        ...
        Pn       un
        C1       v1
        ...
        Cm       vm
    """
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
    """
    Tableau pour afficher une matrice M (ex: coûts marginaux Δ_ij) :
          C1   C2 ... Cm
    P1   M11 M12 ... M1m
    ...
    Pn   Mn1 ...    Mnm
    """
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
    """
    Algorithme de Nord-Ouest.
    Entrée :
        P : liste des provisions (P1..Pn)
        C : liste des commandes (C1..Cm)
    Sortie :
        B : matrice n x m des quantités b_ij
    On suppose le problème équilibré : sum(P) == sum(C).
    """

    n = len(P)
    m = len(C)

    # Matrice de transport initialisée à 0
    B = [[0 for _ in range(m)] for _ in range(n)]

    # Copies locales de P et C pour ne pas modifier les originales
    restP = P.copy()
    restC = C.copy()

    i = 0  # ligne (fournisseur)
    j = 0  # colonne (client)

    while i < n and j < m:
        # quantité qu'on peut mettre sur la case (i, j)
        x = min(restP[i], restC[j])
        B[i][j] = x

        restP[i] -= x
        restC[j] -= x

        # Si le fournisseur i est épuisé, on passe à la ligne suivante
        if restP[i] == 0 and i < n - 1:
            i += 1
        # Sinon, si le client j est servi, on passe à la colonne suivante
        elif restC[j] == 0 and j < m - 1:
            j += 1
        else:
            # Plus rien à affecter (dernier fournisseur ou dernier client)
            break

    return B


# ================================================================
# ALGORITHME BALAS-HAMMER (STUB POUR L'INSTANT)
# ================================================================
def balas_hammer(A, P, C):
    """
    Algorithme de Balas-Hammer.
    Pour l'instant : non implémenté, juste un message et renvoie une matrice de 0.
    On remplira ça plus tard.
    """
    print("\n[INFO] Balas-Hammer n'est pas encore implémenté.")
    n = len(P)
    m = len(C)
    return [[0 for _ in range(m)] for _ in range(n)]



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

        # Boucle de menu interne pour ce problème
        while True:
            print("\n--- MENU POUR LE PROBLEME", num_probleme, "---")
            print("1 - Afficher le tableau des coûts / provisions / commandes")
            print("2 - Afficher la proposition de transport actuelle")
            print("3 - Afficher la table des potentiels (u, v)")
            print("4 - Afficher la table des coûts marginaux")
            print("5 - Appliquer l'algorithme Nord-Ouest à la proposition de transport")
            print("6 - Appliquer l'algorithme Balas-Hammer à la proposition de transport")
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
                # Appliquer Nord-Ouest sur P, C -> B
                B = nord_ouest(P, C)
                headers_B, donnees_B = construire_tableau_proposition(B)
                print("\n--- Proposition de transport (Nord-Ouest) ---")
                afficher_tableau(headers_B, donnees_B)

                # Plus tard : recalculer u, v, Delta ici

            elif choix_action == "6":
                # Appliquer Balas-Hammer (stub pour l’instant)
                B = balas_hammer(A, P, C)
                headers_B, donnees_B = construire_tableau_proposition(B)
                print("\n--- Proposition de transport (Balas-Hammer) ---")
                afficher_tableau(headers_B, donnees_B)

                # Plus tard : recalculer u, v, Delta ici

            else:
                print("Choix invalide, merci de saisir un numéro du menu.")
