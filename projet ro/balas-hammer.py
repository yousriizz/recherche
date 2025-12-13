import math
# (cas où une seule case reste dans une ligne ou une colonne).


# =====================================================
# Lecture du fichier texte
# =====================================================
def lire_fichier(nom):
    # Ouvre le fichier texte en mode lecture
    with open(nom, "r") as f:
        # Lit toutes les lignes non vides
        lignes = [l.strip() for l in f if l.strip()]

    # Première ligne : nombre de lignes (m) et de colonnes (n)
    m, n = map(int, lignes[0].split())

    # Matrice des coûts
    couts = []

    # Vecteur des offres (capacités des sources)
    offres = []

    # Lecture des m lignes suivantes
    # Chaque ligne contient : n coûts + 1 offre
    for i in range(1, m + 1):
        ligne = list(map(float, lignes[i].split()))
        couts.append(ligne[:n])   # Les n premiers éléments sont les coûts
        offres.append(ligne[n])   # Le dernier élément est l’offre

    # Dernière ligne : demandes des destinations
    demandes = list(map(float, lignes[m + 1].split()))

    # Retourne les données du problème de transport
    return couts, offres, demandes


# =====================================================
# Équilibrage du problème
# =====================================================
def equilibrer(couts, offres, demandes):
    # Si la somme des offres est supérieure à la somme des demandes,
    # on ajoute une colonne fictive (demande artificielle)
    if sum(offres) > sum(demandes):
        # Quantité manquante côté demandes
        demandes.append(sum(offres) - sum(demandes))
        # Coût nul pour la colonne fictive
        for i in range(len(couts)):
            couts[i].append(0)

    # Si la somme des demandes est supérieure à la somme des offres,
    # on ajoute une ligne fictive (offre artificielle)
    elif sum(demandes) > sum(offres):
        offres.append(sum(demandes) - sum(offres))
        couts.append([0] * len(couts[0]))

    # Retourne le problème équilibré
    return couts, offres, demandes


# =====================================================
# Algorithme de Balas-Hammer (Vogel)
# =====================================================
def balas_hammer(couts, offres, demandes):
    # Nombre de lignes (origines) et colonnes (destinations)
    m, n = len(couts), len(couts[0])

    # Matrice solution initialisée à 0
    x = [[0] * n for _ in range(m)]

    # Indique si une ligne est encore active (offre non épuisée)
    lignes = [True] * m

    # Indique si une colonne est encore active (demande non satisfaite)
    colonnes = [True] * n

    # Boucle principale :
    # continue tant qu’il reste au moins une ligne et une colonne actives
    while any(lignes) and any(colonnes):

        # -------------------------------------------------
        # Calcul des pénalités pour les lignes
        # -------------------------------------------------
        pr = []
        for i in range(m):
            # Si la ligne est inactive, on lui donne une pénalité négative
            if not lignes[i]:
                pr.append(-1)
                continue

            # Coûts encore disponibles sur la ligne i
            vals = [couts[i][j] for j in range(n) if colonnes[j]]

            # Pénalité = différence entre les deux plus petits coûts
            # Si une seule case est disponible, pénalité infinie
            pr.append(sorted(vals)[1] - sorted(vals)[0]
                      if len(vals) > 1 else math.inf)

        # -------------------------------------------------
        # Calcul des pénalités pour les colonnes
        # -------------------------------------------------
        pc = []
        for j in range(n):
            # Si la colonne est inactive, pénalité négative
            if not colonnes[j]:
                pc.append(-1)
                continue

            # Coûts encore disponibles sur la colonne j
            vals = [couts[i][j] for i in range(m) if lignes[i]]

            # Même logique que pour les lignes
            pc.append(sorted(vals)[1] - sorted(vals)[0]
                      if len(vals) > 1 else math.inf)

        # -------------------------------------------------
        # Choix entre une ligne ou une colonne
        # -------------------------------------------------
        # On choisit celle ayant la pénalité maximale
        if max(pr) >= max(pc):
            # Ligne de pénalité maximale
            i = pr.index(max(pr))

            # Dans cette ligne, on choisit la colonne de coût minimal
            j = min(
                (j for j in range(n) if colonnes[j]),
                key=lambda j: couts[i][j]
            )
        else:
            # Colonne de pénalité maximale
            j = pc.index(max(pc))

            # Dans cette colonne, on choisit la ligne de coût minimal
            i = min(
                (i for i in range(m) if lignes[i]),
                key=lambda i: couts[i][j]
            )

        # -------------------------------------------------
        # Affectation de la quantité
        # -------------------------------------------------
        # Quantité transportée = minimum entre l’offre et la demande
        q = min(offres[i], demandes[j])

        # Affectation dans la matrice solution
        x[i][j] = q

        # Mise à jour des offres et demandes restantes
        offres[i] -= q
        demandes[j] -= q

        # Si l’offre de la ligne est épuisée, on désactive la ligne
        if offres[i] == 0:
            lignes[i] = False

        # Si la demande de la colonne est satisfaite, on désactive la colonne
        if demandes[j] == 0:
            colonnes[j] = False

    # Retourne la solution initiale obtenue par Balas-Hammer
    return x


# =====================================================
# Calcul du coût total
# =====================================================
def cout_total(couts, x):
    # Somme des produits : coût × quantité transportée
    return sum(
        couts[i][j] * x[i][j]
        for i in range(len(couts))
        for j in range(len(couts[0]))
    )


# =====================================================
# Programme principal
# =====================================================
# Demande à l’utilisateur le nom du fichier de données
fichier = input("Nom du fichier (.txt) : ")

# Lecture du problème de transport
couts, offres, demandes = lire_fichier(fichier)

# Équilibrage automatique si nécessaire
couts, offres, demandes = equilibrer(couts, offres, demandes)

# Application de l’algorithme de Balas-Hammer
solution = balas_hammer(couts, offres, demandes)

# Affichage de la solution
print("\nSolution Balas-Hammer :")
for ligne in solution:
    print(ligne)

# Calcul et affichage du coût total
print("\nCoût total =", cout_total(couts, solution))

