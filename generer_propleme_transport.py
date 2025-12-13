import random
# Assurez-vous que le fichier ProblemeDeTransport.py est accessible
from ProblemeDeTransport import ProblemeDeTransport


def generer_probleme_transport(n, m):
    """
    Génère une instance de la classe ProblemeDeTransport avec des données aléatoires.

    Args:
        n (int): Nombre de fournisseurs (lignes)
        m (int): Nombre de clients (colonnes)

    Returns:
        ProblemeDeTransport: L'objet prêt à être résolu.
    """
    # 1. Génération des Coûts (1 à 100)
    couts = [[random.randint(1, 100) for _ in range(m)] for _ in range(n)]

    # 2. Génération de la matrice temporaire pour équilibrer Offre/Demande
    temp = [[random.randint(1, 100) for _ in range(m)] for _ in range(n)]

    # Calcul des Provisions (Somme des lignes de temp)
    provisions = []
    for i in range(n):
        provisions.append(sum(temp[i]))

    # Calcul des Commandes (Somme des colonnes de temp)
    commandes = []
    for j in range(m):
        col_sum = sum(temp[i][j] for i in range(n))
        commandes.append(col_sum)

    # 3. Création et peuplement de l'objet ProblemeDeTransport
    pb = ProblemeDeTransport()

    # Injection des données
    pb.n = n
    pb.m = m
    pb.couts = couts
    pb.provisions = provisions
    pb.commandes = commandes

    # Initialisation de la matrice proposition (vide pour l'instant)
    # C'est important car vos méthodes d'affichage l'utilisent
    pb.proposition = [[0.0 for _ in range(m)] for _ in range(n)]

    # Initialisation des structures internes (potentiels, etc.) si nécessaire
    pb.potentiels_u = [None] * n
    pb.potentiels_v = [None] * m

    return pb

