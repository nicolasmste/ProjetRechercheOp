import sys
import time
import random
import csv
from copy import deepcopy

# Import de la classe métier
try:
    from probleme_de_transport import ProblemeDeTransport
except ImportError:
    print("ERREUR CRITIQUE : Le fichier 'ProblemeDeTransport.py' est introuvable.")
    sys.exit(1)


# ================= FONCTIONS UTILITAIRES =================

def generer_probleme_transport_obj(n, m):
    """
    Génère un objet ProblemeDeTransport avec des données aléatoires.
    Garantit l'équilibre Offre = Demande via une matrice temporaire.
    """
    # Génération des coûts (1-100)
    couts = [[random.randint(1, 100) for _ in range(m)] for _ in range(n)]

    # Matrice temporaire pour générer des provisions/commandes équilibrées
    temp = [[random.randint(1, 100) for _ in range(m)] for _ in range(n)]

    provisions = [sum(temp[i]) for i in range(n)]
    commandes = [sum(temp[i][j] for i in range(n)) for j in range(m)]

    # Création de l'objet
    pb = ProblemeDeTransport()
    pb.n = n
    pb.m = m
    pb.couts = couts
    pb.provisions = provisions
    pb.commandes = commandes
    # Initialisation des structures internes
    pb.proposition = [[0.0 for _ in range(m)] for _ in range(n)]
    pb.potentiels_u = [None] * n
    pb.potentiels_v = [None] * m

    return pb


def lancer_etude_complexite():
    """
    Lance le benchmark sur différentes tailles de problèmes et enregistre les résultats dans un CSV.
    """
    print("\n" + "!" * 60)
    print("       LANCEMENT DE L'ÉTUDE DE COMPLEXITÉ (BENCHMARK)")
    print("!" * 60)

    # Tailles définies dans le PDF
    # Note : N=400 commence à prendre du temps, N=1000+ est très long en Python pur
    tailles_n = [10, 40, 100]
    nb_runs = 50  # Nombre d'exécutions par taille (PDF demande 100)

    resultats = []

    print(f"[INFO] Tailles à tester : {tailles_n}")
    print(f"[INFO] Runs par taille  : {nb_runs}")
    print("-" * 60)

    for n in tailles_n:
        print(f"\n>>> Traitement taille N = {n} ({n}x{n} = {n * n} variables)...")

        for k in range(nb_runs):
            # Barre de progression simple
            if k % 10 == 0:
                sys.stdout.write(f"\r    Progression : {k}/{nb_runs} runs")
                sys.stdout.flush()

            # 1. Génération
            pb_base = generer_probleme_transport_obj(n, n)

            # Copies profondes pour isoler les tests
            pb_no = deepcopy(pb_base)
            pb_bh = deepcopy(pb_base)

            # 2. Mesures Nord-Ouest + Marche-Pied
            t0 = time.perf_counter()
            pb_no.nord_ouest(verbose=False)
            theta_no = time.perf_counter() - t0

            t0 = time.perf_counter()
            pb_no.marche_pied_resolution(verbose=False)
            t_no = time.perf_counter() - t0

            # 3. Mesures Balas-Hammer + Marche-Pied
            t0 = time.perf_counter()
            pb_bh.balas_hammer(verbose=False)
            theta_bh = time.perf_counter() - t0

            t0 = time.perf_counter()
            pb_bh.marche_pied_resolution(verbose=False)
            t_bh = time.perf_counter() - t0

            # Stockage
            resultats.append({
                "N": n, "Run": k,
                "Theta_NO": theta_no, "Theta_BH": theta_bh,
                "T_NO": t_no, "T_BH": t_bh,
                "Total_NO": theta_no + t_no, "Total_BH": theta_bh + t_bh
            })

        print(f"\r    Progression : {nb_runs}/{nb_runs} runs - Terminé.")

    # Export CSV
    nom_fichier = "resultats_complexite.csv"
    try:
        with open(nom_fichier, mode='w', newline='') as f:
            fieldnames = ["N", "Run", "Theta_NO", "Theta_BH", "T_NO", "T_BH", "Total_NO", "Total_BH"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in resultats:
                writer.writerow(row)
        print(f"\n[SUCCÈS] Résultats sauvegardés dans '{nom_fichier}'.")
    except Exception as e:
        print(f"\n[ERREUR] Impossible d'écrire le fichier CSV : {e}")

    input("\nAppuyez sur [Entrée] pour revenir au menu principal...")


def resoudre_probleme_interactif(pb):
    """
    Gère le flux de résolution interactif pour un problème donné :
    Affichage -> Choix Algo Initial -> Optimisation Marche-Pied
    """
    pb.affichage()

    print("\n--- Choix de l'algorithme initial ---")
    print("1. Nord-Ouest (Simple, rapide, mais coût élevé)")
    print("2. Balas-Hammer (Complexe, plus lent, mais coût proche de l'optimal)")
    choix_algo = input("Votre choix (1 ou 2) : ").strip()

    t_start = time.perf_counter()
    if choix_algo == '2':
        pb.balas_hammer(verbose=True)
    else:
        pb.nord_ouest(verbose=True)
    t_end = time.perf_counter()

    print(f"\n[Temps Initialisation] {t_end - t_start:.6f} secondes")

    print("\n--- Proposition Initiale ---")
    pb.affichage()

    input("\nAppuyez sur [Entrée] pour lancer l'optimisation (Marche-Pied)...")

    t_start = time.perf_counter()
    pb.marche_pied_resolution(verbose=True)
    t_end = time.perf_counter()

    print(f"\n[Temps Optimisation] {t_end - t_start:.6f} secondes")
    print("\n--- FIN DU TRAITEMENT ---")
    input("Appuyez sur [Entrée] pour continuer...")