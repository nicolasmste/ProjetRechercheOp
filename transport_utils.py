import sys
import time
import random
import csv
import contextlib
from copy import deepcopy

# Import de la classe métier
try:
    from probleme_de_transport import ProblemeDeTransport
except ImportError:
    print("ERREUR CRITIQUE : Le fichier 'probleme_de_transport.py' est introuvable.")
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


def generer_trace_execution(pb, algo_choix, nom_fichier):
    """
    Exécute la résolution complète (Initialisation + Marche-Pied)
    et redirige TOUTE la sortie console (print) vers un fichier texte.

    Args:
        pb (ProblemeDeTransport): L'objet problème chargé.
        algo_choix (str): '1' pour Nord-Ouest, '2' pour Balas-Hammer.
        nom_fichier (str): Chemin du fichier de sortie (ex: NEW2-4-trace5-no.txt).
    """
    print(f"[INFO] Génération de la trace dans '{nom_fichier}' en cours...")

    try:
        # On ouvre le fichier en écriture
        with open(nom_fichier, 'w', encoding='utf-8') as f:
            # On redirige stdout (la console) vers ce fichier
            with contextlib.redirect_stdout(f):
                print("=" * 60)
                print(f"TRACE D'EXÉCUTION GÉNÉRÉE AUTOMATIQUEMENT")
                print(f"Fichier cible : {nom_fichier}")
                print(f"Algorithme Initial : {'Nord-Ouest' if algo_choix == '1' else 'Balas-Hammer'}")
                print("=" * 60 + "\n")

                # 1. Affichage Données
                pb.affichage()

                # 2. Algo Initial
                t_start = time.perf_counter()
                if algo_choix == '2':
                    pb.balas_hammer(verbose=True)
                else:
                    pb.nord_ouest(verbose=True)
                t_end = time.perf_counter()

                print(f"\n[Temps Initialisation] {t_end - t_start:.6f} secondes")
                print("\n--- Proposition Initiale ---")
                pb.affichage()

                # 3. Optimisation
                print("\n" + "=" * 40)
                print(" Lancement de l'optimisation (Marche-Pied)")
                print("=" * 40)

                t_start = time.perf_counter()
                pb.marche_pied_resolution(verbose=True)
                t_end = time.perf_counter()

                print(f"\n[Temps Optimisation] {t_end - t_start:.6f} secondes")
                print("\n--- FIN DE LA TRACE ---")

        print(f"[SUCCÈS] Trace sauvegardée dans '{nom_fichier}'.")

    except Exception as e:
        print(f"[ERREUR] Échec de la génération de trace : {e}")


def lancer_etude_complexite():
    """
    Lance le benchmark sur différentes tailles de problèmes et enregistre les résultats dans un CSV.
    """
    print("\n" + "!" * 60)
    print("       LANCEMENT DE L'ÉTUDE DE COMPLEXITÉ (BENCHMARK)")
    print("!" * 60)

    # Tailles définies dans le PDF
    tailles_n = [10, 40, 100]
    nb_runs = 100

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
    print("1. Nord-Ouest")
    print("2. Balas-Hammer")
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