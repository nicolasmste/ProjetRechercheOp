import sys
import time
import random
import csv
import contextlib
import glob  # Ajout pour scanner les fichiers CSV
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
    Lance le benchmark sur différentes tailles de problèmes.
    Génère un fichier CSV distinct pour chaque taille N (10, 40, 100).
    """
    print("\n" + "!" * 60)
    print("       LANCEMENT DE L'ÉTUDE DE COMPLEXITÉ (BENCHMARK)")
    print("!" * 60)

    # Tailles définies dans le PDF
    tailles_n = [10, 40, 100, 400]
    nb_runs = 100

    print(f"[INFO] Tailles à tester : {tailles_n}")
    print(f"[INFO] Runs par taille  : {nb_runs}")
    print("-" * 60)

    for n in tailles_n:
        print(f"\n>>> Traitement taille N = {n} ({n}x{n} = {n * n} variables)...")

        resultats_n = []  # Liste pour stocker uniquement les résultats de cette taille

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
            resultats_n.append({
                "N": n, "Run": k + 1,
                "Theta_NO": theta_no, "Theta_BH": theta_bh,
                "T_NO": t_no, "T_BH": t_bh,
                "Total_NO": theta_no + t_no, "Total_BH": theta_bh + t_bh
            })

        print(f"\r    Progression : {nb_runs}/{nb_runs} runs - Terminé.")

        # Export CSV spécifique pour ce N
        nom_fichier = f"resultats_complexite_n{n}.csv"
        try:
            with open(nom_fichier, mode='w', newline='') as f:
                fieldnames = ["N", "Run", "Theta_NO", "Theta_BH", "T_NO", "T_BH", "Total_NO", "Total_BH"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in resultats_n:
                    writer.writerow(row)
            print(f"[SUCCÈS] Résultats pour N={n} sauvegardés dans '{nom_fichier}'.")
        except Exception as e:
            print(f"[ERREUR] Impossible d'écrire le fichier CSV pour N={n} : {e}")

    input("\nAppuyez sur [Entrée] pour revenir au menu principal...")


def generer_graphiques_complexite():
    """
    Lit les fichiers CSV générés par l'étude de complexité et génère les graphiques
    (Nuages de points + Enveloppe pire cas) avec Matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[ERREUR] Le module 'matplotlib' n'est pas installé.")
        print("Veuillez l'installer via : pip install matplotlib")
        return

    print("\n" + "=" * 60)
    print("       GÉNÉRATION DES GRAPHIQUES DE COMPLEXITÉ")
    print("=" * 60)

    # Recherche des fichiers CSV correspondant au pattern
    fichiers_csv = glob.glob("resultats_complexite_n*.csv")

    if not fichiers_csv:
        print("[ERREUR] Aucun fichier 'resultats_complexite_n*.csv' trouvé.")
        print("Veuillez d'abord lancer l'option 3 (Étude de complexité).")
        return

    print(f"[INFO] {len(fichiers_csv)} fichiers de résultats trouvés.")

    # Chargement et agrégation des données
    data_by_n = {}  # Format: { n: [ {row_dict}, ... ] }

    for fichier in fichiers_csv:
        try:
            with open(fichier, mode='r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows: continue

                # Récupération du N depuis la première ligne
                n = int(rows[0]["N"])

                # Conversion des valeurs numériques
                clean_rows = []
                for row in rows:
                    clean_item = {}
                    for k, v in row.items():
                        if k == "N" or k == "Run":
                            clean_item[k] = int(v)
                        else:
                            clean_item[k] = float(v)
                    clean_rows.append(clean_item)

                if n not in data_by_n:
                    data_by_n[n] = []
                data_by_n[n].extend(clean_rows)

        except Exception as e:
            print(f"[ATTENTION] Erreur lecture {fichier} : {e}")

    if not data_by_n:
        print("[ERREUR] Aucune donnée valide extraite.")
        return

    sorted_n = sorted(data_by_n.keys())
    print(f"[INFO] Tailles N traitées : {sorted_n}")

    # Définition des métriques à tracer (Correspondant au PDF)
    metriques = [
        ("Theta_NO", "Initialisation Nord-Ouest (Theta_NO)"),
        ("Theta_BH", "Initialisation Balas-Hammer (Theta_BH)"),
        ("T_NO", "Optimisation Marche-Pied post-NO (T_NO)"),
        ("T_BH", "Optimisation Marche-Pied post-BH (T_BH)"),
        ("Total_NO", "Temps Total Nord-Ouest (Init + MP)"),
        ("Total_BH", "Temps Total Balas-Hammer (Init + MP)"),
    ]

    # Création des graphiques individuels
    for key, title in metriques:
        plt.figure(figsize=(10, 6))

        pire_cas_x = []
        pire_cas_y = []

        # Pour chaque N, on trace le nuage de points
        for n in sorted_n:
            valeurs = [d[key] for d in data_by_n[n]]
            x_vals = [n] * len(valeurs)

            # Nuage de points (Scatter)
            plt.scatter(x_vals, valeurs, alpha=0.5, s=20, c='blue', label='Exécution' if n == sorted_n[0] else "")

            # Identification du pire cas (Max)
            if valeurs:
                max_val = max(valeurs)
                pire_cas_x.append(n)
                pire_cas_y.append(max_val)
                plt.scatter([n], [max_val], c='red', s=50, marker='x', zorder=10,
                            label='Pire cas (Max)' if n == sorted_n[0] else "")

        # Tracé de l'enveloppe du pire cas
        plt.plot(pire_cas_x, pire_cas_y, 'r--', label='Enveloppe Pire Cas')

        plt.title(f"Complexité : {title}")
        plt.xlabel("Taille du problème (N)")
        plt.ylabel("Temps (secondes)")
        plt.legend()
        plt.grid(True, which="both", linestyle='--', alpha=0.7)

        # Échelles logarithmiques (Standard pour analyse de complexité)
        plt.xscale('log')
        plt.yscale('log')

        nom_img = f"graphique_{key}.png"
        plt.savefig(nom_img)
        plt.close()
        print(f"   -> Graphique généré : {nom_img}")

    # Graphique de comparaison finale (Total NO vs Total BH - Pires cas)
    plt.figure(figsize=(10, 6))

    max_total_no = []
    max_total_bh = []

    for n in sorted_n:
        vals_no = [d["Total_NO"] for d in data_by_n[n]]
        vals_bh = [d["Total_BH"] for d in data_by_n[n]]
        if vals_no: max_total_no.append(max(vals_no))
        if vals_bh: max_total_bh.append(max(vals_bh))

    plt.plot(sorted_n, max_total_no, 'b-o', label='Nord-Ouest + MP (Pire Cas)')
    plt.plot(sorted_n, max_total_bh, 'g-s', label='Balas-Hammer + MP (Pire Cas)')

    plt.title("Comparaison des performances globales (Pire Cas)")
    plt.xlabel("Taille du problème (N)")
    plt.ylabel("Temps Total (secondes)")
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')

    plt.savefig("graphique_comparaison_total.png")
    plt.close()
    print("   -> Graphique généré : graphique_comparaison_total.png")

    print("\n[SUCCÈS] Tous les graphiques ont été générés dans le dossier courant.")


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