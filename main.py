import os
import sys

# Import de la classe métier (pour la lecture de fichier dans le main)
try:
    from probleme_de_transport import ProblemeDeTransport
except ImportError:
    print("ERREUR CRITIQUE : Le fichier 'probleme_de_transport.py' est introuvable.")
    sys.exit(1)

# Import des fonctions utilitaires
try:
    from transport_utils import (
        generer_probleme_transport_obj,
        lancer_etude_complexite,
        resoudre_probleme_interactif,
        generer_trace_execution,
        generer_graphiques_complexite  # <--- NOUVEL IMPORT
    )
except ImportError:
    print("ERREUR CRITIQUE : Le fichier 'transport_utils.py' est introuvable.")
    sys.exit(1)

# ================= PROGRAMME PRINCIPAL =================

if __name__ == "__main__":
    while True:
        # Nettoyage console (optionnel, compatible Windows/Linux)
        # os.system('cls' if os.name == 'nt' else 'clear')

        print("\n" + "=" * 60)
        print("   PROJET DE TRANSPORT - RECHERCHE OPÉRATIONNELLE")
        print("=" * 60)
        print("1. Charger un fichier de transport (ex: transport1.txt)")
        print("2. Générer un problème aléatoire (Taille N x N)")
        print("3. Lancer l'étude de complexité (Benchmark)")
        print("4. Générer TOUTES les traces d'exécution (Automatique)")
        print("5. Générer les graphiques de complexité (à partir des CSV)")
        print("Q. Quitter")
        print("-" * 60)

        choix = input("Votre choix : ").strip().lower()

        if choix == 'q':
            print("Au revoir !")
            break

        elif choix == '1':
            nom_f = input("Entrez le numéro ou le nom du fichier (ex: '1' pour transport1.txt) : ")
            # Gestion raccourci
            if not nom_f.endswith(".txt"):
                fichier = f"problemes_de_transport/transport{nom_f}.txt"
            else:
                fichier = f"problemes_de_transport/{nom_f}"

            if not os.path.exists(fichier):
                print(f"[ERREUR] Le fichier '{fichier}' est introuvable.")
                input("Appuyez sur [Entrée]...")
                continue

            print(f"\nLecture du fichier '{fichier}'...")
            try:
                pb = ProblemeDeTransport()
                pb.lire_fichier(fichier)
                resoudre_probleme_interactif(pb)
            except Exception as e:
                print(f"[ERREUR] Problème lors de la lecture du fichier : {e}")

        elif choix == '2':
            try:
                n_input = input("Saisir la taille N du problème : ")
                n = int(n_input)
                if n <= 1: raise ValueError
                print(f"\nGénération d'un problème {n}x{n}...")
                pb = generer_probleme_transport_obj(n, n)
                resoudre_probleme_interactif(pb)
            except ValueError:
                print("[ERREUR] Veuillez entrer un entier valide > 1.")

        elif choix == '3':
            lancer_etude_complexite()

        elif choix == '4':
            print("\n--- GÉNÉRATION AUTOMATIQUE DES TRACES (.txt) ---")
            dossier_source = "problemes_de_transport"
            dossier_traces = "traces"

            if not os.path.exists(dossier_source):
                print(f"[ERREUR] Le dossier '{dossier_source}' n'existe pas.")
                input("Appuyez sur [Entrée] pour continuer...")
                continue

            # Création du dossier traces s'il n'existe pas
            if not os.path.exists(dossier_traces):
                print(f"[INFO] Création du dossier '{dossier_traces}'...")
                os.makedirs(dossier_traces)

            print(f"Recherche des fichiers 'transport*.txt' dans le dossier '{dossier_source}'...")

            # Recherche des fichiers
            fichiers_trouves = [f for f in os.listdir(dossier_source) if
                                f.startswith('transport') and f.endswith('.txt')]

            # Tri intelligent (pour avoir transport2 avant transport10)
            try:
                fichiers_trouves.sort(key=lambda x: int(x.replace('transport', '').replace('.txt', '')))
            except ValueError:
                fichiers_trouves.sort()  # Fallback tri alphabétique si noms non standards

            if not fichiers_trouves:
                print("[AVERTISSEMENT] Aucun fichier 'transport*.txt' trouvé.")

            count = 0
            for fichier in fichiers_trouves:
                print(f"\n[Traitement] {fichier}...")

                # Extraction du numéro pour le nommage (transport5.txt -> 5)
                base_name = fichier.replace('.txt', '')
                num = base_name.replace('transport', '')

                # Construction du chemin complet source
                full_path_in = os.path.join(dossier_source, fichier)

                try:
                    # 1. Version Nord-Ouest (NO)
                    nom_out_no = os.path.join(dossier_traces, f"trace{num}-no.txt")
                    print(f"   -> Génération de {nom_out_no}...")

                    pb_no = ProblemeDeTransport()
                    pb_no.lire_fichier(full_path_in)
                    generer_trace_execution(pb_no, '1', nom_out_no)

                    # 2. Version Balas-Hammer (BH)
                    nom_out_bh = os.path.join(dossier_traces, f"trace{num}-bh.txt")
                    print(f"   -> Génération de {nom_out_bh}...")

                    pb_bh = ProblemeDeTransport()
                    pb_bh.lire_fichier(full_path_in)
                    generer_trace_execution(pb_bh, '2', nom_out_bh)

                    count += 1
                except Exception as e:
                    print(f"   [ERREUR] Échec sur {fichier} : {e}")

            if count > 0:
                print(
                    f"\n[SUCCÈS] {count} problèmes traités. {count * 2} fichiers de trace générés dans le dossier '{dossier_traces}/'.")

            input("Appuyez sur [Entrée] pour continuer...")

        elif choix == '5':
            generer_graphiques_complexite()
            input("Appuyez sur [Entrée] pour continuer...")

        else:
            print("[ERREUR] Choix non reconnu.")