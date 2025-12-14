import os
import sys

# Import de la classe métier (pour la lecture de fichier dans le main)
try:
    from probleme_de_transport import ProblemeDeTransport
except ImportError:
    print("ERREUR CRITIQUE : Le fichier 'ProblemeDeTransport.py' est introuvable.")
    sys.exit(1)

# Import des fonctions utilitaires
try:
    from transport_utils import generer_probleme_transport_obj, lancer_etude_complexite, resoudre_probleme_interactif
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
        print("Q. Quitter")
        print("-" * 60)

        choix = input("Votre choix : ").strip().lower()

        if choix == 'q':
            print("Au revoir !")
            break

        elif choix == '1':
            nom_f = input("Entrez le numéro ou le nom du fichier (ex: '1' pour transport1.txt) : ")

            # Gestion raccourci : si l'utilisateur tape juste "1", on complète
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
                # On passe la main à l'utilitaire interactif
                resoudre_probleme_interactif(pb)
            except Exception as e:
                print(f"[ERREUR] Problème lors de la lecture du fichier : {e}")

        elif choix == '2':
            try:
                n_input = input("Saisir la taille N du problème (N fournisseurs, N clients) : ")
                n = int(n_input)
                if n <= 1:
                    raise ValueError

                print(f"\nGénération d'un problème {n}x{n}...")
                pb = generer_probleme_transport_obj(n, n)
                resoudre_probleme_interactif(pb)

            except ValueError:
                print("[ERREUR] Veuillez entrer un entier valide > 1.")

        elif choix == '3':
            lancer_etude_complexite()

        else:
            print("[ERREUR] Choix non reconnu.")