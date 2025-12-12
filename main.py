import os
import sys

# IMPORT IMPORTANT : On importe la classe depuis le fichier transport.py
# Assurez-vous que transport.py est dans le même dossier
try:
    from ProblemeDeTransport import ProblemeDeTransport
except ImportError:
    print("ERREUR CRITIQUE : Le fichier 'transport.py' est introuvable ou contient des erreurs.")
    print("Veuillez vérifier que 'transport.py' est présent dans le même dossier.")
    sys.exit(1)

# ================= MAIN =================
if __name__ == "__main__":
    while True:
        print("\n" + "=" * 50)
        print("   PROJET DE TRANSPORT - RECHERCHE OPÉRATIONNELLE")
        print("=" * 50)

        # Demande du fichier
        f = input("Entrez le nom du fichier (.txt) ou 'q' pour quitter : ")
        if f.lower() == 'q':
            print("Au revoir !")
            break


        f = f"transport{f}.txt"
        if not os.path.exists(f):
            print(f"[ERREUR] Le fichier '{f}' est introuvable.")
            continue

        # Instanciation et Lecture
        print(f"\nLecture du fichier '{f}'...")
        pb = ProblemeDeTransport()
        pb.lire_fichier(f)
        pb.affichage()

        # Choix de l'algorithme initial
        print("\n--- Choix de l'algorithme initial ---")
        print("1. Nord-Ouest")
        print("2. Balas-Hammer")
        c = input("Votre choix (1/2) : ")

        if c == '2':
            pb.balas_hammer()
        else:
            pb.nord_ouest()

        # Affichage résultat initial
        print("\n--- Proposition Initiale ---")
        pb.affichage()

        # Lancement de l'optimisation
        input("\nAppuyez sur [Entrée] pour lancer l'optimisation (Marche-Pied)...")
        pb.marche_pied_resolution()

        print("\n--- FIN DU TRAITEMENT ---")