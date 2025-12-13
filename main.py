import os
import sys
import time
# IMPORT IMPORTANT : On importe la classe depuis le fichier transport.py
# Assurez-vous que transport.py est dans le même dossier
try:
    from ProblemeDeTransport import ProblemeDeTransport
except ImportError:
    print("ERREUR CRITIQUE : Le fichier 'transport.py' est introuvable ou contient des erreurs.")
    print("Veuillez vérifier que 'transport.py' est présent dans le même dossier.")
    sys.exit(1)
    
def lancer_campagne_mesures():
    print("\n" + "="*60)
    print("      CAMPAGNE DE MESURES DE PERFORMANCE (Temps CPU)")
    print("="*60)
    
    try:
        n_input = input("Entrez la taille n du problème (ex: 50) : ")
        n = int(n_input)
    except ValueError:
        print("Erreur: Entier requis.")
        return

    pb = ProblemeDeTransport()
    
    # 1. Génération
    print(f"\nGénération du problème {n}x{n}...")
    pb.generer_aleatoire(n)
    
    print("\nDébut des mesures (affichage désactivé pour précision)...")

    # ---------------------------------------------------------
    # SCÉNARIO 1 : NORD-OUEST + MARCHE-PIED
    # ---------------------------------------------------------
    
    # Mesure Theta_NO (Initialization Nord-Ouest)
    t_start = time.process_time()
    pb.nord_ouest(verbose=False)
    t_end = time.process_time()
    theta_NO = t_end - t_start

    # Mesure t_NO (Optimisation Marche-Pied depuis NO)
    t_start = time.process_time()
    pb.marche_pied_resolution(verbose=False)
    t_end = time.process_time()
    t_NO = t_end - t_start
    
    cout_final_NO = pb.calcul_cout_total()

    # ---------------------------------------------------------
    # RESET (On garde les mêmes coûts/prov/cmd, on vide la solution)
    # ---------------------------------------------------------
    pb.reset_solution()

    # ---------------------------------------------------------
    # SCÉNARIO 2 : BALAS-HAMMER + MARCHE-PIED
    # ---------------------------------------------------------

    # Mesure Theta_BH (Initialization Balas-Hammer)
    t_start = time.process_time()
    pb.balas_hammer(verbose=False)
    t_end = time.process_time()
    theta_BH = t_end - t_start

    # Mesure t_BH (Optimisation Marche-Pied depuis BH)
    t_start = time.process_time()
    pb.marche_pied_resolution(verbose=False)
    t_end = time.process_time()
    t_BH = t_end - t_start
    
    cout_final_BH = pb.calcul_cout_total()

    # ---------------------------------------------------------
    # RÉSULTATS
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print(f" RÉSULTATS POUR TAILLE n = {n}")
    print("-"*60)
    print(f"{'Algorithme':<20} | {'Init (s)':<15} | {'Optim (s)':<15} | {'Total (s)':<15}")
    print("-"*60)
    print(f"{'Nord-Ouest':<20} | {theta_NO:<15.6f} | {t_NO:<15.6f} | {theta_NO + t_NO:<15.6f}")
    print(f"{'Balas-Hammer':<20} | {theta_BH:<15.6f} | {t_BH:<15.6f} | {theta_BH + t_BH:<15.6f}")
    print("-"*60)
    print(f"Coût final (Via NO) : {cout_final_NO:.2f}")
    print(f"Coût final (Via BH) : {cout_final_BH:.2f}")
    print("-"*60)
    
    # Vérification de cohérence (optionnelle mais rassurante)
    if abs(cout_final_NO - cout_final_BH) > 1e-5:
        print("/!\\ ATTENTION : Les optimums trouvés diffèrent (Minima locaux ?)")
    else:
        print("Les deux méthodes ont convergé vers le même coût (Probable Optimum Global).")
    
# ================= MAIN =================
if __name__ == "__main__":
    while True:
        print("\n=== MENU PRINCIPAL ===")
        print("1. Charger un fichier")
        print("2. Mode manuel (Générer + Voir étapes)")
        print("3. Mode Performance (Mesurer temps)")
        print("q. Quitter")
        
        choix = input("Votre choix : ")
        
        if choix == '3':
            lancer_campagne_mesures()
        elif choix == 'q':
            break
        
        pb = ProblemeDeTransport()

        # BRANCHE 1 : GÉNÉRATION ALÉATOIRE
        if choix.lower() == 'gen':
            try:
                n_val = int(input("Entrez la taille de la matrice (n) : "))
                if n_val <= 1:
                    print("La taille doit être > 1.")
                    continue
                pb.generer_aleatoire(n_val)
                pb.affichage()
            except ValueError:
                print("Erreur : Veuillez entrer un nombre entier valide.")
                continue

        # BRANCHE 2 : LECTURE FICHIER
        else:
            # Gestion du nom de fichier (ajoute transport et .txt si manquant pour faciliter la saisie)
            if not choix.startswith("transport"):
                f_name = f"transport{choix}.txt"
            else:
                f_name = choix
                if not f_name.endswith(".txt"):
                    f_name += ".txt"

            if not os.path.exists(f_name):
                print(f"[ERREUR] Le fichier '{f_name}' est introuvable.")
                continue

            print(f"\nLecture du fichier '{f_name}'...")
            pb.lire_fichier(f_name)
            pb.affichage()

        # --- SUITE DU PROGRAMME (Identique pour les deux cas) ---
        
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