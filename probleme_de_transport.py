import copy
import sys
import os

# Constante pour gérer la dégénérescence (valeur très petite considérée comme 0 mais basique)
EPSILON = 1e-9


class ProblemeDeTransport:
    """
    Classe représentant un problème de transport.
    Elle contient les données du problème (coûts, provisions, commandes)
    et les méthodes pour trouver une solution initiale et l'optimiser.
    """

    def __init__(self):
        self.n = 0
        self.m = 0
        self.couts = []
        self.provisions = []
        self.commandes = []
        self.proposition = []

        # Potentiels
        self.potentiels_u = []
        self.potentiels_v = []

        # Matrice des coûts marginaux (pour affichage)
        self.marginaux = []

    def lire_fichier(self, chemin_fichier):
        """
        Lit les données du problème à partir d'un fichier .txt.
        Format attendu (PDF):
        n
        m
        [Matrice n lignes : couts... provision]
        [Ligne commandes]
        """
        try:
            with open(chemin_fichier, 'r', encoding='utf-8') as f:
                content = f.read()
                tokens = content.split()

                if not tokens:
                    raise ValueError("Fichier vide.")

                iterator = iter(tokens)

                try:
                    self.n = int(next(iterator))
                    self.m = int(next(iterator))

                    self.couts = []
                    self.provisions = []
                    self.proposition = [[0.0 for _ in range(self.m)] for _ in range(self.n)]

                    # Lecture Matrice (Coûts + Prov)
                    for i in range(self.n):
                        row_costs = []
                        for _ in range(self.m):
                            row_costs.append(int(next(iterator)))
                        self.couts.append(row_costs)
                        self.provisions.append(int(next(iterator)))

                    # Lecture Commandes
                    self.commandes = []
                    for _ in range(self.m):
                        self.commandes.append(int(next(iterator)))

                    print(f"Données chargées : {self.n} fournisseurs, {self.m} clients.")

                except StopIteration:
                    print("Erreur : Le fichier s'arrête prématurément.")
                except ValueError as e:
                    print(f"Erreur de conversion : {e}")

        except Exception as e:
            print(f"Erreur lecture fichier : {e}")

    def affichage(self):
        """Affiche les tableaux de manière formatée et soignée"""
        if not self.couts:
            return

        print("\n" + "=" * 60)
        print("          DASHBOARD DU PROBLÈME DE TRANSPORT")
        print("=" * 60)

        # 1. Données Initiales
        print("\n--- 1. Données (Coûts | Provisions) ---")
        header = "      " + "".join([f" C{j + 1:<4}" for j in range(self.m)]) + " | PROV"
        print(header)
        print("-" * len(header))

        for i in range(self.n):
            row = f" P{i + 1:<3} |"
            for j in range(self.m):
                val = self.couts[i][j]
                row += f" {val:<4} "
            row += f"| {self.provisions[i]}"
            print(row)

        print("-" * len(header))
        cmd_row = " CMD  |"
        for c in self.commandes:
            cmd_row += f" {c:<4} "
        print(cmd_row)

        # 2. Proposition de Transport
        print("\n--- 2. Proposition de Transport (Actuelle) ---")
        self._afficher_matrice(self.proposition)
        print(f">>> Coût Total : {self.calcul_cout_total():.2f}")

        # 3. Potentiels
        if self.potentiels_u and self.potentiels_v:
            print("\n--- 3. Potentiels (Ui / Vj) ---")
            # Filtrer les None pour l'affichage
            u_str = "Ui : " + ", ".join(
                [f"P{i + 1}={val}" for i, val in enumerate(self.potentiels_u) if val is not None])
            v_str = "Vj : " + ", ".join(
                [f"C{j + 1}={val}" for j, val in enumerate(self.potentiels_v) if val is not None])
            print(u_str)
            print(v_str)

        # 4. Coûts Marginaux
        if self.marginaux:
            print("\n--- 4. Table des Coûts Marginaux (Delta) ---")
            self._afficher_matrice(self.marginaux)

    def _afficher_matrice(self, matrice):
        """Helper pour afficher une matrice n x m proprement"""
        header = "      " + "".join([f" C{j + 1:<6}" for j in range(self.m)])
        print(header)
        print("-" * len(header))
        for i in range(self.n):
            row_str = f" P{i + 1:<3} |"
            for j in range(self.m):
                val = matrice[i][j]
                if val is None:
                    txt = " . "
                elif val == 0 and not self._is_basic(i, j):
                    txt = " . "  # Case vide non basique
                elif val == EPSILON:
                    txt = " eps  "  # Affichage explicite epsilon
                else:
                    # Arrondi pour l'affichage
                    txt = f" {val:<6.4g}" if isinstance(val, float) else f" {val:<6}"
                row_str += f"{txt:<8}"
            print(row_str)

    def _is_basic(self, i, j):
        """Une case est basique si > 0 ou si c'est un epsilon ajouté"""
        return self.proposition[i][j] >= EPSILON

    def calcul_cout_total(self):
        total = 0
        for i in range(self.n):
            for j in range(self.m):
                # On ignore EPSILON pour le coût réel
                val = self.proposition[i][j]
                if val >= 1:  # Si c'est un vrai transport
                    total += val * self.couts[i][j]
        return total

    # --- ALGORITHMES INITIAUX ---

    def nord_ouest(self, verbose=True):
        if verbose: print("\n[Algorithme] Nord-Ouest")
        prov = list(self.provisions)
        cmd = list(self.commandes)
        self.proposition = [[0.0] * self.m for _ in range(self.n)]

        i, j = 0, 0
        while i < self.n and j < self.m:
            q = min(prov[i], cmd[j])
            self.proposition[i][j] = float(q)
            prov[i] -= q
            cmd[j] -= q

            if prov[i] == 0 and cmd[j] == 0:
                # Cas dégénéré simultané : on avance en diagonale
                i += 1
                j += 1
            elif prov[i] == 0:
                i += 1
            else:
                j += 1

    def balas_hammer(self, verbose=True):
        if verbose: print("\n[Algorithme] Balas-Hammer (VAM) - Strict")

        # Copies pour ne pas modifier les originaux
        prov = list(self.provisions)
        cmd = list(self.commandes)

        # Initialisation de la matrice résultat
        self.proposition = [[0.0] * self.m for _ in range(self.n)]

        # Ensembles des indices actifs
        lignes_restantes = set(range(self.n))
        cols_restantes = set(range(self.m))

        while lignes_restantes and cols_restantes:
            candidats = []

            # ---------------------------------------------------------
            # 1. ANALYSE DES LIGNES
            # ---------------------------------------------------------
            for i in lignes_restantes:
                # Récupérer les coûts (valeur, index_colonne) pour cette ligne
                couts_ligne = sorted([(self.couts[i][j], j) for j in cols_restantes], key=lambda x: x[0])

                # Calcul de la pénalité (Delta)
                if len(couts_ligne) >= 2:
                    pen = couts_ligne[1][0] - couts_ligne[0][0]
                elif len(couts_ligne) == 1:
                    pen = couts_ligne[0][0]
                else:
                    continue

                meilleur_cout = couts_ligne[0][0]
                idx_meilleure_col = couts_ligne[0][1]

                # Règle 2 : Capacité réelle de la case
                capa_case = min(prov[i], cmd[idx_meilleure_col])

                # (Pénalité, Capacité, -Coût, -IndexLigne, Type, IndexLigne, ListeCouts)
                candidats.append((pen, capa_case, -meilleur_cout, -i, 'ligne', i, couts_ligne))

            # ---------------------------------------------------------
            # 2. ANALYSE DES COLONNES
            # ---------------------------------------------------------
            for j in cols_restantes:
                # Récupérer les coûts (valeur, index_ligne) pour cette colonne
                couts_col = sorted([(self.couts[i][j], i) for i in lignes_restantes], key=lambda x: x[0])

                if len(couts_col) >= 2:
                    pen = couts_col[1][0] - couts_col[0][0]
                elif len(couts_col) == 1:
                    pen = couts_col[0][0]
                else:
                    continue

                meilleur_cout = couts_col[0][0]
                idx_meilleure_ligne = couts_col[0][1]

                capa_case = min(prov[idx_meilleure_ligne], cmd[j])

                # (Pénalité, Capacité, -Coût, -IndexCol, Type, IndexCol, ListeCouts)
                candidats.append((pen, capa_case, -meilleur_cout, -j, 'colonne', j, couts_col))

            # ---------------------------------------------------------
            # 3. CHOIX (Pénalité Max -> Capacité Max -> Coût Min -> Arbitraire G/H)
            # ---------------------------------------------------------
            if not candidats:
                break

            gagnant = max(candidats)
            pen_max, cap_max, neg_cout, _, type_choix, index_choisi, couts_tries = gagnant

            # ---------------------------------------------------------
            # 4. AFFECTATION
            # ---------------------------------------------------------
            if type_choix == 'ligne':
                r = index_choisi
                c = couts_tries[0][1]
            else:
                c = index_choisi
                r = couts_tries[0][1]

                # Quantité maximale permise
            q = min(prov[r], cmd[c])

            # Affichage demandé par le PDF (si verbose)
            if verbose:
                # On reconvertit le coût négatif en positif pour l'affichage
                print(
                    f"   -> Max Pénalité: {pen_max} (sur {type_choix} {index_choisi}) | Choix arête: ({r}, {c}) | Quantité: {q}")

            self.proposition[r][c] = float(q)

            # Mise à jour des stocks/demandes
            prov[r] -= q
            cmd[c] -= q

            # Nettoyage
            if prov[r] == 0:
                lignes_restantes.discard(r)
            if cmd[c] == 0:
                cols_restantes.discard(c)

    # --- MÉTHODE DU MARCHE-PIED & GRAPHES ---

    def est_connexe(self):
        """
        Vérifie la connexité du graphe des bases (bipartite).
        Retourne (bool, set_visited_nodes)
        Les noeuds sont indexés : 0..n-1 (Provs), n..n+m-1 (Commandes)
        """
        adj = {k: [] for k in range(self.n + self.m)}
        bases_count = 0

        # Construction du graphe uniquement sur les variables de base
        for i in range(self.n):
            for j in range(self.m):
                if self._is_basic(i, j):
                    u, v = i, self.n + j
                    adj[u].append(v)
                    adj[v].append(u)
                    bases_count += 1

        if bases_count == 0:
            return False, set()

        # BFS
        start_node = 0
        queue = [start_node]
        visited = {start_node}

        while queue:
            curr = queue.pop(0)
            for neigh in adj[curr]:
                if neigh not in visited:
                    visited.add(neigh)
                    queue.append(neigh)

        is_connected = len(visited) == (self.n + self.m)
        return is_connected, visited

    def rendre_connexe(self, verbose=True):
        """
        Modifie la proposition pour la rendre connexe.
        Si le graphe est déconnecté, on ajoute des arêtes artificielles (Epsilon).
        """
        is_conn, visited = self.est_connexe()

        # Affichage détaillé des sous-graphes si déconnecté (Demande PDF)
        if not is_conn and verbose:
            unvisited = set(range(self.n + self.m)) - visited
            print(f"   [Détail Connexité] Sous-graphe 1 (Visités): {visited}")
            print(f"   [Détail Connexité] Sous-graphe 2 (Isolés): {unvisited}")

        while not is_conn:
            if verbose: print("[Info] Graphe non connexe -> Correction dégénérescence...")

            min_cost = float('inf')
            best_cell = None

            unvisited = set(range(self.n + self.m)) - visited

            for u in visited:
                if u < self.n:  # u est une ligne
                    row = u
                    for col_node in unvisited:
                        if col_node >= self.n:
                            col = col_node - self.n
                            if self.couts[row][col] < min_cost:
                                min_cost = self.couts[row][col]
                                best_cell = (row, col)
                else:  # u est une colonne
                    col = u - self.n
                    for row_node in unvisited:
                        if row_node < self.n:
                            row = row_node
                            if self.couts[row][col] < min_cost:
                                min_cost = self.couts[row][col]
                                best_cell = (row, col)

            if best_cell:
                r, c = best_cell
                if verbose: print(f" -> Ajout lien artificiel (epsilon) en [{r}, {c}]")
                self.proposition[r][c] = EPSILON
                # Mise à jour rapide des visités
                is_conn, visited = self.est_connexe()
            else:
                if verbose: print("Erreur critique : Impossible de connecter le graphe.")
                break

    def calcul_potentiels(self):
        """Calcul des potentiels Ui et Vj tels que Ui + Vj = Cij"""
        self.potentiels_u = [None] * self.n
        self.potentiels_v = [None] * self.m

        self.potentiels_u[0] = 0

        changed = True
        while changed:
            changed = False
            for i in range(self.n):
                for j in range(self.m):
                    if self._is_basic(i, j):
                        if self.potentiels_u[i] is not None and self.potentiels_v[j] is None:
                            self.potentiels_v[j] = self.couts[i][j] - self.potentiels_u[i]
                            changed = True
                        elif self.potentiels_v[j] is not None and self.potentiels_u[i] is None:
                            self.potentiels_u[i] = self.couts[i][j] - self.potentiels_v[j]
                            changed = True

    def calcul_couts_marginaux(self):
        """Calcule Delta_ij = Cij - Ui - Vj pour les cases non basiques."""
        min_delta = 0
        cell_min = (-1, -1)

        self.marginaux = [[None] * self.m for _ in range(self.n)]

        for i in range(self.n):
            for j in range(self.m):
                if not self._is_basic(i, j):
                    if self.potentiels_u[i] is not None and self.potentiels_v[j] is not None:
                        delta = self.couts[i][j] - self.potentiels_u[i] - self.potentiels_v[j]
                        self.marginaux[i][j] = delta
                        if delta < min_delta:
                            min_delta = delta
                            cell_min = (i, j)
                    else:
                        self.marginaux[i][j] = 0

        return min_delta, cell_min

    def get_cycle_path(self, start_cell):
        """Trouve le cycle unique créé en ajoutant start_cell."""
        start_u = start_cell[0]
        start_v_node = self.n + start_cell[1]

        adj = {node: [] for node in range(self.n + self.m)}
        for i in range(self.n):
            for j in range(self.m):
                if self._is_basic(i, j):
                    u, v = i, self.n + j
                    adj[u].append(v)
                    adj[v].append(u)

        queue = [(start_u, [start_u])]
        visited = {start_u}
        path_found = None

        while queue:
            curr, path = queue.pop(0)
            if curr == start_v_node:
                path_found = path
                break
            for neigh in adj[curr]:
                if neigh not in visited:
                    visited.add(neigh)
                    queue.append((neigh, path + [neigh]))

        if not path_found:
            return None

        cycle_coords = []
        cycle_coords.append(start_cell)

        path_edges = []
        for k in range(len(path_found) - 1):
            n1 = path_found[k]
            n2 = path_found[k + 1]
            if n1 < self.n:
                r, c = n1, n2 - self.n
            else:
                r, c = n2, n1 - self.n
            path_edges.append((r, c))

        return cycle_coords + path_edges[::-1]

    def maximiser_transport_sur_cycle(self, cell_entree, verbose=True):
        cycle = self.get_cycle_path(cell_entree)
        if not cycle: return

        plus_cells = []
        minus_cells = []
        min_val = float('inf')

        if verbose: print(f" -> Cycle trouvé (longueur {len(cycle)})")

        for k, (r, c) in enumerate(cycle):
            if k % 2 == 0:
                plus_cells.append((r, c))
            else:
                minus_cells.append((r, c))
                val = self.proposition[r][c]
                if val < min_val:
                    min_val = val

        if min_val == float('inf'): min_val = 0

        if verbose: print(f" -> Quantité déplacée theta = {min_val:.4g}")

        # Mise à jour
        variable_sortante_trouvee = False

        for r, c in plus_cells:
            self.proposition[r][c] += min_val

        for r, c in minus_cells:
            self.proposition[r][c] -= min_val

            # Gestion sortie de base stricte (sécurité numérique)
            if abs(self.proposition[r][c]) < 1e-12:
                # Si c'est la première variable qui tombe à 0, elle sort vraiment de la base
                if not variable_sortante_trouvee:
                    self.proposition[r][c] = 0.0
                    variable_sortante_trouvee = True
                else:
                    # Si une DEUXIÈME variable tombe à 0 en même temps (dégénérescence simultanée)
                    # On la force à rester dans la base avec EPSILON pour ne pas briser la chaîne
                    self.proposition[r][c] = EPSILON
                    if verbose: print(f"   [Info] Maintien artificiel de ({r},{c}) avec Epsilon (Dégénérescence)")

    def marche_pied_resolution(self, verbose=True):
        iteration = 0
        max_iter = 500  # Augmenté pour les grands problèmes de complexité

        while iteration < max_iter:
            iteration += 1
            if verbose: print(f"\n################ ITÉRATION {iteration} ################")

            # 1. Vérification / Correction Connexité
            if not self.est_connexe()[0]:
                self.rendre_connexe(verbose=verbose)

            # 2. Calculs Potentiels & Marginaux
            self.calcul_potentiels()
            min_delta, cell_in = self.calcul_couts_marginaux()

            if verbose: self.affichage()

            if min_delta >= -1e-9:
                if verbose: print("\n>>> CRITÈRE D'OPTIMALITÉ ATTEINT : Solution Optimale trouvée.")
                break

            if verbose: print(f"\n[Amélioration] Candidat entrée : {cell_in} avec gain marginal {min_delta}")

            # 3. Modification sur cycle
            self.maximiser_transport_sur_cycle(cell_in, verbose=verbose)

        if iteration == max_iter:
            if verbose: print("\n[Attention] Nombre max d'itérations atteint.")

        if verbose:
            print("\n--- RÉSULTAT FINAL ---")
            self.affichage()