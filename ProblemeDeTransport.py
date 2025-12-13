import copy
import sys
import os
import random
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

    def generer_aleatoire(self, n):
        """Génère un problème n x n équilibré (voir réponse précédente)"""
        self.n = n
        self.m = n
        self.couts = [[random.randint(1, 100) for _ in range(n)] for _ in range(n)]
        
        # Matrice temp pour l'équilibre
        temp = [[random.randint(1, 100) for _ in range(n)] for _ in range(n)]
        self.provisions = [sum(row) for row in temp]
        self.commandes = [sum(temp[i][j] for i in range(n)) for j in range(n)]
        
        self.reset_solution()
        print(f"Problème {n}x{n} généré.")

    def reset_solution(self):
        """Remet à zéro la proposition et les calculs intermédiaires"""
        self.proposition = [[0.0 for _ in range(self.m)] for _ in range(self.n)]
        self.potentiels_u = []
        self.potentiels_v = []
        self.marginaux = []

    def affichage(self,verbose=True):
        """Affiche les tableaux de manière formatée et soignée"""
        if not verbose:
            return
        
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
            u_str = "Ui : " + ", ".join([f"P{i + 1}={val}" for i, val in enumerate(self.potentiels_u)])
            v_str = "Vj : " + ", ".join([f"C{j + 1}={val}" for j, val in enumerate(self.potentiels_v)])
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
        # Note : Dans cette implémentation, on stocke epsilon directement dans self.proposition
        return self.proposition[i][j] >= EPSILON

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
                # Cela crée un graphe non connexe (il manquera une variable de base)
                # On laisse 'rendre_connexe' régler cela plus tard.
                i += 1
                j += 1
            elif prov[i] == 0: #si c'est les provisions qui ont été épuisées on passe à la ligne suivante
                i += 1
            else: #sinon c'est que les commandes qui ont été épuisées on passe à la colonne suivante
                j += 1

    def balas_hammer(self, verbose=True):
        if verbose: print("\n[Algorithme] Balas-Hammer")
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
                    pen = couts_ligne[0][0]  # Ou une autre logique si une seule case reste
                else:
                    continue

                # Données pour les égalités
                meilleur_cout = couts_ligne[0][0]
                idx_meilleure_col = couts_ligne[0][1]

                # Règle 2 : Capacité de la case minimale = min(Offre dispo, Demande dispo)
                capa_case = min(prov[i], cmd[idx_meilleure_col])

                # On ajoute au tableau des candidats.
                # Structure du tuple pour le tri :
                # (Pénalité, Capacité, -Coût, -Index, Type, Index_Réel, Liste_Couts)
                # Note : On utilise des valeurs négatives pour Coût et Index car max() privilégie les grands nombres,
                # or nous voulons le plus PETIT coût et le plus PETIT index.
                candidats.append((pen, capa_case, -meilleur_cout, -i, 'ligne', i, couts_ligne))

            # ---------------------------------------------------------
            # 1. ANALYSE DES COLONNES
            # ---------------------------------------------------------
            for j in cols_restantes:
                # Récupérer les coûts (valeur, index_ligne) pour cette colonne
                couts_col = sorted([(self.couts[i][j], i) for i in lignes_restantes], key=lambda x: x[0])

                # Calcul de la pénalité (Delta)
                if len(couts_col) >= 2:
                    pen = couts_col[1][0] - couts_col[0][0]
                elif len(couts_col) == 1:
                    pen = couts_col[0][0]
                else:
                    continue

                # Données pour les égalités
                meilleur_cout = couts_col[0][0]
                idx_meilleure_ligne = couts_col[0][1]

                # Règle 2 : Capacité de la case minimale
                capa_case = min(prov[idx_meilleure_ligne], cmd[j])

                # Ajout candidat (Même structure que pour les lignes)
                candidats.append((pen, capa_case, -meilleur_cout, -j, 'colonne', j, couts_col))

            # ---------------------------------------------------------
            # 2. CHOIX (Pénalité Max -> Capacité Max -> Coût Min -> Arbitraire)
            # ---------------------------------------------------------
            if not candidats:
                break

            # La fonction max compare les tuples élément par élément :
            # 1. pen (le plus grand gagne)
            # 2. capa_case (le plus grand gagne)
            # 3. -meilleur_cout (le plus grand gagne, donc le plus petit coût réel gagne)
            # 4. -index (le plus grand gagne, donc l'index le plus proche de 0 gagne -> arbitraire gauche/haut)
            gagnant = max(candidats)

            _, _, _, _, type_choix, index_choisi, couts_tries = gagnant

            # ---------------------------------------------------------
            # 3. AFFECTATION
            # ---------------------------------------------------------
            if type_choix == 'ligne':
                r = index_choisi
                c = couts_tries[0][1]  # La colonne du coût minimal
            else:
                c = index_choisi
                r = couts_tries[0][1]  # La ligne du coût minimal

            # Quantité maximale permise
            q = min(prov[r], cmd[c])
            self.proposition[r][c] = float(q)

            # Mise à jour des stocks/demandes
            prov[r] -= q
            cmd[c] -= q

            # ---------------------------------------------------------
            # 4. NETTOYAGE DES LIGNES/COLONNES SATURÉES
            # ---------------------------------------------------------
            if prov[r] == 0:
                lignes_restantes.discard(r)
            if cmd[c] == 0:
                cols_restantes.discard(c)

    def calcul_cout_total(self):
        total = 0
        for i in range(self.n):
            for j in range(self.m):
                # On ignore EPSILON pour le coût réel
                val = self.proposition[i][j]
                if val >= 1:  # Si c'est un vrai transport
                    total += val * self.couts[i][j]
        return total

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
        start_node = 0  # On commence arbitrairement par le fournisseur 0
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

    def est_acyclique(self):
        """
        Vérifie si le graphe des bases contient un cycle.
        PDF : "Test pour savoir si la proposition est acyclique... parcours en largeur"
        """
        adj = {k: [] for k in range(self.n + self.m)}
        for i in range(self.n):
            for j in range(self.m):
                if self._is_basic(i, j):
                    u, v = i, self.n + j
                    adj[u].append(v)
                    adj[v].append(u)

        visited = set()
        parent = {}

        for node in range(self.n + self.m):
            if node not in visited:
                # Lancement BFS/DFS sur la composante
                queue = [node]
                visited.add(node)
                parent[node] = -1

                while queue:
                    u = queue.pop(0)
                    for v in adj[u]:
                        if v == parent[u]:
                            continue
                        if v in visited:
                            return False  # Cycle détecté
                        visited.add(v)
                        parent[v] = u
                        queue.append(v)
        return True

    def rendre_connexe(self):
        """
        Modifie la proposition pour la rendre connexe (Solution non dégénérée).
        Si le graphe est déconnecté, on ajoute des arêtes artificielles (Epsilon).
        """
        is_conn, visited = self.est_connexe()

        while not is_conn:
            print("[Info] Graphe non connexe -> Correction dégénérescence...")

            # On cherche une arête (i, j) reliant un noeud visité à un non-visité
            # avec le coût minimal pour perturber le moins possible l'optimisation.
            min_cost = float('inf')
            best_cell = None

            unvisited = set(range(self.n + self.m)) - visited

            # On cherche lien entre une Ligne Visité et une Colonne Non-Visitée
            # Ou une Ligne Non-Visitée et une Colonne Visitée

            for u in visited:
                if u < self.n:  # u est une ligne
                    # Chercher col j non visitée (col index = j + n)
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
                print(f" -> Ajout lien artificiel (epsilon) en [{r}, {c}]")
                self.proposition[r][c] = EPSILON
                # Mise à jour rapide des visités
                is_conn, visited = self.est_connexe()
            else:
                print("Erreur critique : Impossible de connecter le graphe.")
                break

    def calcul_potentiels(self):
        """
        Calcul des potentiels Ui et Vj tels que Ui + Vj = Cij pour les cases de base.
        """
        self.potentiels_u = [None] * self.n
        self.potentiels_v = [None] * self.m

        # On fixe arbitrairement U0 = 0
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
        """
        Calcule Delta_ij = Cij - Ui - Vj pour les cases non basiques.
        Retourne le meilleur gain (le plus négatif) et la cellule associée.
        """
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
                        self.marginaux[i][j] = 0  # Erreur potentiel

        return min_delta, cell_min

    def get_cycle_path(self, start_cell):
        """
        Trouve le cycle unique créé en ajoutant start_cell (u, v) au graphe des bases.
        Retourne une liste ordonnée de coordonnées [(r, c), ...] représentant le cycle.
        """
        start_u = start_cell[0]
        start_v_node = self.n + start_cell[1]  # Indexé n..n+m-1

        # Graphe des bases
        adj = {node: [] for node in range(self.n + self.m)}
        for i in range(self.n):
            for j in range(self.m):
                if self._is_basic(i, j):
                    u, v = i, self.n + j
                    adj[u].append(v)
                    adj[v].append(u)

        # BFS pour trouver chemin Tree entre start_u et start_v_node
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

        # Reconstruction du cycle : Arête ajoutée + Chemin inverse
        cycle_coords = []
        # 1. L'arête entrante (celle qu'on ajoute)
        cycle_coords.append(start_cell)

        # 2. Les arêtes du chemin existant
        # Le chemin BFS est [u_start, node1, node2, ..., v_target]
        # On veut fermer la boucle.
        # Le cycle est : (u_start, v_target) -> (v_target, node_k) -> ... -> (node_1, u_start)
        # Donc on parcourt path_found à l'envers ou on le construit et on voit la parité.

        # Parcourons les arêtes du chemin BFS
        # path_found : [U_source, ... , V_target]
        path_edges = []
        for k in range(len(path_found) - 1):
            n1 = path_found[k]
            n2 = path_found[k + 1]
            # Convertir noeuds en (row, col)
            if n1 < self.n:
                r, c = n1, n2 - self.n
            else:
                r, c = n2, n1 - self.n
            path_edges.append((r, c))

        # L'ordre du cycle doit être suivi pour l'alternance +/-
        # Cycle starts at start_cell (Entrance, +).
        # Next edge in cycle must share a column with start_cell.
        # start_cell = (r_start, c_start).
        # path_found[-1] is the column node corresponding to c_start.
        # So we should traverse path_edges in REVERSE order.

        return cycle_coords + path_edges[::-1]

    def maximiser_transport_sur_cycle(self, cell_entree,verbose=True):
        cycle = self.get_cycle_path(cell_entree)
        if not cycle: return

        # cycle[0] est la variable entrante (+)
        # cycle[1] est la variable sortante (-)
        # etc.

        plus_cells = []
        minus_cells = []

        min_val = float('inf')

        if verbose:print(f" -> Cycle trouvé (longueur {len(cycle)})")

        for k, (r, c) in enumerate(cycle):
            if k % 2 == 0:
                plus_cells.append((r, c))
            else:
                minus_cells.append((r, c))
                val = self.proposition[r][c]
                if val < min_val:
                    min_val = val

        if min_val == float('inf'): min_val = 0

        if verbose:print(f" -> Quantité déplacée theta = {min_val:.4g}")

        # Mise à jour
        for r, c in plus_cells:
            self.proposition[r][c] += min_val
        for r, c in minus_cells:
            self.proposition[r][c] -= min_val
            # Gestion sortie de base (on en enlève une seule si égalité)
            # Si val devient 0, elle devient non-basique (sauf si dégénérescence requise)
            # Pour simplifier, on laisse 0.0, mais marche-pied traitera < EPSILON comme non-basic
            # SAUF qu'il faut maintenir la connexité.
            # L'usage de EPSILON gère ça : si on a soustrait epsilon, ça devient 0 strict.
            if abs(self.proposition[r][c]) < 1e-12:
                self.proposition[r][c] = 0.0

    def marche_pied_resolution(self,verbose=True):
        iteration = 0
        max_iter = 500

        while iteration < max_iter:
            iteration += 1
            if verbose:print(f"\n################ ITÉRATION {iteration} ################")

            # 1. Vérification / Correction Connexité
            if not self.est_connexe()[0]:
                self.rendre_connexe()

            # 2. Calculs Potentiels & Marginaux
            self.calcul_potentiels()
            min_delta, cell_in = self.calcul_couts_marginaux()

            self.affichage(verbose=verbose)

            if min_delta >= -1e-9:  # Optimale (tous delta >= 0)
                if verbose:print("\n>>> CRITÈRE D'OPTIMALITÉ ATTEINT : Solution Optimale trouvée.")
                break

            if verbose:print(f"\n[Amélioration] Candidat entrée : {cell_in} avec gain marginal {min_delta}")

            # 3. Modification sur cycle
            self.maximiser_transport_sur_cycle(cell_in)

        if iteration == max_iter:
            if verbose:print("\n[Attention] Nombre max d'itérations atteint.")

        if verbose:print("\n--- RÉSULTAT FINAL ---")
        self.affichage(verbose=True)