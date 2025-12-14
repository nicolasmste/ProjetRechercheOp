import sys
import copy

# Constante pour gérer la dégénérescence et les comparaisons flottantes
EPSILON = 1e-9


class ProblemeDeTransport:
    """
    Classe représentant un problème de transport optimisée.
    Complexité améliorée :
    - Balas-Hammer : O(N^2 log N) via pré-tri
    - Opérations Graphe (Potentiels, Cycles) : O(N) via liste d'adjacence
    """

    def __init__(self):
        self.n = 0
        self.m = 0
        self.couts = []
        self.provisions = []
        self.commandes = []

        # Matrice des quantités transportées (valeurs réelles)
        self.proposition = []

        # OPTIMISATION CRITIQUE :
        # Ensemble des coordonnées (i, j) des variables de base.
        # Évite de parcourir toute la matrice pour construire le graphe.
        self.bases = set()

        # Potentiels
        self.potentiels_u = []
        self.potentiels_v = []

        # Matrice des coûts marginaux
        self.marginaux = []

    def lire_fichier(self, chemin_fichier):
        """Lit les données du problème à partir d'un fichier .txt."""
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
                    self.bases = set()  # Reset bases

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

    # --- GESTION DES VARIABLES DE BASE (OPTIMISATION) ---

    def _ajouter_base(self, i, j, val):
        """Définit une variable comme basique et met à jour la structure."""
        self.proposition[i][j] = float(val)
        self.bases.add((i, j))

    def _retirer_base(self, i, j):
        """Retire une variable de la base (la met à 0)."""
        self.proposition[i][j] = 0.0
        if (i, j) in self.bases:
            self.bases.remove((i, j))

    def _is_basic(self, i, j):
        """Vérifie si une case est dans la base via le set (O(1))."""
        return (i, j) in self.bases

    # --- AFFICHAGE ---

    def affichage(self):
        """Affiche les tableaux de manière formatée."""
        if not self.couts:
            return

        print("\n" + "=" * 60)
        print("          DASHBOARD DU PROBLÈME DE TRANSPORT")
        print("=" * 60)

        # 1. Données Initiales
        print("\n--- 1. Données (Coûts | Provisions) ---")
        header = "      " + "".join([f" {f'C{j + 1}':<7}" for j in range(self.m)]) + "| PROV"
        print(header)
        print("-" * len(header))

        for i in range(self.n):
            row = f" P{i + 1:<3} |"
            for j in range(self.m):
                row += f" {self.couts[i][j]:<6} "
            row += f"| {self.provisions[i]}"
            print(row)

        print("-" * len(header))
        cmd_row = " CMD  |"
        for c in self.commandes:
            cmd_row += f" {c:<6} "
        print(cmd_row)

        # 2. Proposition de Transport
        print("\n--- 2. Proposition de Transport (Actuelle) ---")
        self._afficher_matrice(self.proposition, show_basis=True)
        print(f">>> Coût Total : {self.calcul_cout_total():.2f}")

        # 3. Potentiels
        if self.potentiels_u and self.potentiels_v:
            print("\n--- 3. Potentiels (Ui / Vj) ---")
            u_str = "Ui : " + ", ".join(
                [f"P{i + 1}={val}" for i, val in enumerate(self.potentiels_u) if val is not None])
            v_str = "Vj : " + ", ".join(
                [f"C{j + 1}={val}" for j, val in enumerate(self.potentiels_v) if val is not None])
            print(u_str)
            print(v_str)

        # 4. Coûts Marginaux
        if self.marginaux:
            print("\n--- 4. Table des Coûts Marginaux (Delta) ---")
            self._afficher_matrice(self.marginaux, show_basis=False)

    def _afficher_matrice(self, matrice, show_basis=False):
        header = "      " + "".join([f" {f'C{j + 1}':<7}" for j in range(self.m)])
        print(header)
        print("-" * len(header))
        for i in range(self.n):
            row_str = f" P{i + 1:<3} |"
            for j in range(self.m):
                val = matrice[i][j]

                # Logique d'affichage spécifique
                if show_basis:
                    # Pour la proposition, on veut voir si c'est une base (même si 0 ou epsilon)
                    if (i, j) in self.bases:
                        if val == EPSILON or (val == 0 and (i, j) in self.bases):
                            txt = " eps  "
                        else:
                            txt = f" {val:<6.4g}"
                    else:
                        txt = " . "
                else:
                    # Pour les marginaux
                    if val is None:
                        txt = " . "
                    else:
                        txt = f" {val:<6.4g}"

                row_str += f"{txt:<8}"
            print(row_str)

    def calcul_cout_total(self):
        total = 0
        for (i, j) in self.bases:
            val = self.proposition[i][j]
            if val > EPSILON:  # On ignore epsilon pour le coût
                total += val * self.couts[i][j]
        return total

    def test_degenerescence(self, verbose=True):
        """Vérifie si nb_bases < N + M - 1."""
        nb_bases = len(self.bases)
        cible = self.n + self.m - 1
        est_degeneree = nb_bases < cible

        if verbose:
            print(f"   [Test Dégénérescence] Variables de base : {nb_bases} / {cible} attendues.")
            if est_degeneree:
                print("   /!\\ SOLUTION DÉGÉNÉRÉE DÉTECTÉE /!\\")
        return est_degeneree

    # --- ALGORITHMES INITIAUX ---

    def nord_ouest(self, verbose=True):
        if verbose: print("\n[Algorithme] Nord-Ouest")
        # Reset
        self.proposition = [[0.0] * self.m for _ in range(self.n)]
        self.bases = set()

        prov = list(self.provisions)
        cmd = list(self.commandes)

        i, j = 0, 0
        while i < self.n and j < self.m:
            q = min(prov[i], cmd[j])

            # Ajout à la base
            self._ajouter_base(i, j, q)

            prov[i] -= q
            cmd[j] -= q

            if prov[i] == 0 and cmd[j] == 0:
                # Dégénérescence simultanée : on avance en diagonale
                i += 1
                j += 1
                if i < self.n and j < self.m:
                    # On ajoute artificiellement une base à 0 (epsilon)
                    self._ajouter_base(i, j - 1, EPSILON)
            elif prov[i] == 0:
                i += 1
            else:
                j += 1

        # Sécurité pour atteindre N+M-1 bases si besoin
        while len(self.bases) < self.n + self.m - 1:
            if verbose: print("   [Info] Ajout base artificielle post-Nord-Ouest")
            self.rendre_connexe(verbose=False)

    def balas_hammer(self, verbose=True):
        """
        Version optimisée de Balas-Hammer.
        Trie les coûts une seule fois au début (O(N^2 log N)).
        """
        if verbose: print("\n[Algorithme] Balas-Hammer (VAM) - Optimisé")

        self.proposition = [[0.0] * self.m for _ in range(self.n)]
        self.bases = set()

        prov = list(self.provisions)
        cmd = list(self.commandes)

        # 1. Pré-traitement : Tri des lignes et colonnes
        sorted_rows = []
        for i in range(self.n):
            ligne = [(self.couts[i][j], j) for j in range(self.m)]
            ligne.sort(key=lambda x: x[0])
            sorted_rows.append(ligne)

        sorted_cols = []
        for j in range(self.m):
            colonne = [(self.couts[i][j], i) for i in range(self.n)]
            colonne.sort(key=lambda x: x[0])
            sorted_cols.append(colonne)

        lignes_restantes = set(range(self.n))
        cols_restantes = set(range(self.m))

        while lignes_restantes and cols_restantes:
            candidats = []

            # --- Analyse Lignes ---
            for i in lignes_restantes:
                min1, min2 = None, None
                valid_costs = []
                for cost, j in sorted_rows[i]:
                    if j in cols_restantes:
                        valid_costs.append((cost, j))
                        if len(valid_costs) == 2: break

                if len(valid_costs) >= 2:
                    pen = valid_costs[1][0] - valid_costs[0][0]
                    meilleur = valid_costs[0]
                elif len(valid_costs) == 1:
                    pen = valid_costs[0][0]
                    meilleur = valid_costs[0]
                else:
                    continue

                capa = min(prov[i], cmd[meilleur[1]])
                # Tuple: (Penalité, Capacité, -Coût, r, c, type)
                candidats.append((pen, capa, -meilleur[0], i, meilleur[1], 'ligne'))

            # --- Analyse Colonnes ---
            for j in cols_restantes:
                min1, min2 = None, None
                valid_costs = []
                for cost, i in sorted_cols[j]:
                    if i in lignes_restantes:
                        valid_costs.append((cost, i))
                        if len(valid_costs) == 2: break

                if len(valid_costs) >= 2:
                    pen = valid_costs[1][0] - valid_costs[0][0]
                    meilleur = valid_costs[0]
                elif len(valid_costs) == 1:
                    pen = valid_costs[0][0]
                    meilleur = valid_costs[0]
                else:
                    continue

                capa = min(prov[meilleur[1]], cmd[j])
                candidats.append((pen, capa, -meilleur[0], meilleur[1], j, 'col'))

            if not candidats: break

            gagnant = max(candidats)
            pen, cap, neg_cost, r, c, type_choix = gagnant

            q = min(prov[r], cmd[c])

            # --- AFFICHAGE FORMATÉ DEMANDÉ ---
            if verbose:
                loc_txt = ""
                if type_choix == 'col':
                    loc_txt = f"colonne {c}"
                else:
                    loc_txt = f"ligne {r}"
                print(f"   -> Max Pénalité: {pen} (sur {loc_txt}) | Choix arête: ({r}, {c}) | Quantité: {q}")

            # Affectation
            self._ajouter_base(r, c, q)
            prov[r] -= q
            cmd[c] -= q

            # Mise à jour des ensembles restants
            if prov[r] == 0 and cmd[c] == 0:
                if len(lignes_restantes) > 1:
                    lignes_restantes.discard(r)
                else:
                    cols_restantes.discard(c)
            elif prov[r] == 0:
                lignes_restantes.discard(r)
            elif cmd[c] == 0:
                cols_restantes.discard(c)

        # Finition : Bases artificielles si besoin
        while len(self.bases) < self.n + self.m - 1:
            self.rendre_connexe(verbose=False)

    # --- MÉTHODES OPTIMISÉES GRAPHES & MARCHE-PIED ---

    def _build_adj_list(self):
        adj = {node: [] for node in range(self.n + self.m)}
        for (i, j) in self.bases:
            u = i
            v = self.n + j
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def est_connexe(self):
        if not self.bases:
            return False, set()

        adj = self._build_adj_list()
        start_node = list(adj.keys())[0]

        visited = set()
        queue = [start_node]
        visited.add(start_node)

        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

        is_conn = len(visited) == (self.n + self.m)
        return is_conn, visited

    def rendre_connexe(self, verbose=True):
        """Ajoute des arêtes artificielles (Epsilon) pour connecter le graphe."""
        is_conn, visited = self.est_connexe()

        # --- AFFICHAGE FORMATÉ DEMANDÉ (SOUS-GRAPHES) ---
        if not is_conn and verbose:
            unvisited = sorted(list(set(range(self.n + self.m)) - visited))
            visited_list = sorted(list(visited))
            print(f"   [Détail Connexité] Sous-graphe 1 (Visités): {set(visited_list)}")
            print(f"   [Détail Connexité] Sous-graphe 2 (Isolés): {set(unvisited)}")
            print("[Info] Graphe non connexe -> Correction dégénérescence...")

        loop_guard = 0
        while not is_conn and loop_guard < (self.n * self.m):
            loop_guard += 1
            unvisited = set(range(self.n + self.m)) - visited

            best_link = None
            min_cost = float('inf')

            for u in visited:
                if u < self.n:  # Ligne
                    r = u
                    for v in unvisited:
                        if v >= self.n:
                            c = v - self.n
                            if self.couts[r][c] < min_cost:
                                min_cost = self.couts[r][c]
                                best_link = (r, c)
                else:  # Colonne
                    c = u - self.n
                    for v in unvisited:
                        if v < self.n:
                            r = v
                            if self.couts[r][c] < min_cost:
                                min_cost = self.couts[r][c]
                                best_link = (r, c)

            if best_link:
                r, c = best_link
                if verbose:
                    print(f" -> Ajout lien artificiel (epsilon) en [{r}, {c}]")
                self._ajouter_base(r, c, EPSILON)
                is_conn, visited = self.est_connexe()

                # Répéter l'affichage si encore déconnecté ?
                # Le format demandé semble impliquer un log par ajout
                if not is_conn and verbose:
                    print("[Info] Graphe non connexe -> Correction dégénérescence...")
            else:
                break

    def calcul_potentiels(self):
        adj = self._build_adj_list()
        self.potentiels_u = [None] * self.n
        self.potentiels_v = [None] * self.m

        start_node = 0
        self.potentiels_u[0] = 0

        queue = [start_node]
        visited = {start_node}

        while queue:
            curr = queue.pop(0)
            is_row = (curr < self.n)
            idx_curr = curr if is_row else (curr - self.n)
            val_curr = self.potentiels_u[idx_curr] if is_row else self.potentiels_v[idx_curr]

            if val_curr is None: continue

            for neighbor in adj[curr]:
                if neighbor in visited: continue

                visited.add(neighbor)
                queue.append(neighbor)

                if is_row:
                    j = neighbor - self.n
                    cost = self.couts[idx_curr][j]
                    self.potentiels_v[j] = cost - val_curr
                else:
                    i = neighbor
                    cost = self.couts[i][idx_curr]
                    self.potentiels_u[i] = cost - val_curr

    def calcul_couts_marginaux(self):
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
        start_u = start_cell[0]
        start_v_node = self.n + start_cell[1]
        adj = self._build_adj_list()

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

        if not path_found: return None

        cycle_coords = [start_cell]
        path_nodes = path_found
        for k in range(len(path_nodes) - 1):
            n1 = path_nodes[k]
            n2 = path_nodes[k + 1]
            if n1 < self.n:
                r, c = n1, n2 - self.n
            else:
                r, c = n2, n1 - self.n
            cycle_coords.append((r, c))

        return cycle_coords

    def maximiser_transport_sur_cycle(self, cell_entree, verbose=True):
        cycle = self.get_cycle_path(cell_entree)
        if not cycle: return

        if verbose: print(f" -> Cycle trouvé (longueur {len(cycle)})")

        plus_cells = []
        minus_cells = []

        for k, (r, c) in enumerate(cycle):
            if k % 2 == 0:
                plus_cells.append((r, c))
            else:
                minus_cells.append((r, c))

        valeurs_minus = [self.proposition[r][c] for r, c in minus_cells]
        theta = min(valeurs_minus) if valeurs_minus else 0
        if theta < EPSILON: theta = 0

        if verbose: print(f" -> Quantité déplacée theta = {theta}")

        # --- AFFICHAGE FORMATÉ DEMANDÉ (DÉTAILS TRANSFERT) ---
        if verbose:
            print("    Détails du transfert :")
            for k, (r, c) in enumerate(cycle):
                old_val = self.proposition[r][c]
                # Formatage propre de la valeur (0 pour epsilon ou entier)
                val_display = 0 if old_val < EPSILON else old_val
                if abs(val_display - round(val_display)) < 1e-9:
                    val_display = int(round(val_display))

                op = "+" if k % 2 == 0 else "-"
                # Conversion index 0-based -> 1-based (P1, C1...)
                print(f"    - Case P{r + 1}-C{c + 1} ({val_display}) : {op} {theta}")

        # Mise à jour
        val_entree = self.proposition[cell_entree[0]][cell_entree[1]] + theta
        self._ajouter_base(cell_entree[0], cell_entree[1], val_entree)

        for (r, c) in plus_cells[1:]:
            self.proposition[r][c] += theta

        variable_sortie_faite = False
        for (r, c) in minus_cells:
            self.proposition[r][c] -= theta

            if not variable_sortie_faite and self.proposition[r][c] <= EPSILON:
                if self.proposition[r][c] <= EPSILON:
                    self._retirer_base(r, c)
                    variable_sortie_faite = True
            elif self.proposition[r][c] < EPSILON:
                self.proposition[r][c] = EPSILON
                if verbose: print(f"   [Dégénérescence] ({r},{c}) maintenue à Epsilon")

    def marche_pied_resolution(self, verbose=True):
        iteration = 0
        max_iter = 100000

        while iteration < max_iter:
            iteration += 1
            if verbose: print(f"\n################ ITÉRATION {iteration} ################")

            # 1. Connexité (Si jamais perdue, mais robuste normalement)
            if len(self.bases) < self.n + self.m - 1:
                # Si c'est juste un manque de bases (dégénérescence), on appelle rendre_connexe
                # qui contient le test_degenerescence implicite pour l'affichage
                pass

                # Appel explicite pour l'affichage requis par le PDF
            self.test_degenerescence(verbose=verbose)

            # Correction effective
            if not self.est_connexe()[0]:
                self.rendre_connexe(verbose=verbose)
            elif len(self.bases) < self.n + self.m - 1:
                # Cas où c'est connexe mais dégénéré (manque une arête quelque part qui ne connecte rien de nouveau ?)
                # Rare en transport simple, mais on force l'ajout
                if verbose: print("[Info] Graphe connexe mais dégénéré -> Ajout epsilon.")
                self.rendre_connexe(verbose=verbose)

            # 2. Potentiels
            self.calcul_potentiels()

            # 3. Marginaux
            min_delta, cell_in = self.calcul_couts_marginaux()

            if verbose:
                self.affichage()

            if min_delta >= -1e-9:
                if verbose: print("\n>>> CRITÈRE D'OPTIMALITÉ ATTEINT : Solution Optimale trouvée.")
                break

            if verbose: print(f"\n[Amélioration] Candidat entrée : {cell_in} avec gain marginal {min_delta}")

            # 4. Cycle & Update
            self.maximiser_transport_sur_cycle(cell_in, verbose=verbose)

        if verbose:
            print("\n--- RÉSULTAT FINAL ---")
            self.affichage()