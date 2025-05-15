## Description du Projet

Ce projet a pour objectif d'implémenter et de comparer différentes méthodes numériques pour résoudre l'équation de la chaleur 1D :

$\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2}$

avec des conditions aux limites $u(0, t) = 0$ et $u(1, t) = 0$, et différentes conditions initiales $u(x, 0) = f(x)$. Nous avons exploré la solution analytique ainsi que trois schémas numériques aux différences finies : le schéma explicite , le schéma implicite  et le schéma de Crank-Nicolson.

## Méthodes Numériques Implémentées

1.  **Solution Analytique :** Utilisée comme référence pour évaluer la précision des méthodes numériques.
2.  **Schéma Explicite  :** Méthode du premier ordre en temps et du second ordre en espace.
3.  **Schéma Implicite  :** Méthode du premier ordre en temps et du second ordre en espace, inconditionnellement stable.
4.  **Schéma de Crank-Nicolson :** Méthode du second ordre en temps et en espace, inconditionnellement stable et généralement plus précise.

## Conditions Initiales Testées

Nous avons testé les méthodes numériques avec les deux conditions initiales suivantes :

1.  $f_1(x) = \sin(2\pi x)$
2.  $f_2(x) = \begin{cases}
  2x & \text{si } 0 \leq x \leq 0.5 \\
  2(1-x) & \text{si } 0.5 < x \leq 1
\end{cases}$ (fonction triangle)

## Visualisation des Résultats

Les résultats de chaque méthode numérique et pour chaque condition initiale sont visualisés à travers :

* **"Slices" temporelles :** Graphiques de la température $u(x, t)$ en fonction de la position $x$ à différents instants $t$, comparés à la solution analytique (lorsqu'elle existe).
* **Cartes de couleurs (Heatmaps) :** Visualisation de l'évolution de la température $u(x, t)$ sur le domaine spatio-temporel $[0, 1] \times [0, t_{final}]$.

## Execution
Pour exécuter chaque schéma numérique, naviguez vers le répertoire correspondant dans votre terminal et exécutez le script Python correspondant :
```bash
python schema_explicite.py
python schema_implicite.py
python schema_crank-nicolson.py