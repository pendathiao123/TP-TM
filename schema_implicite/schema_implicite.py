import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

def solve_heat_implicit(f, nx, nt, t_final):
    """
    Résout l'équation de la chaleur 1D avec un schéma implicite (algorithme de Thomas).

    Args:
        f (function): Condition initiale u(x, 0) = f(x).
        nx (int): Nombre de points spatiaux.
        nt (int): Nombre de pas de temps.
        t_final (float): Temps final de la simulation.

    Returns:
        tuple: Un tuple contenant les mailles spatiales (x) et temporelles (t),
               et la solution u(x, t) sous forme de tableau numpy.
    """
    # Domaine spatial
    x = np.linspace(0, 1, nx)
    dx = x[1] - x[0]

    # Domaine temporel
    t = np.linspace(0, t_final, nt)
    dt = t[1] - t[0]

    # Rapport de discrétisation
    r = dt / dx**2

    # Initialisation de la solution
    u = np.zeros((nt, nx))
    u[0, :] = [f(xi) for xi in x]

    # Conditions aux limites (Dirichlet)
    u[:, 0] = 0
    u[:, -1] = 0

    # Construction de la matrice A (tridiagonale) sous forme banded pour solve_banded
    diagonal = np.ones(nx - 2) * (1 + 2 * r)
    upper_diagonal = np.concatenate([np.ones(nx - 3) * (-r), [0]])
    lower_diagonal = np.concatenate([[0], np.ones(nx - 3) * (-r)])
    A_banded = np.array([upper_diagonal, diagonal, lower_diagonal])

    # Boucle temporelle
    for j in range(0, nt - 1):
        b = u[j, 1:-1] # Le vecteur u^j (points intérieurs)
        u[j + 1, 1:-1] = solve_banded((1, 1), A_banded, b)

    return x, t, u

def analytical_solution_f1(x, t):
    """Solution analytique pour la condition initiale f1(x) = sin(2*pi*x)."""
    return np.exp(-(2 * np.pi)**2 * t) * np.sin(2 * np.pi * x)

def analytical_solution_f2_truncated(x, t, num_terms=50):
    """Solution analytique tronquée pour la condition initiale f2(x) (fonction triangle)."""
    u = np.zeros_like(x)
    for n in range(1, num_terms + 1):
        cn = 0
        if n % 2 != 0:
            cn = 8 / (np.pi**2 * n**2)
        u += cn * np.exp(-(n * np.pi)**2 * t) * np.sin(n * np.pi * x)
    return u

if __name__ == '__main__':
    # Paramètres de la simulation
    nx = 51
    nt = 501
    t_final = 0.1
    t_plot = [0, 0.02, 0.05, 0.1] # Instants de temps pour la comparaison
    num_terms_f2 = 50 # Nombre de termes pour la série de Fourier de f2
    colors = plt.cm.viridis(np.linspace(0, 1, len(t_plot))) # Palette de couleurs pour les slices

    # Condition initiale 1: f1(x) = sin(2*pi*x)
    def f1(x):
        return np.sin(2 * np.pi * x)

    # Résolution numérique pour f1
    x1_imp, t1_imp, u1_imp = solve_heat_implicit(f1, nx, nt, t_final)

    # Condition initiale 2: f2(x) (fonction triangle)
    def f2(x):
        if 0 <= x <= 0.5:
            return 2 * x
        elif 0.5 < x <= 1:
            return 2 * (1 - x)
        else:
            return 0

    # Résolution numérique pour f2
    x2_imp, t2_imp, u2_imp = solve_heat_implicit(f2, nx, nt, t_final)

    # --- Plot 1: Cartes de couleurs combinées (implicite) ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    im_f1_imp = plt.imshow(u1_imp, aspect='auto', extent=[0, 1, t_final, 0], cmap='plasma') # Changement de cmap
    plt.colorbar(im_f1_imp, label='u(x, t)', shrink=0.7)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Implicite avec f1(x)')

    plt.subplot(1, 2, 2)
    im_f2_imp = plt.imshow(u2_imp, aspect='auto', extent=[0, 1, t_final, 0], cmap='plasma') # Changement de cmap
    plt.colorbar(im_f2_imp, label='u(x, t)', shrink=0.7)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Implicite avec f2(x)')

    plt.tight_layout()
    plt.savefig('heatmaps_implicit_f1_f2_styled.png')
    plt.close()

    # --- Plot 2: Comparaison à différents instants (implicite avec analytique) ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1) # Premier sous-plot pour f1
    for i, time in enumerate(t_plot):
        index_t_num = np.argmin(np.abs(t1_imp - time))
        plt.plot(x1_imp, u1_imp[index_t_num, :], color=colors[i], label=f'Numérique (t={time:.2f})')
        plt.plot(x1_imp, analytical_solution_f1(x1_imp, time), '--', color=colors[i], alpha=0.7, label=f'Analytique (t={time:.2f})')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title('Comparaison implicite avec f1(x)')
    plt.legend(fontsize='small')
    plt.grid(True)

    plt.subplot(1, 2, 2) # Deuxième sous-plot pour f2
    for i, time in enumerate(t_plot):
        index_t_num_f2 = np.argmin(np.abs(t2_imp - time))
        plt.plot(x2_imp, u2_imp[index_t_num_f2, :], color=colors[i], label=f'Numérique (t={time:.2f})')
        plt.plot(x2_imp, analytical_solution_f2_truncated(x2_imp, time, num_terms_f2), '--', color=colors[i], alpha=0.7, label=f'Analytique tronquée (t={time:.2f})')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title('Comparaison implicite avec f2(x)')
    plt.legend(fontsize='small')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('comparison_slices_implicit_f1_f2_analytical_styled.png')
    plt.close()