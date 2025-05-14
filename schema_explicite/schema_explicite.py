import numpy as np
import matplotlib.pyplot as plt

def solve_heat_explicit(f, nx, nt, t_final):
    """
    Résout l'équation de la chaleur 1D avec un schéma explicite.

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

    # Vérification de la condition de stabilité
    if r > 0.5:
        raise ValueError("Le schéma explicite est instable pour ce choix de dt et dx (r > 0.5).")

    # Initialisation de la solution
    u = np.zeros((nt, nx))
    u[0, :] = [f(xi) for xi in x]

    # Conditions aux limites (Dirichlet)
    u[:, 0] = 0
    u[:, -1] = 0

    # Boucle temporelle
    for j in range(0, nt - 1):
        for i in range(1, nx - 1):
            u[j + 1, i] = (1 - 2 * r) * u[j, i] + r * u[j, i + 1] + r * u[j, i - 1]

    return x, t, u

def analytical_solution_f1(x, t):
    """
    Solution analytique pour la condition initiale f1(x) = sin(2*pi*x).
    """
    return np.sin(2 * np.pi * x) * np.exp(-4 * np.pi**2 * t)

def analytical_solution_f2_truncated(x, t, num_terms=50):
    """
    Solution analytique tronquée pour la condition initiale f2(x).
    """
    u = np.zeros_like(x)
    for n_odd in range(1, 2 * num_terms, 2):
        term = (8 / (n_odd * np.pi)**2) * (-1)**((n_odd - 1) // 2) * np.sin(n_odd * np.pi * x) * np.exp(-(n_odd * np.pi)**2 * t)
        u += term
    return u

if __name__ == '__main__':
    # Paramètres de la simulation
    nx = 51
    nt = 501
    t_final = 0.1
    t_plot = [0, 0.02, 0.05, 0.1] # Instants de temps pour la comparaison
    num_terms_f2 = 50 # Nombre de termes pour la série de Fourier de f2

    # Condition initiale 1: f1(x) = sin(2*pi*x)
    def f1(x):
        return np.sin(2 * np.pi * x)

    # Résolution numérique pour f1
    x_num_f1, t_num_f1, u_num_f1 = solve_heat_explicit(f1, nx, nt, t_final)

    # Condition initiale 2: f2(x) (fonction triangle)
    def f2(x):
        if 0 <= x <= 0.5:
            return 2 * x
        elif 0.5 < x <= 1:
            return 2 * (1 - x)
        else:
            return 0

    # Résolution numérique pour f2
    x_num_f2, t_num_f2, u_num_f2 = solve_heat_explicit(f2, nx, nt, t_final)

    # --- Plot 1: Comparaison à différents instants (f1 et f2) ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1) # Premier sous-plot pour f1
    for time in t_plot:
        index_t_num = np.argmin(np.abs(t_num_f1 - time))
        plt.plot(x_num_f1, u_num_f1[index_t_num, :], label=f'Numérique (t={time:.2f})')
        u_analytical = analytical_solution_f1(x_num_f1, time)
        plt.plot(x_num_f1, u_analytical, '--', label=f'Analytique (t={time:.2f})')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title('Comparaison avec f1(x)')
    plt.legend(fontsize='small')
    plt.grid(True)

    plt.subplot(1, 2, 2) # Deuxième sous-plot pour f2
    for time in t_plot:
        index_t_num_f2 = np.argmin(np.abs(t_num_f2 - time))
        plt.plot(x_num_f2, u_num_f2[index_t_num_f2, :], label=f'Numérique (t={time:.2f})')
        u_analytical_f2 = analytical_solution_f2_truncated(x_num_f2, time, num_terms_f2)
        plt.plot(x_num_f2, u_analytical_f2, '--', label=f'Analytique tronquée (t={time:.2f})')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title('Comparaison avec f2(x)')
    plt.legend(fontsize='small')
    plt.grid(True)

    plt.tight_layout() # Ajuste l'espacement entre les sous-plots
    plt.savefig('comparison_slices_f1_f2.png')
    plt.close()

    # --- Plot 2: Cartes de couleurs (f1 et f2) ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1) # Premier sous-plot pour f1
    im_f1 = plt.imshow(u_num_f1, aspect='auto', extent=[0, 1, t_final, 0], cmap='viridis')
    plt.colorbar(im_f1, label='u(x, t)', shrink=0.7) # Réduit la taille de la colorbar
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Numérique avec f1(x)')

    plt.subplot(1, 2, 2) # Deuxième sous-plot pour f2
    im_f2 = plt.imshow(u_num_f2, aspect='auto', extent=[0, 1, t_final, 0], cmap='viridis')
    plt.colorbar(im_f2, label='u(x, t)', shrink=0.7) # Réduit la taille de la colorbar
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Numérique avec f2(x)')

    plt.tight_layout()
    plt.savefig('heatmaps_f1_f2.png')
    plt.close()

    # Affichage du rapport de discrétisation pour information
    dx = x_num_f1[1] - x_num_f1[0]
    dt = t_num_f1[1] - t_num_f1[0]
    r = dt / dx**2
    print(f"Rapport de discrétisation r = {r:.4f}")