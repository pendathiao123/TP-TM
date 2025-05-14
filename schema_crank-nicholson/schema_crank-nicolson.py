import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

def solve_heat_crank_nicolson(f, nx, nt, t_final):
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

    # Construction des matrices A et B (tridiagonales) sous forme banded
    diagonal_A = np.ones(nx - 2) * (2 + 2 * r)
    upper_diagonal_A = np.ones(nx - 3) * (-r)
    lower_diagonal_A = np.ones(nx - 3) * (-r)
    A_banded = np.array([np.append(upper_diagonal_A, 0), diagonal_A, np.append(0, lower_diagonal_A)])

    diagonal_B = np.ones(nx - 2) * (2 - 2 * r)
    upper_diagonal_B = np.ones(nx - 3) * (r)
    lower_diagonal_B = np.ones(nx - 3) * (r)
    B_banded = np.array([np.append(upper_diagonal_B, 0), diagonal_B, np.append(0, lower_diagonal_B)])

    # Boucle temporelle
    for j in range(0, nt - 1):
        b = np.zeros(nx - 2)
        # Calcul de B * u^j
        b = diagonal_B * u[j, 1:-1]
        if nx > 2:
            if upper_diagonal_B.size > 0:
                b[:-1] += upper_diagonal_B * u[j, 2:-1]  # Correction ici
            if lower_diagonal_B.size > 0:
                b[1:] += lower_diagonal_B * u[j, 1:-2]   # Correction ici

        # Résolution de A * u^{j+1} = b
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
    t_plot = [0, 0.02, 0.05, 0.1]
    num_terms_f2 = 50
    colors = plt.cm.viridis(np.linspace(0, 1, len(t_plot)))

    # Condition initiale 1: f1(x) = sin(2*pi*x)
    def f1(x):
        return np.sin(2 * np.pi * x)

    # Condition initiale 2: f2(x) (fonction triangle)
    def f2(x):
        if 0 <= x <= 0.5:
            return 2 * x
        elif 0.5 < x <= 1:
            return 2 * (1 - x)
        else:
            return 0

    # Résolution numérique avec le schéma de Crank-Nicolson pour f1
    x_cn1, t_cn1, u_cn1 = solve_heat_crank_nicolson(f1, nx, nt, t_final)

    # Résolution numérique avec le schéma de Crank-Nicolson pour f2
    x_cn2, t_cn2, u_cn2 = solve_heat_crank_nicolson(f2, nx, nt, t_final)

    # --- Plot 1: Cartes de couleurs combinées (Crank-Nicolson) ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    im_cn1 = plt.imshow(u_cn1, aspect='auto', extent=[0, 1, t_final, 0], cmap='plasma')
    plt.colorbar(im_cn1, label='u(x, t)', shrink=0.7)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Crank-Nicolson avec f1(x)')

    plt.subplot(1, 2, 2)
    im_cn2 = plt.imshow(u_cn2, aspect='auto', extent=[0, 1, t_final, 0], cmap='plasma')
    plt.colorbar(im_cn2, label='u(x, t)', shrink=0.7)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Crank-Nicolson avec f2(x)')

    plt.tight_layout()
    plt.savefig('heatmaps_crank_nicolson_f1_f2.png')
    plt.close()

    # --- Plot 2: Comparaison à différents instants (Crank-Nicolson avec analytique) ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1) # Premier sous-plot pour f1
    for i, time in enumerate(t_plot):
        index_t_num = np.argmin(np.abs(t_cn1 - time))
        plt.plot(x_cn1, u_cn1[index_t_num, :], color=colors[i], label=f'Numérique (t={time:.2f})')
        plt.plot(x_cn1, analytical_solution_f1(x_cn1, time), '--', color=colors[i], alpha=0.7, label=f'Analytique (t={time:.2f})')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title('Crank-Nicolson avec f1(x)')
    plt.legend(fontsize='small')
    plt.grid(True)

    plt.subplot(1, 2, 2) # Deuxième sous-plot pour f2
    for i, time in enumerate(t_plot):
        index_t_num_f2 = np.argmin(np.abs(t_cn2 - time))
        plt.plot(x_cn2, u_cn2[index_t_num_f2, :], color=colors[i], label=f'Numérique (t={time:.2f})')
        plt.plot(x_cn2, analytical_solution_f2_truncated(x_cn2, time, num_terms_f2), '--', color=colors[i], alpha=0.7, label=f'Analytique tronquée (t={time:.2f})')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title('Crank-Nicolson avec f2(x)')
    plt.legend(fontsize='small')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('comparison_slices_crank_nicolson_f1_f2_analytical.png')
    plt.close()