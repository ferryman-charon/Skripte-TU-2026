import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift


def compare_fft_analytical(func, analytical_solution, N=4096, L=40.0):
    """
    Vergleicht die numerische FFT einer Funktion mit ihrer analytischen Fourier-Transformation.
    """
    x = np.linspace(-L/2, L/2, N)
    dx = x[1] - x[0]
    fx = func(x)
    Fk_numerical = fftshift(fft(fx)) * dx
    k = fftshift(fftfreq(N, d=dx)) * 2 * np.pi
    Fk_analytical = analytical_solution(k)
    
    results = {
        'x': x,
        'fx': fx,
        'k': k,
        'Fk_numerical': Fk_numerical,
        'Fk_analytical': Fk_analytical
    }
    
    return results


# ============================================================================
# Verschiebungssatz der Fourier-Transformation
# ============================================================================


b = 2  # Verschiebung

def f_original(x): 
    return np.maximum(1 - np.abs(x), 0)

def F_original(k):
    result = np.zeros_like(k, dtype=complex)
    mask = k != 0
    result[mask] = 2 * (1 - np.cos(k[mask])) / k[mask]**2
    result[~mask] = 1.0
    return result

def f_shifted(x): 
    return np.maximum(1 - np.abs(x - b), 0)

def F_shifted(k):
    result = np.zeros_like(k, dtype=complex)
    mask = k != 0
    result[mask] = 2 * (1 - np.cos(k[mask])) / k[mask]**2
    result[~mask] = 1.0
    return np.exp(1j * b * k) * result


results0 = compare_fft_analytical(f_original, F_original)
results1 = compare_fft_analytical(f_shifted, F_shifted)

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
k_plot = results0['k']

# Betrag
axes2[0].plot(k_plot, np.abs(results0['Fk_analytical']), 'b-', linewidth=2, label='Original')
axes2[0].plot(k_plot, np.abs(results1['Fk_analytical']), 'r--', linewidth=2, label=f'Verschoben (b={b})')
axes2[0].set_xlabel(r"$k$", fontsize=14)
axes2[0].set_ylabel(r"$|\mathcal{F}(k)|$", fontsize=14)
axes2[0].set_title("Vergleich: Beträge", fontsize=14, fontweight='bold')
axes2[0].legend(fontsize=12)
axes2[0].grid(True, alpha=0.3)
axes2[0].set_xlim(-10, 10)

# Phase
axes2[1].plot(k_plot, np.angle(results0['Fk_analytical']), 'b-', linewidth=2, label='Original')
axes2[1].plot(k_plot, np.angle(results1['Fk_analytical']), 'r-', linewidth=2, label=f'Verschoben (b={b})')
axes2[1].plot(k_plot, b * k_plot, 'g--', linewidth=2, alpha=0.7, label=f'Lineare Phase: b·k = {b}k')
axes2[1].set_xlabel(r"$k$", fontsize=14)
axes2[1].set_ylabel(r"$\arg(\mathcal{F}(k))$", fontsize=14)
axes2[1].set_title("Vergleich: Phasen", fontsize=14, fontweight='bold')
axes2[1].legend(fontsize=12)
axes2[1].grid(True, alpha=0.3)
axes2[1].set_xlim(-10, 10)
axes2[1].set_ylim(-3*np.pi, 3*np.pi)
axes2[1].axhline(y=0, color='k', linestyle=':', alpha=0.3)

plt.tight_layout()
plt.show()