import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre, factorial

# Physical constants (in atomic units: ℏ = m_e = e = 4πε₀ = 1)
a0 = 1  

def radial_wavefunction(n, l, r):
    """
    Calculate the radial wavefunction R_{nl}(r) for hydrogen atom
    
    Parameters:
    n : int - principal quantum number (n ≥ 1)
    l : int - angular momentum quantum number (0 ≤ l < n)
    r : array - radial distance (in Bohr radii)
    
    Returns:
    R : array - radial wavefunction values
    """
    rho = 2 * r / (n * a0)
    
    norm = np.sqrt(
        (2 / (n * a0))**3 * 
        factorial(n - l - 1) / 
        (2 * n * factorial(n + l))
    )
    
    laguerre_poly = genlaguerre(n - l - 1, 2 * l + 1)
    
    R = norm * np.exp(-rho / 2) * rho**l * laguerre_poly(rho)
    
    return R

def plot_single_state(n, l):
    """Plot radial wavefunction and probability density for a single state"""
    r_max = 30 * n**2
    r = np.linspace(0.01, r_max, 1000)
    
    R = radial_wavefunction(n, l, r)
    prob_density = r**2 * R**2
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(r, R, 'b-', linewidth=2, label=f'n={n}, l={l}')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('r (Bohr radii)', fontsize=12)
    ax1.set_ylabel('R(r)', fontsize=12)
    ax1.set_title('Radial Wavefunction', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    ax2.plot(r, prob_density, 'r-', linewidth=2, label=f'n={n}, l={l}')
    ax2.set_xlabel('r (Bohr radii)', fontsize=12)
    ax2.set_ylabel('r²|R(r)|²', fontsize=12)
    ax2.set_title('Probability Density', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    energy = -13.6 / n**2
    fig.suptitle(f'Hydrogen Atom: n={n}, l={l} | Energy = {energy:.3f} eV', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_single_state(n=2, l=1)
    
