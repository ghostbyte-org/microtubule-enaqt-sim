import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# --- CONFIGURATION (The "Messy" Biology) ---
NUM_SITES = 7        # Tryptophan chain length
COUPLING = 1.0       # Connection strength (J)
TIME_STEPS = 100     # Duration (fs)
DT = 0.5             # Time step size
DISORDER_STRENGTH = 5.0  # <--- HUGE DISORDER (The "Mess")
SAMPLES = 50         # Average over 50 different random "messy" chains

# We compare 3 regimes:
# 1. Pure Quantum (No noise)
# 2. Optimal Noise (The "Sweet Spot")
# 3. High Noise (Too hot)
DEPHASING_RATES = [0.0, 1.5, 10.0] 

print(f"--- GHOSTBYTE SIM: SEARCHING FOR ENAQT ---")
print(f"[*] Injecting Static Disorder: {DISORDER_STRENGTH}")
print(f"[*] Averaging over {SAMPLES} biological samples...")

def get_messy_hamiltonian(n_sites, coupling, disorder):
    # Create a jagged energy landscape
    H = np.zeros((n_sites, n_sites))
    
    # Random site energies (Anderson Disorder)
    site_energies = np.random.uniform(-disorder, disorder, n_sites)
    
    for i in range(n_sites):
        H[i, i] = site_energies[i]
        
    # Couplings (interactions)
    for i in range(n_sites - 1):
        H[i, i+1] = coupling
        H[i+1, i] = coupling
    return H

def simulate_ensemble(dephasing_rate):
    # Store the efficiency for each time step
    avg_efficiency = np.zeros(TIME_STEPS)
    
    # Run the experiment SAMPLES times (Monte Carlo)
    for s in range(SAMPLES):
        # Generate a NEW random messy protein each time
        H = get_messy_hamiltonian(NUM_SITES, COUPLING, DISORDER_STRENGTH)
        
        rho = np.zeros((NUM_SITES, NUM_SITES), dtype=complex)
        rho[0, 0] = 1.0 # Start at input
        
        sample_trajectory = []
        
        for t in range(TIME_STEPS):
            # 1. Coherent Evolution
            U = expm(-1j * H * DT)
            rho = U @ rho @ U.conj().T
            
            # 2. Dephasing (Noise)
            if dephasing_rate > 0:
                decay = np.exp(-dephasing_rate * DT)
                mask = np.eye(NUM_SITES) + (1 - np.eye(NUM_SITES)) * decay
                rho = rho * mask
            
            # Measure target
            sample_trajectory.append(np.real(rho[NUM_SITES-1, NUM_SITES-1]))
            
        avg_efficiency += np.array(sample_trajectory)
        
    # Normalize
    return avg_efficiency / SAMPLES

# --- PLOT THE EVIDENCE ---
plt.figure(figsize=(10, 6))

colors = ['blue', 'orange', 'red']
labels = ['Pure Quantum (Rate=0)', 'Optimal Noise (Rate=1.5)', 'Too Hot (Rate=10)']

for i, rate in enumerate(DEPHASING_RATES):
    print(f"[*] Simulating Rate {rate}...")
    data = simulate_ensemble(rate)
    plt.plot(data, linewidth=2.5, color=colors[i], label=labels[i])

plt.title(f"PROOF: Noise Beats Quantum (Disorder={DISORDER_STRENGTH})")
plt.xlabel("Time (fs)")
plt.ylabel("Transport Efficiency")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()
