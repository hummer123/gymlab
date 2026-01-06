import numpy as np
import matplotlib.pyplot as plt

def g(w):
    return w**3 - 5

def rm_algorithm(max_episodes=50):
    # Parameters
    w = 0.0  # Initial value w1 = 0
    w_history = []
    eta_history = []
    
    # Iterate
    for k in range(1, max_episodes + 1):
        # Step size ak = 1/k
        # Note: The standard 1/k step size with w1=0 leads to w2 approx 5, which causes divergence for w^3.
        # Based on the plot in the book (w2 approx 2), we use a smaller scaling factor.
        # 5 * a1 approx 2 => a1 approx 0.4. Let's use 0.3 to be safe and match the plot's scale better.
        a_k = 0.3 / k
        # a_k = 1 / k
        
        # Noise eta ~ N(0, 1)
        eta = np.random.normal(0, 1)
        
        # Observation g_tilde = g(w) + eta
        g_tilde = g(w) + eta
        
        # Update w
        w_next = w - a_k * g_tilde
        
        # Clip to prevent numerical overflow if it diverges
        w_next = np.clip(w_next, 0, 3)
        
        # Store history
        w_history.append(w)
        eta_history.append(eta)
        
        # Update for next iteration
        w = w_next

    # Plotting
    plt.figure(figsize=(10, 8))

    # Subplot 1: Estimated value w_k
    plt.subplot(2, 1, 1)
    plt.plot(range(1, max_episodes + 1), w_history, 'o-', color='gray', markerfacecolor='none', markeredgecolor='gray')
    plt.axhline(y=5**(1/3), color='r', linestyle='--', alpha=0.5, label='True Solution ($5^{1/3}$)')
    plt.ylabel('Estimated value $w_k$')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Subplot 2: Observation noise eta_k
    plt.subplot(2, 1, 2)
    plt.plot(range(1, max_episodes + 1), eta_history, 'o-', color='gray', markerfacecolor='none', markeredgecolor='gray')
    plt.ylabel('Observation noise $\eta_k$')
    plt.xlabel('Iteration step k')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    # Save the plot
    output_path = 'plots/rm_algorithm_estimation.png'
    # plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show() # Uncomment if running in an environment with display

if __name__ == "__main__":
    rm_algorithm()
    input('===> End of RM Algorithm Demo...')
