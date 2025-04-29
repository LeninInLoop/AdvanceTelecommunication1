import os
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erfc

# Constants
COHERENT_BPSK = 1
COHERENT_DBPSK = 2
NONCOHERENT_DBPSK = 3

def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))

def create_random_bits(number: int) -> np.ndarray:
    random_numbers = np.random.rand(int(number))
    return np.where(random_numbers >= 0.5, 1, 0)

def coder(input_bits: np.ndarray) -> np.ndarray:
    return np.where(input_bits == 1, 1, -1)

def decoder(input_symbols: np.ndarray) -> np.ndarray:
    return np.where(input_symbols == 1, 1, 0)

def generate_gaussian_noise(variance: float, size: int) -> np.ndarray:
    return np.sqrt(variance) * np.random.randn(int(size))

def calculate_error(a: np.ndarray, b: np.ndarray) -> float:
    return np.mean(np.where(a == b, 0, 1))

def run_coherent_bpsk_simulation(variance: float, number_of_bits: int, threshold: float) -> tuple:
    random_bits = create_random_bits(number_of_bits)
    encoded_symbols = coder(random_bits)

    noise = generate_gaussian_noise(variance=variance, size=number_of_bits)
    receiver_input = encoded_symbols + noise

    detected_symbols = np.where(receiver_input >= threshold, 1, -1)
    decoded_bits = decoder(detected_symbols)

    return decoded_bits, random_bits

# Differential Encoder
def xor_block_encoder(input_vector: np.ndarray) -> np.ndarray:
    dk_minus_1 = 0
    output_vector = np.zeros(input_vector.shape, dtype=int)
    for idx, bit in enumerate(input_vector):
        output_vector[idx] = bit ^ dk_minus_1
        dk_minus_1 = output_vector[idx]
    return output_vector

# Differential Decoder
def xor_block_decoder(input_vector: np.ndarray) -> np.ndarray:
    dk_hat_minus_1 = 0
    output_vector = np.zeros(input_vector.shape, dtype=int) # Use int for bits
    for idx, received_bit in enumerate(input_vector):
        output_vector[idx] = received_bit ^ dk_hat_minus_1
        dk_hat_minus_1 = received_bit
    return output_vector


def calculate_correlation_with_previous(a: np.ndarray) -> np.ndarray:
    if len(a) <= 1:
        return np.array([])
    return a[1:] * a[:-1]

def run_coherent_dbpsk_simulation(variance: float, number_of_bits: int, threshold: float) -> tuple:

    random_bits = create_random_bits(number_of_bits)
    differentially_encoded_bits = xor_block_encoder(random_bits)
    encoded_symbols = coder(differentially_encoded_bits)

    noise = generate_gaussian_noise(variance=variance, size=number_of_bits)
    receiver_input = encoded_symbols + noise

    detected_symbols = np.where(receiver_input >= threshold, 1, -1)
    detected_diff_bits = decoder(detected_symbols)

    decoded_original_bits = xor_block_decoder(detected_diff_bits)
    return decoded_original_bits, random_bits


def run_non_coherent_dbpsk_simulation(variance: float, number_of_bits: int, threshold: float) -> tuple:
    random_bits = create_random_bits(number_of_bits)
    differentially_encoded_bits = xor_block_encoder(random_bits)

    encoded_symbols = coder(differentially_encoded_bits)

    i_noise = generate_gaussian_noise(variance=variance, size=number_of_bits)
    q_noise = generate_gaussian_noise(variance=variance, size=number_of_bits)

    receiver_input_i = encoded_symbols + i_noise
    receiver_input_q = q_noise

    receiver_i_corr = calculate_correlation_with_previous(receiver_input_i)
    receiver_q_corr = calculate_correlation_with_previous(receiver_input_q)

    receiver = receiver_q_corr + receiver_i_corr
    decoded_bits = np.where(receiver > threshold, 0, 1)

    return decoded_bits, random_bits[1:]

def run_monte_carlo(
        snr_db_values: np.ndarray,
        number_of_bits: int,
        threshold: float,
        num_mc_runs_per_snr: int, # Renamed for clarity
        type_of_simulation: int
) -> List[float]:
    error_rates = []
    print(f"Starting Monte Carlo Simulation (Type: {type_of_simulation})")
    print(f"Bits per run: {number_of_bits}, Runs per SNR: {num_mc_runs_per_snr}")

    for snr_db in snr_db_values:
        snr_linear = 10 ** (snr_db / 10)

        variance = 1 / (2 * snr_linear)

        iteration_errors = []
        print(f"  Running SNR = {snr_db:.2f} dB (var = {variance:.4e})... ", end="")
        for i in range(num_mc_runs_per_snr):
            if type_of_simulation == COHERENT_BPSK:
                decoded_bits, original_bits = run_coherent_bpsk_simulation(
                    variance=variance,
                    number_of_bits=number_of_bits,
                    threshold=threshold
                )
            elif type_of_simulation == COHERENT_DBPSK:
                decoded_bits, original_bits = run_coherent_dbpsk_simulation(
                    variance=variance,
                    number_of_bits=number_of_bits,
                    threshold=threshold
                )
            elif type_of_simulation == NONCOHERENT_DBPSK:
                decoded_bits, original_bits = run_non_coherent_dbpsk_simulation(
                    variance=variance,
                    number_of_bits=number_of_bits,
                    threshold=threshold
                )
            else:
                raise ValueError(f"Unknown simulation type: {type_of_simulation}")

            error = calculate_error(decoded_bits, original_bits)
            iteration_errors.append(error)

        avg_error = np.mean(iteration_errors)
        error_rates.append(avg_error)
        print(f"Avg BER = {avg_error:.4e}")

    print("Monte Carlo Simulation Finished.")
    return error_rates

def plot_comparison(
        x_values: np.ndarray,
        theory_values: np.ndarray,
        theory_label: str,
        sim_values: List[float],
        sim_label: str,
        title: str,
        file_name: str = None
) -> None:
    plt.figure(figsize=(10, 7))

    plt.semilogy(x_values, theory_values, 'r-', linewidth=2, label=theory_label)
    plt.semilogy(x_values, sim_values, 'bo', markersize=8, label=sim_label)

    plt.xlabel("SNR per bit, $E_b/N_0$ (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title(title)

    min_ber = np.min([val for val in sim_values if val > 0])
    if min_ber is np.nan or min_ber <= 0: min_ber = 1e-7
    plt.ylim([max(min_ber * 0.1, 1e-8), 1.0])

    plt.grid(True, which="both", ls="-", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    if file_name:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        plt.savefig(file_name, dpi=300)
        print(f"Plot saved to {file_name}")
    plt.show()

def main():
    # --- Simulation Parameters ---
    NUM_BITS_PER_RUN = int(1e6)    # Number of bits per simulation run (adjust for speed vs accuracy)
    THRESHOLD = 0.0                # Optimal threshold for coherent BPSK/DBPSK
    NUM_MC_RUNS_PER_SNR = 25       # Number of Monte Carlo runs to average per SNR point

    # --- SNR Range ---
    SNR_DB_START_SIM = 0
    SNR_DB_END_SIM = 12
    SNR_DB_POINTS_SIM = 13
    snr_db_values_sim = np.linspace(SNR_DB_START_SIM, SNR_DB_END_SIM, SNR_DB_POINTS_SIM)
    snr_linear_sim = 10 ** (snr_db_values_sim / 10)

    # Define SNR range for the broad theoretical plot
    SNR_DB_START_THEORY = 0
    SNR_DB_END_THEORY = 15
    SNR_DB_POINTS_THEORY = 50

    snr_db_values_theory = np.linspace(SNR_DB_START_THEORY, SNR_DB_END_THEORY, SNR_DB_POINTS_THEORY)
    snr_linear_theory = 10 ** (snr_db_values_theory / 10)

    # Create directory for saving images
    os.makedirs("Images", exist_ok=True)

    # ===========================================================================================
    # Theoretical BER Curves Plot
    # ===========================================================================================
    print("\nCalculating theoretical BER curves...")
    gamma_b_theory = snr_linear_theory

    # BPSK (Coherent)
    theory_bpsk_ber = qfunc(np.sqrt(2 * gamma_b_theory))

    # BFSK (Coherent)
    theory_bfsk_coherent_ber = qfunc(np.sqrt(gamma_b_theory))

    # DBPSK (Coherent)
    theory_dbpsk_coherent_ber = 2 * qfunc(np.sqrt(2 * gamma_b_theory))

    # DBPSK (Non-Coherent)
    theory_dbpsk_noncoherent_ber = 0.5 * np.exp(-gamma_b_theory)

    plt.figure(figsize=(12, 8))
    plt.semilogy(
        snr_db_values_theory,
        theory_bpsk_ber,
        'b-',
        linewidth=2,
        label=r'BPSK (Coherent): $Q(\sqrt{2 \gamma_b})$'
    )
    plt.semilogy(
        snr_db_values_theory,
        theory_bfsk_coherent_ber,
        'g-',
        linewidth=2,
        label=r'BFSK (Coherent): $Q(\sqrt{\gamma_b})$'
    )
    plt.semilogy(
        snr_db_values_theory,
        theory_dbpsk_noncoherent_ber,
        'c--',
        linewidth=2,
        label=r'DBPSK (Non-Coherent): $\frac{1}{2} e^{-\gamma_b}$'
    )
    plt.semilogy(
        snr_db_values_theory,
        theory_dbpsk_coherent_ber,
        'r--',
        linewidth=2,
        label=r'DBPSK (Coherent): $2Q(\sqrt{2 \gamma_b})$'
    )

    plt.xlabel(r"SNR per bit, $E_b/N_0 = \gamma_b$ (dB)", fontsize=12)
    plt.ylabel("Bit Error Rate (BER)", fontsize=12)
    plt.title("Theoretical BER Performance Comparison", fontsize=14)

    plt.grid(True, which="both", ls="-", alpha=0.6)
    plt.legend(fontsize=10)

    plt.xlim([SNR_DB_START_THEORY, SNR_DB_END_THEORY])
    plt.ylim([1e-7, 1.0])
    plt.tight_layout()

    plt.savefig("Images/modulation_schemes_comparison_theory.png", dpi=300)
    print("Theoretical comparison plot saved.")
    plt.show()

    # ===========================================================================================
    # COHERENT BPSK Simulation vs Theory
    # ===========================================================================================
    print("\n--- Running Coherent BPSK Simulation ---")

    theory_bpsk_ber_sim_points = qfunc(np.sqrt(2 * snr_linear_sim))

    sim_error_rates_bpsk = run_monte_carlo(
        snr_db_values=snr_db_values_sim,
        number_of_bits=NUM_BITS_PER_RUN,
        threshold=THRESHOLD,
        num_mc_runs_per_snr=NUM_MC_RUNS_PER_SNR,
        type_of_simulation=COHERENT_BPSK
    )

    plot_comparison(
        x_values=snr_db_values_sim,
        theory_values=theory_bpsk_ber_sim_points,
        theory_label=r'BPSK Theory: $Q(\sqrt{2 \gamma_b})$',
        sim_values=sim_error_rates_bpsk,
        sim_label='BPSK Simulation',
        title="BER vs. SNR for BPSK with Coherent Detection",
        file_name="Images/bpsk_coherent_comparison.png"
    )

    # ===========================================================================================
    # COHERENT DBPSK Simulation vs Theory
    # ===========================================================================================
    print("\n--- Running Coherent DBPSK Simulation ---")

    theory_dbpsk_coherent_ber_sim_points = 2 * qfunc(np.sqrt(2 * snr_linear_sim))

    sim_error_rates_dbpsk = run_monte_carlo(
        snr_db_values=snr_db_values_sim,
        number_of_bits=NUM_BITS_PER_RUN,
        threshold=THRESHOLD,
        num_mc_runs_per_snr=NUM_MC_RUNS_PER_SNR,
        type_of_simulation=COHERENT_DBPSK
    )

    plot_comparison(
        x_values=snr_db_values_sim,
        theory_values=theory_dbpsk_coherent_ber_sim_points,
        theory_label=r'DBPSK Theory (Coherent Approx): $2Q(\sqrt{2 \gamma_b})$',
        sim_values=sim_error_rates_dbpsk,
        sim_label='DBPSK Simulation (Coherent)',
        title="BER vs. SNR for DBPSK with Coherent Detection",
        file_name="Images/dbpsk_coherent_comparison.png"
    )

    # ===========================================================================================
    # Non-COHERENT DBPSK Simulation vs Theory
    # ===========================================================================================
    print("\n--- Running Non-Coherent DBPSK Simulation ---")

    theory_dbpsk_non_coherent_ber_sim_points = 0.5 * np.exp(-snr_linear_sim)

    sim_error_rates_dbpsk_non_coherent = run_monte_carlo(
        snr_db_values=snr_db_values_sim,
        number_of_bits=NUM_BITS_PER_RUN,
        threshold=THRESHOLD,
        num_mc_runs_per_snr=NUM_MC_RUNS_PER_SNR,
        type_of_simulation=NONCOHERENT_DBPSK
    )

    plot_comparison(
        x_values=snr_db_values_sim,
        theory_values=theory_dbpsk_non_coherent_ber_sim_points,
        theory_label=r'DBPSK Theory (Coherent Approx): $\frac{1}{2} e^{-\gamma_b}$',
        sim_values=sim_error_rates_dbpsk_non_coherent,
        sim_label='DBPSK Simulation (Coherent)',
        title="BER vs. SNR for DBPSK with Non-Coherent Detection",
        file_name="Images/dbpsk_non_coherent_comparison.png"
    )

    plt.figure(figsize=(12, 8))
    plt.semilogy(
        snr_db_values_sim,
        sim_error_rates_bpsk,
        'b-',
        linewidth=2,
        label=r'BPSK (Coherent)'
    )
    plt.semilogy(
        snr_db_values_sim,
        sim_error_rates_dbpsk,
        'c--',
        linewidth=2,
        label=r'DBPSK (Non-Coherent)'
    )
    plt.semilogy(
        snr_db_values_sim,
        sim_error_rates_dbpsk_non_coherent,
        'r--',
        linewidth=2,
        label=r'DBPSK (Coherent)'
    )

    plt.xlabel(r"SNR per bit, $E_b/N_0 = \gamma_b$ (dB)", fontsize=12)
    plt.ylabel("Bit Error Rate (BER)", fontsize=12)
    plt.title("Simulation BER Performance Comparison", fontsize=14)

    plt.grid(True, which="both", ls="-", alpha=0.6)
    plt.legend(fontsize=10)

    plt.xlim([SNR_DB_START_THEORY, SNR_DB_END_THEORY])
    plt.ylim([1e-7, 1.0])
    plt.tight_layout()

    plt.savefig("Images/modulation_schemes_comparison_simulation.png", dpi=300)
    print("Simulation comparison plot saved.")
    plt.show()

    print("\nAll simulations and plots finished.")

if __name__ == '__main__':
    main()