# Simulation and Performance Analysis of Adaptive Modulation Techniques in Wireless Communication

## 1. Abstract
Wireless channels fluctuate because of multipath fading, so a fixed modulation scheme is either
unreliable in deep fades or wastes capacity in good channel conditions. Adaptive modulation (AM) lets the
transmitter pick the modulation order from channel-state information (CSI). This project simulates AM over
a Rayleigh block-fading channel using BPSK, QPSK, 16-QAM and 64-QAM, and compares it with fixed
modulation in terms of BER, spectral efficiency, outage and sensitivity to CSI quality.

## 2. Objectives
1. Implement Gray-coded BPSK/QPSK/16-QAM/64-QAM modulators and demodulators.
2. Model a Rayleigh block-fading AWGN channel with coherent equalisation.
3. Design an SNR-threshold adaptation policy that meets a target BER.
4. Compare adaptive vs. fixed schemes (BER, throughput, mode usage).
5. Study the impact of imperfect/delayed CSI and of the target BER.

## 3. System model
y = h·x + n, with h ~ CN(0,1) constant over a block of 100 symbols, n ~ CN(0, 1/γ̄).
Instantaneous SNR γ = γ̄·|h|². The receiver feeds back an estimate
ĥ = ρh + √(1−ρ²)·e, where ρ is the CSI correlation (ρ = 1 is perfect CSI).

## 4. Adaptation policy
Mode k is used when γ̂ ≥ γₖ, where γₖ solves BER_AWGN(Mₖ, γₖ) = BER_target:

- BPSK: Q(√(2γ))
- M-QAM: (4/log₂M)(1 − 1/√M)·Q(√(3γ/(M−1)))

If γ̂ < γ_BPSK, nothing is sent (outage). Thresholds for BER_target = 10⁻³:
BPSK 6.8 dB, QPSK 9.8 dB, 16-QAM 16.5 dB, 64-QAM 22.6 dB.

## 5. Performance metrics
- BER over transmitted bits
- Spectral efficiency = Σ P(mode k)·log₂Mₖ·(1 − BERₖ) in bits/s/Hz
- Mode usage probability and outage probability
- Sensitivity to CSI correlation ρ and to BER target

## 6. How to run
```
pip install numpy matplotlib
python adaptive_modulation_sim.py
```
Outputs go to `results/`: `ber_comparison.png`, `spectral_efficiency.png`, `mode_usage.png`,
`csi_impact.png`, `target_tradeoff.png`, `results.csv`.

Edit the constants at the top of the script (`TARGET_BER`, `BLOCK`, `N_BLOCKS`, `SNR_DB`, `MODES`).

## 7. Key results (from the included run)
- Adaptive BER stays below the 10⁻³ target at every SNR, whereas fixed 16-QAM and 64-QAM
  stay far above it in Rayleigh fading at low and moderate SNR.
- Adaptive spectral efficiency rises smoothly from ~0 to ~5.6 bits/s/Hz (0 → 30 dB), tracking the best
  fixed scheme at each SNR instead of being locked to one.
- Degrading CSI (lower ρ) causes wrong mode choices, so BER rises above target (see `csi_impact.png`).
- A stricter BER target lowers throughput (see `target_tradeoff.png`).

## 8. Suggested report structure
1. Introduction and motivation  2. Literature review (Goldsmith & Chua 1997, variable-rate M-QAM)
3. System model  4. Adaptation algorithm  5. Simulation setup  6. Results and discussion
7. Conclusion and future work.

## 9. Future extensions
Add coding (adaptive coding and modulation), power adaptation, MIMO/OFDM per-subcarrier loading,
Nakagami-m or Rician channels, hysteresis to reduce mode switching, and a comparison with
theoretical closed-form average BER.
