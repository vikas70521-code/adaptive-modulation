"""
Simulation and Performance Analysis of Adaptive Modulation
in Wireless Communication (Rayleigh block-fading channel)

Modes : No-Tx (outage), BPSK, QPSK, 16-QAM, 64-QAM  (Gray coded)
Policy: choose the highest-order modulation whose theoretical AWGN BER at the
        (estimated) instantaneous SNR is below a target BER.
Outputs: PNG plots + results.csv in ./results
"""
import math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
MODES = [2, 4, 16, 64]                     # modulation orders M
BLOCK = 100                                # symbols per fading block
N_BLOCKS = 4000                            # blocks per SNR point
TARGET_BER = 1e-3
SNR_DB = np.arange(0, 31, 2)
OUT = "results"; os.makedirs(OUT, exist_ok=True)

# ---------------- Modulation / demodulation (Gray, unit Es) ----------------
def _gray(n):  return n ^ (n >> 1)

def modulate(bits, M):
    k = int(np.log2(M))
    if M == 2:
        return 1 - 2.0 * bits
    b = bits.reshape(-1, k)
    kh = k // 2                                    # bits per axis
    L = 2 ** kh
    def axis(bb):
        g = bb @ (1 << np.arange(kh - 1, -1, -1))  # Gray index
        # inverse Gray -> level index
        n = g.copy(); s = g >> 1
        while s.any():
            n ^= s; s >>= 1
        return 2.0 * n - (L - 1)
    sym = axis(b[:, :kh]) + 1j * axis(b[:, kh:])
    return sym / math.sqrt(2 * (M - 1) / 3)

def demodulate(rx, M):
    k = int(np.log2(M))
    if M == 2:
        return (rx.real < 0).astype(int)
    kh = k // 2; L = 2 ** kh
    sc = math.sqrt(2 * (M - 1) / 3)
    def axis(v):
        n = np.clip(np.round((v * sc + (L - 1)) / 2), 0, L - 1).astype(int)
        g = _gray(n)
        return ((g[:, None] >> np.arange(kh - 1, -1, -1)) & 1)
    return np.hstack([axis(rx.real), axis(rx.imag)]).ravel()

# ---------------- Theory ----------------
def Q(x): return 0.5 * math.erfc(x / math.sqrt(2))

def ber_awgn(M, g):
    """Approx. Gray-coded BER for M-QAM / BPSK at symbol SNR g (linear)."""
    if M == 2: return Q(math.sqrt(2 * g))
    k = math.log2(M)
    return (4 / k) * (1 - 1 / math.sqrt(M)) * Q(math.sqrt(3 * g / (M - 1)))

def snr_threshold(M, target):
    lo, hi = 1e-3, 1e5
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if ber_awgn(M, mid) > target else (lo, mid)
    return hi

THRESH = [snr_threshold(M, TARGET_BER) for M in MODES]   # linear SNR thresholds

def select_mode(snr_est):
    """Return index into MODES per block, -1 = no transmission."""
    idx = np.full(snr_est.shape, -1)
    for i, t in enumerate(THRESH):
        idx[snr_est >= t] = i
    return idx

# ---------------- Link simulation ----------------
def channel(n, rho):
    h = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)
    e = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)
    h_est = rho * h + math.sqrt(1 - rho ** 2) * e      # imperfect / delayed CSI
    return h, h_est

def transmit(M, h, g):
    """Send BLOCK symbols per block with gains h at avg SNR g. Return (errors, bits)."""
    n = len(h); k = int(np.log2(M))
    bits = rng.integers(0, 2, n * BLOCK * k)
    s = modulate(bits, M).reshape(n, BLOCK)
    noise = (rng.standard_normal(s.shape) + 1j * rng.standard_normal(s.shape)) / math.sqrt(2 * g)
    y = h[:, None] * s + noise
    rx = (y / h[:, None]).ravel()                       # coherent equalisation
    return np.sum(bits != demodulate(rx, M)), bits.size

def run_fixed(M, snr_db):
    g = 10 ** (snr_db / 10); h, _ = channel(N_BLOCKS, 1.0)
    err, nb = transmit(M, h, g)
    k = np.log2(M)
    return err / nb, k * (1 - err / nb)

def run_adaptive(snr_db, rho=1.0):
    g = 10 ** (snr_db / 10); h, h_est = channel(N_BLOCKS, rho)
    idx = select_mode(g * np.abs(h_est) ** 2)
    err = bits = 0; thr = 0.0
    usage = np.zeros(len(MODES) + 1)
    usage[0] = np.mean(idx == -1)
    for i, M in enumerate(MODES):
        sel = idx == i
        usage[i + 1] = sel.mean()
        if sel.any():
            e, b = transmit(M, h[sel], g)
            err += e; bits += b
            thr += (sel.sum() / N_BLOCKS) * np.log2(M) * (1 - e / b)
    return (err / bits if bits else np.nan), thr, usage

# ---------------- Experiments ----------------
def main():
    global THRESH
    print("Switching thresholds (dB):",
          {f"M={M}": round(10 * math.log10(t), 2) for M, t in zip(MODES, THRESH)})
    names = {2: "BPSK", 4: "QPSK", 16: "16-QAM", 64: "64-QAM"}

    fixed = {M: [run_fixed(M, s) for s in SNR_DB] for M in MODES}
    adap = [run_adaptive(s) for s in SNR_DB]
    a_ber = np.array([a[0] for a in adap]); a_se = np.array([a[1] for a in adap])
    usage = np.array([a[2] for a in adap])

    # 1) BER
    plt.figure(figsize=(7, 5))
    for M in MODES:
        plt.semilogy(SNR_DB, np.maximum([f[0] for f in fixed[M]], 1e-7), "o--", label=f"Fixed {names[M]}")
    plt.semilogy(SNR_DB, np.maximum(a_ber, 1e-7), "k-s", lw=2, label="Adaptive")
    plt.axhline(TARGET_BER, color="r", ls=":", label="Target BER")
    plt.ylim(1e-5, 1); plt.grid(True, which="both", alpha=.3)
    plt.xlabel("Average SNR (dB)"); plt.ylabel("BER"); plt.title("BER: fixed vs adaptive (Rayleigh)")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/ber_comparison.png", dpi=150); plt.close()

    # 2) Spectral efficiency
    plt.figure(figsize=(7, 5))
    for M in MODES:
        plt.plot(SNR_DB, [f[1] for f in fixed[M]], "o--", label=f"Fixed {names[M]}")
    plt.plot(SNR_DB, a_se, "k-s", lw=2, label="Adaptive")
    plt.grid(alpha=.3); plt.xlabel("Average SNR (dB)"); plt.ylabel("Spectral efficiency (bits/s/Hz)")
    plt.title("Throughput: fixed vs adaptive"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/spectral_efficiency.png", dpi=150); plt.close()

    # 3) Mode usage
    plt.figure(figsize=(7, 5))
    labels = ["No Tx"] + [names[M] for M in MODES]
    plt.stackplot(SNR_DB, usage.T, labels=labels, alpha=.85)
    plt.xlabel("Average SNR (dB)"); plt.ylabel("Probability"); plt.title("Modulation mode usage")
    plt.legend(loc="center right"); plt.margins(x=0); plt.tight_layout()
    plt.savefig(f"{OUT}/mode_usage.png", dpi=150); plt.close()

    # 4) Effect of CSI quality at 20 dB
    rhos = [1.0, 0.99, 0.95, 0.9, 0.8, 0.6]
    res = [run_adaptive(20, r) for r in rhos]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].semilogy(rhos, [r[0] for r in res], "o-"); ax[0].axhline(TARGET_BER, color="r", ls=":")
    ax[0].set_xlabel("CSI correlation ρ"); ax[0].set_ylabel("BER"); ax[0].grid(True, which="both", alpha=.3)
    ax[0].set_title("BER vs CSI quality (20 dB)"); ax[0].invert_xaxis()
    ax[1].plot(rhos, [r[1] for r in res], "o-"); ax[1].set_xlabel("CSI correlation ρ")
    ax[1].set_ylabel("Spectral efficiency"); ax[1].grid(alpha=.3); ax[1].invert_xaxis()
    ax[1].set_title("Throughput vs CSI quality (20 dB)")
    plt.tight_layout(); plt.savefig(f"{OUT}/csi_impact.png", dpi=150); plt.close()

    # 5) Target BER trade-off at 20 dB (re-derive thresholds)
    base = THRESH[:]; tb = [1e-2, 1e-3, 1e-4, 1e-5]; tr = []
    for t in tb:
        THRESH = [snr_threshold(M, t) for M in MODES]; tr.append(run_adaptive(20))
    THRESH = base
    plt.figure(figsize=(6, 4.5))
    plt.plot([r[1] for r in tr], [r[0] for r in tr], "o-")
    for t, r in zip(tb, tr): plt.annotate(f"{t:g}", (r[1], r[0]), textcoords="offset points", xytext=(5, 5))
    plt.yscale("log"); plt.xlabel("Spectral efficiency (bits/s/Hz)"); plt.ylabel("Achieved BER")
    plt.title("BER target trade-off (20 dB)"); plt.grid(True, which="both", alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/target_tradeoff.png", dpi=150); plt.close()

    # CSV
    with open(f"{OUT}/results.csv", "w") as f:
        f.write("snr_db,adaptive_ber,adaptive_se,p_notx,p_bpsk,p_qpsk,p_16qam,p_64qam\n")
        for i, s in enumerate(SNR_DB):
            f.write(f"{s},{a_ber[i]:.3e},{a_se[i]:.3f}," + ",".join(f"{u:.3f}" for u in usage[i]) + "\n")
    print("Done. See ./results")
    print("SNR  AdaptiveBER  SE")
    for s, b, e in zip(SNR_DB, a_ber, a_se): print(f"{s:3d}  {b:.2e}  {e:.2f}")

if __name__ == "__main__":
    main()
