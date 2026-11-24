"""Regenerate poster figures at print resolution (A0).
fig7 values are read off the published fig7_pod_curve.png (+-0.005); replace POD dict with raw values if available."""
import math, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 26, "axes.titlesize": 28, "axes.labelsize": 28, "legend.fontsize": 23,
                     "mathtext.fontset": "stix", "savefig.transparent": True})

# ---- Fig 7: POD vs lead time (Southern India, 2.5 mm/hr, 2022-2023) ----
lead = [30, 60, 90, 120, 150, 180]
POD = {"INDRA-Sat-Diff (with KA)": ([0.720, 0.617, 0.543, 0.484, 0.433, 0.386], "#E8710A", "^", "-"),
       "INDRA-Sat-Diff (no KA)":   ([0.705, 0.583, 0.490, 0.423, 0.375, 0.327], "#1F5FB4", "o", "-"),
       "Persistence":              ([0.742, 0.612, 0.518, 0.448, 0.392, 0.342], "#1A9E3A", "s", "--")}
fig, ax = plt.subplots(figsize=(8, 5.6))
for name, (v, c, m, ls) in POD.items():
    ax.plot(lead, v, ls, color=c, marker=m, ms=10, lw=3, label=name)
ax.set_xlabel("Lead time (min)"); ax.set_ylabel("POD")
ax.set_xticks(lead); ax.set_ylim(0.25, 0.85); ax.grid(ls=":", alpha=.6)
ax.legend(loc="upper right", frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig7_pod_curve.png", dpi=300); plt.close(fig)

# ---- Fig 2: log-normalisation (same transform as climate_forecast/datasets/preprocess_logic.py) ----
rng = np.random.default_rng(0); k, th, CLIP = 1.35, 16.0, 100.0
x = rng.gamma(k, th, 400000); x = x[x <= CLIP]
y = np.log1p(x) / np.log1p(CLIP); mu, sd = y.mean(), y.std()
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5))
a.hist(x, bins=70, density=True, color="#87CEEB", alpha=.75, label="Rain-rate samples")
xs = np.linspace(0.05, CLIP, 400)
pdf = xs**(k-1) * np.exp(-xs/th) / (math.gamma(k) * th**k); pdf /= np.trapezoid(pdf, xs)
a.plot(xs, pdf, color="navy", lw=3, label="Gamma PDF"); a.set_title("Original (heavy-tailed gamma)")
a.set_xlabel("Rain rate (mm/hr)"); a.legend(frameon=False)
b.hist(y, bins=60, density=True, color="#A8A852", alpha=.85, label="log1p(x)/log1p(100)")
ys = np.linspace(0, 1, 300)
b.plot(ys, np.exp(-(ys-mu)**2/(2*sd**2))/(sd*math.sqrt(2*math.pi)), "r--", lw=3, label=f"Normal fit μ={mu:.2f}, σ={sd:.2f}")
b.set_ylim(0, 3.7); b.set_title("Log-normalised (near-Gaussian)"); b.set_xlabel("Normalised value"); b.legend(frameon=False, loc="upper left")
for z in (a, b): z.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_lognorm.png", dpi=300); plt.close(fig)
print("fig2 fit", round(mu, 3), round(sd, 3))

# ---- KA equation ----
fig = plt.figure(figsize=(12, 1.3))
fig.text(0.5, 0.5, r"$p_{\theta,\phi}(z_t\,|\,z_{t+1},y,\mathcal{F}_0)\ \propto\ p_\theta(z_t\,|\,z_{t+1},z_{cond})\cdot"
         r"\exp\!\left(-\lambda_{\mathcal{F}}\,\|\,U_\phi(z_t,t,y)-\mathcal{F}_0(y)\,\|\right)$", ha="center", va="center", fontsize=30)
fig.savefig(f"{OUT}/eq_ka.png", dpi=300, transparent=True); plt.close(fig)

# ---- QR code to GitHub repo ----
import qrcode
qrcode.make("https://github.com/vishwajitsarnobat/INDRA-Sat-Diff", box_size=20, border=1).save(f"{OUT}/qr_github.png")
