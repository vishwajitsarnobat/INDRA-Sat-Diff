"""Poster figure: validation result of the inference-time fix (reads tools/val_study.py JSON output).

Left: CSI vs lead time on the test year; right: CSI gain over persistence with 95 % bootstrap CI.
PySTEPS gain is taken from the SAC tables (all 2022-23 sequences, its own persistence) for reference.
usage: python make_validation_figs.py RESULTS_DIR
"""
import json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "figs", "results")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 22, "axes.labelsize": 24, "legend.fontsize": 19, "xtick.labelsize": 20,
                     "ytick.labelsize": 20, "savefig.transparent": True, "axes.spines.top": False,
                     "axes.spines.right": False})
C = {"fixed": "#E8710A", "old": "#1F5FB4", "pers": "#1A9E3A", "steps": "#7B3294"}
REGIONS = [("south_jul2025", "sac_valid_south", "Southern India", "plots_pysteps_valid_south", "southern_india"),
           ("ne_jul2025", "sac_valid_ne", "Northeast India", "plots_pysteps_valid_ne", "northeastern_india")]
LEADS = [30, 60, 90, 120, 150, 180]


def pysteps_gain(folder, name):
    p = os.path.join(ROOT, "artifacts", "results", "sac_2025", folder, f"final_scores_table_{name}.txt")
    if not os.path.exists(p):
        return None
    rows = [l.split() for l in open(p) if re.search(r"\bCSI\b", l) and "min" in l]
    steps = [float(r[-2]) for r in rows][:6]
    pers = [float(r[-1]) for r in rows][:6]
    return np.array(steps) - np.array(pers)


def main(results_dir):
    os.makedirs(OUT, exist_ok=True)
    avail = [(r, json.load(open(os.path.join(results_dir, f"val_study_{r[0]}_{r[1]}.json"))))
             for r in REGIONS if os.path.exists(os.path.join(results_dir, f"val_study_{r[0]}_{r[1]}.json"))]
    avail = [(r, d) for r, d in avail if d.get("test")]
    fig, axes = plt.subplots(len(avail), 2, figsize=(16, 6.2 * len(avail)), squeeze=False)
    for row, ((model, ds, title, steps_dir, steps_name), d) in enumerate(avail):
        t = d["test"]
        sel = f"{d['selected']['variant']}/{d['selected']['method']}"
        v, meth = d["selected"]["variant"], d["selected"]["method"]
        ka = "no KA" if v == "noka" else f"KA ×{v[2:]}"
        how = {"pm": "prob. matching", "prob": "prob. threshold", "mean": "ensemble mean", "member": "1 sample"}[meth]
        sel_label = f"INDRA-Sat-Diff: 4 members, {how}, {ka}"
        csi = lambda key: np.array(t[key]["mean"])[:, 2] if key != "persistence" else np.array(t["persistence"])[:, 2]
        ax = axes[row, 0]
        ax.plot(LEADS, csi("persistence"), "--s", color=C["pers"], lw=3.5, ms=11, label="Persistence")
        ax.plot(LEADS, csi("ka2/member"), "-o", color=C["old"], lw=3.5, ms=11, label="Single sample, KA ×2 (SAC eval)")
        ax.plot(LEADS, csi(sel), "-^", color=C["fixed"], lw=4, ms=13, label=sel_label)
        ax.set_ylabel("CSI (≥1 mm/h)"); ax.set_xticks(LEADS); ax.grid(ls=":", alpha=.6)
        ax.set_title(f"{title}, {d['test_year']} ({t['n_sequences']} sequences)", fontsize=22, loc="left")
        ax2 = axes[row, 1]
        lo, hi = (np.array(x) for x in t[sel]["csi_minus_persistence_ci95"])
        gain = csi(sel) - csi("persistence")
        ax2.axhline(0, color=C["pers"], ls="--", lw=2.5)
        ax2.fill_between(LEADS, lo, hi, color="#F8D2AE", lw=0, label="95 % CI", zorder=1)
        ax2.plot(LEADS, gain, "-^", color=C["fixed"], lw=4, ms=13, label="INDRA-Sat-Diff (calibrated)", zorder=4)
        old = csi("ka2/member") - csi("persistence")
        ax2.plot(LEADS, old, "-o", color=C["old"], lw=3.5, ms=11, label="Single sample (SAC eval)")
        if "pysteps/prob" in t:
            ps = np.array(t["pysteps/prob"]["mean"])[:, 2]
            ax.plot(LEADS, ps, "-.D", color=C["steps"], lw=3, ms=10, label="PySTEPS, 50 members (same sequences)")
            ax2.plot(LEADS, ps - csi("persistence"), "-.D", color=C["steps"], lw=3, ms=10,
                     label="PySTEPS, 50 members (same sequences)")
        else:
            g = pysteps_gain(steps_dir, steps_name)
            if g is not None:
                ax2.plot(LEADS, g, "-.D", color=C["steps"], lw=3, ms=10, label="PySTEPS (SAC eval, 2022–23)")
        ax2.set_ylabel("CSI gain over persistence"); ax2.set_xticks(LEADS); ax2.grid(ls=":", alpha=.6)
        for a in (ax, ax2):
            a.set_xlabel("Lead time (min)")
        ax.legend(frameon=False, loc="upper right", fontsize=17)
        if row == 0:
            h, l = ax2.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.09 / len(avail), 1, 1))
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=19)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_validation_fix.{ext}"), dpi=300)
    print("saved", os.path.join(OUT, "fig_validation_fix.png"))


if __name__ == "__main__":
    main(sys.argv[1])
