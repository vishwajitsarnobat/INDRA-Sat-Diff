"""Results figures for the INDRA-Sat-Diff InGARSS 2026 poster (A0 print).

Reads every artifacts/results/sac_2025/*/final_scores_table*.txt (the averaged score tables written
by the SAC evaluation scripts prediff_test.py / pysteps_test.py) and writes, into figs/results/:

  fig_categorical_<split>.{png,pdf}   POD / FAR / CSI vs lead time, rows = regions
  fig_fss_<split>.{png,pdf}           FSS vs lead time (one scale) and FSS vs spatial scale (two leads)
  fig_scorecard_<split>.{png,pdf}     CSI heat-table, coloured by CSI difference to Persistence
  summary.csv, summary.md             all numbers (long format) + key tables + data-driven takeaways

<split> is "valid" (2022-2023 validation; the only split with a PySTEPS run) or "test".
No score is hard-coded: everything is parsed from the tables. Re-run:  python make_results_figs.py
"""
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(REPO, "artifacts", "results", "sac_2025")
OUT = os.path.join(HERE, "figs", "results")

# Evaluation settings, as set in archive/sac_code_drive/scripts/test/{prediff_test,pysteps_test}.py
THRESH = "1 mm/h"

KA, NOKA, PERS, STEPS = "INDRA-Sat-Diff (with KA)", "INDRA-Sat-Diff (no KA)", "Persistence", "PySTEPS (STEPS)"
RENAME = {"PreDiff (Aligned)": KA, "PreDiff (Unaligned)": NOKA, "Persistence": PERS, "STEPS": STEPS}
ORDER = [KA, NOKA, PERS, STEPS]
STYLE = {KA:    dict(color="#E8710A", ls="-",  marker="^", ms=13, zorder=5),
         NOKA:  dict(color="#1F5FB4", ls="-",  marker="o", ms=11, zorder=4),
         PERS:  dict(color="#1A9E3A", ls="--", marker="s", ms=10, zorder=3),
         STEPS: dict(color="#7B3294", ls="-.", marker="D", ms=10, zorder=3)}
REGIONS = {"south": "South India", "ne": "Northeast India"}
SPLITS = {"valid": "validation 2022-2023", "test": "test set"}
INK = "#1a1a1a"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 22, "axes.labelsize": 26, "xtick.labelsize": 20,
    "ytick.labelsize": 20, "legend.fontsize": 23, "axes.linewidth": 1.4, "lines.linewidth": 3.6,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
    "xtick.major.width": 1.4, "ytick.major.width": 1.4, "xtick.major.size": 6, "ytick.major.size": 6,
    "savefig.transparent": True, "pdf.fonttype": 42, "figure.dpi": 100,
})


# ----------------------------------------------------------------------------- data
def parse_table(path):
    """-> {method: {metric: {lead_min: value}}} for one final_scores_table*.txt"""
    cols, out = None, {}
    row = re.compile(r"^\s*\d+\s+(?:Categorical|FSS)\s+(.+?)\s+(\d+) min((?:\s+-?\d*\.\d+|\s+nan)+)\s*$", re.I)
    with open(path) as fh:
        for line in fh:
            if "Score Type" in line:
                cols = re.split(r"\s{2,}", line.strip())
                continue
            m = row.match(line)
            if not m:
                continue
            vals = [float(v) for v in m.group(3).split()]
            names = [RENAME.get(c, c) for c in cols[-len(vals):]]
            for name, v in zip(names, vals):
                out.setdefault(name, {}).setdefault(m.group(1).strip(), {})[int(m.group(2))] = v
    return out


def load():
    """-> data[split][region][method][metric][lead]. The PySTEPS run carries its own Persistence
    column (slightly different sequence sampling/NaN handling), kept as method 'Persistence@STEPS'."""
    data = {}
    for path in sorted(glob.glob(os.path.join(SRC, "*", "final_scores_table*.txt"))):
        folder = os.path.basename(os.path.dirname(path))
        tokens = folder.split("_")
        region = "south" if "south" in tokens else "ne" if "ne" in tokens else None
        if region is None:
            continue
        split = "test" if "test" in tokens else "valid"
        slot = data.setdefault(split, {}).setdefault(region, {})
        for method, metrics in parse_table(path).items():
            if "pysteps" in tokens and method == PERS:
                method = PERS + "@STEPS"
            slot[method] = metrics
    return data


def series(d, method, metric, leads):
    return np.array([d[method][metric][l] for l in leads])


def leads_of(d):
    return sorted(next(iter(d.values()))["CSI"])


def fss_metrics(d):
    ms = [m for m in next(iter(d.values())) if m.startswith("FSS")]
    return sorted(ms, key=lambda s: float(re.search(r"([\d.]+)\s*km", s).group(1)))


def methods_in(d):
    return [m for m in ORDER if m in d]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------- figures
def tidy(ax):
    ax.grid(True, ls=":", lw=1.1, color="#888888", alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def pad_limits(lo, hi, frac=0.07):
    p = (hi - lo) * frac
    return lo - p, hi + p


def top_legend(fig, methods, y=0.995):
    handles = [plt.Line2D([], [], lw=3.6, **{k: v for k, v in STYLE[m].items() if k != "zorder"}) for m in methods]
    fig.legend(handles, methods, loc="upper center", bbox_to_anchor=(0.5, y), ncol=len(methods), frameon=False,
               handlelength=2.6, columnspacing=1.6, handletextpad=0.6)


def row_label(ax, text):
    ax.annotate(text, xy=(0, 0.5), xycoords="axes fraction", xytext=(-108, 0), textcoords="offset points",
                rotation=90, ha="center", va="center", fontsize=27, fontweight="bold")


def fig_categorical(split, dsplit):
    regions = [r for r in REGIONS if r in dsplit]
    metrics = [("POD", "POD  (higher = better)"), ("FAR", "FAR  (lower = better)"), ("CSI", "CSI  (higher = better)")]
    methods = [m for m in ORDER if any(m in dsplit[r] for r in regions)]
    fig, axes = plt.subplots(len(regions), 3, figsize=(19, 5.1 * len(regions) + 1.0), sharex=True, squeeze=False)
    for j, (key, label) in enumerate(metrics):
        allv = []
        for i, r in enumerate(regions):
            ax, d = axes[i, j], dsplit[r]
            leads = leads_of(d)
            for m in methods_in(d):
                v = series(d, m, key, leads)
                allv.extend(v)
                ax.plot(leads, v, **STYLE[m])
            ax.set_xticks(leads)
            ax.set_ylabel(label)
            tidy(ax)
            if i == len(regions) - 1:
                ax.set_xlabel("Lead time (min)")
            if j == 0:
                row_label(ax, REGIONS[r])
        for i in range(len(regions)):  # same y-limits per metric across regions
            axes[i, j].set_ylim(*pad_limits(min(allv), max(allv)))
    top_legend(fig, methods)
    fig.tight_layout(rect=(0.03, 0, 1, 0.925), w_pad=2.0, h_pad=1.6)
    save(fig, f"fig_categorical_{split}")


def fig_fss(split, dsplit):
    regions = [r for r in REGIONS if r in dsplit]
    methods = [m for m in ORDER if any(m in dsplit[r] for r in regions)]
    d0 = dsplit[regions[0]]
    leads, scales = leads_of(d0), fss_metrics(d0)
    km = [float(re.search(r"([\d.]+)\s*km", s).group(1)) for s in scales]
    ref_scale = scales[-2] if len(scales) > 1 else scales[0]          # 44 km neighbourhood
    lead_a, lead_b = leads[1], leads[-1]                              # +60 and +180 min
    panels = [("lead", ref_scale, f"FSS at {km[scales.index(ref_scale)]:.0f} km"),
              ("scale", lead_a, f"FSS at +{lead_a} min"), ("scale", lead_b, f"FSS at +{lead_b} min")]
    fig, axes = plt.subplots(len(regions), 3, figsize=(19, 5.1 * len(regions) + 1.0), sharex="col", squeeze=False)
    for j, (kind, sel, label) in enumerate(panels):
        allv = []
        for i, r in enumerate(regions):
            ax, d = axes[i, j], dsplit[r]
            for m in methods_in(d):
                if kind == "lead":
                    x, v = leads, series(d, m, sel, leads)
                else:
                    x, v = km, np.array([d[m][s][sel] for s in scales])
                allv.extend(v)
                ax.plot(x, v, **STYLE[m])
            if kind == "lead":
                ax.set_xticks(leads)
                xl = "Lead time (min)"
            else:
                ax.set_xscale("log")
                ax.set_xticks(km)
                ax.set_xticklabels([f"{k:.0f}" for k in km])
                ax.minorticks_off()
                ax.set_xlim(km[0] / 1.18, km[-1] * 1.18)
                xl = "Neighbourhood scale (km)"
            ax.set_ylabel(label)
            tidy(ax)
            if i == len(regions) - 1:
                ax.set_xlabel(xl)
            if j == 0:
                row_label(ax, REGIONS[r])
        for i in range(len(regions)):
            axes[i, j].set_ylim(*pad_limits(min(allv), max(allv)))
    top_legend(fig, methods)
    fig.tight_layout(rect=(0.03, 0, 1, 0.925), w_pad=2.0, h_pad=1.6)
    save(fig, f"fig_fss_{split}")


def pers_ref(d, method):
    """Persistence run that was scored on the same sequences as `method`."""
    return PERS + "@STEPS" if method == STEPS and PERS + "@STEPS" in d else PERS


def fig_scorecard(split, dsplit):
    regions = [r for r in REGIONS if r in dsplit]
    nrows = [len(methods_in(dsplit[r])) for r in regions]
    fig, axes = plt.subplots(len(regions), 1, figsize=(13.5, 1.25 * sum(nrows) + 1.55 * len(regions) + 1.5),
                             squeeze=False, gridspec_kw=dict(height_ratios=nrows))
    lim = max(abs(series(dsplit[r], m, "CSI", leads_of(dsplit[r])) - series(dsplit[r], pers_ref(dsplit[r], m), "CSI",
              leads_of(dsplit[r]))).max() for r in regions for m in methods_in(dsplit[r]))
    lim = max(0.02, np.ceil(lim * 50) / 50)
    norm, cmap = TwoSlopeNorm(0, -lim, lim), plt.get_cmap("RdBu")
    for ax, r in zip(axes[:, 0], regions):
        d = dsplit[r]
        leads, methods = leads_of(d), methods_in(d)
        csi = np.array([series(d, m, "CSI", leads) for m in methods])
        diff = np.array([series(d, m, "CSI", leads) - series(d, pers_ref(d, m), "CSI", leads) for m in methods])
        im = ax.imshow(diff, cmap=cmap, norm=norm, aspect="auto")
        for a in range(len(methods)):
            for b in range(len(leads)):
                rgb = cmap(norm(diff[a, b]))[:3]
                fg = "white" if 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] < 0.5 else INK
                if methods[a] == PERS:
                    ax.text(b, a, f"{csi[a, b]:.3f}", ha="center", va="center", fontsize=27, fontweight="bold", color=fg)
                else:
                    ax.text(b, a - 0.16, f"{csi[a, b]:.3f}", ha="center", va="center", fontsize=27, fontweight="bold", color=fg)
                    ax.text(b, a + 0.24, f"{diff[a, b]:+.3f}", ha="center", va="center", fontsize=20, color=fg)
        ax.set_xticks(range(len(leads)))
        ax.set_xticklabels([f"+{l} min" for l in leads], fontsize=23)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=24)
        for lab, m in zip(ax.get_yticklabels(), methods):
            lab.set_color(STYLE[m]["color"])
            lab.set_fontweight("bold")
        ax.set_xticks(np.arange(-0.5, len(leads)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(methods)), minor=True)
        ax.grid(which="minor", color="white", lw=3)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{REGIONS[r]}  -  CSI", fontsize=27, fontweight="bold", loc="left", pad=44)
    fh = fig.get_figheight()
    fig.tight_layout(rect=(0, 1.75 / fh, 0.985, 1), h_pad=1.6)
    left, right = axes[0, 0].get_position().x0, axes[-1, 0].get_position().x1
    cax = fig.add_axes([left, 1.2 / fh, right - left, 0.28 / fh])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("CSI difference to Persistence (small number in each cell)", fontsize=23, labelpad=6)
    cb.set_ticks(np.linspace(-lim, lim, 5))
    cb.ax.set_xticklabels([f"{t:+.2f}" if abs(t) > 1e-9 else "0" for t in np.linspace(-lim, lim, 5)])
    cb.ax.tick_params(labelsize=20, length=5)
    cb.outline.set_visible(False)
    save(fig, f"fig_scorecard_{split}")


# ----------------------------------------------------------------------------- summary
def write_csv(data):
    path = os.path.join(OUT, "summary.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "region", "method", "metric", "lead_min", "value"])
        for split in data:
            for region in data[split]:
                for method, metrics in data[split][region].items():
                    for metric, byl in metrics.items():
                        for lead, v in sorted(byl.items()):
                            w.writerow([split, region, method.replace("@STEPS", " (PySTEPS run)"), metric, lead, f"{v:.4f}"])
    print("wrote summary.csv")


def compare(d, a, b, metric, higher_better=True):
    """sentence fragment: at how many lead times a beats b on metric, and the range of a-b."""
    leads = leads_of(d)
    diff = series(d, a, metric, leads) - series(d, b, metric, leads)
    win = diff > 0 if higher_better else diff < 0
    where = ", ".join(f"+{l}" for l, w in zip(leads, win) if w)
    txt = f"{int(win.sum())}/{len(leads)} lead times"
    if 0 < win.sum() < len(leads):
        txt += f" ({where} min)"
    return txt, diff.min(), diff.max(), diff.mean()


def write_md(data):
    L = ["# INDRA-Sat-Diff - results summary (auto-generated by make_results_figs.py)", "",
         "Source: `artifacts/results/sac_2025/*/final_scores_table*.txt`. Full long-format numbers: `summary.csv`.", "",
         "## Definitions (from `archive/sac_code_drive/scripts/test/prediff_test.py`, `pysteps_test.py`)", "",
         f"- Rain / no-rain threshold: **{THRESH}** (pixel counts as rain if rate >= 1.0), IMERG 0.1 deg grid, 30-min frames, "
         "7 input frames -> 6 forecast frames (+30 ... +180 min).",
         "- POD = H/(H+M), FAR = FA/(H+FA), CSI = H/(H+M+FA), computed per forecast frame of each sequence, then "
         "averaged over sequences (mean of per-sequence scores, not pooled counts).",
         "- FSS = 1 - MSE(Pf, Po) / (mean(Pf^2) + mean(Po^2)), Pf/Po = rain fractions in a square moving window of "
         "1, 2, 4, 8 pixels (labelled 11, 22, 44, 89 km using 0.1 deg x 111 km); same threshold. The 11 km value is pixel-scale.",
         "- INDRA-Sat-Diff: one diffusion sample per sequence (no ensemble), back-transformed from log1p scaling (clip 100 mm/h). "
         "With KA = knowledge-alignment guidance whose target is 2 x the mean intensity of the input frames; no KA = same weights, guidance off.",
         "- Persistence: last observed frame repeated for all six lead times.",
         "- PySTEPS: STEPS nowcast, Lucas-Kanade motion, **mean of 50 ensemble members**, no velocity perturbation, precip_thr 1 mm/h. "
         "Run on the validation folders only; there is no PySTEPS run on the test sets.",
         "- Caveat: the PySTEPS script cuts the file list into fixed 13-frame blocks and uses pysteps' own score functions "
         "(undefined scores are NaN and skipped), while the diffusion script scans for contiguous sequences and uses an epsilon "
         "(undefined scores become 0, or FSS 1). The two runs therefore have slightly different Persistence values; "
         "PySTEPS differences to Persistence below use the Persistence of the PySTEPS run.", ""]

    for split in [s for s in SPLITS if s in data]:
        for region in [r for r in REGIONS if r in data[split]]:
            d = data[split][region]
            leads = leads_of(d)
            rows = methods_in(d) + ([PERS + "@STEPS"] if PERS + "@STEPS" in d else [])
            L += [f"## {REGIONS[region]} - {SPLITS[split]}", "",
                  "CSI by lead time (min):", "",
                  "| Method | " + " | ".join(f"+{l}" for l in leads) + " | mean |",
                  "|---|" + "---|" * (len(leads) + 1)]
            for m in rows:
                v = series(d, m, "CSI", leads)
                L.append(f"| {m.replace('@STEPS', ' (PySTEPS run)')} | " + " | ".join(f"{x:.3f}" for x in v) + f" | {v.mean():.3f} |")
            fss = fss_metrics(d)
            L += ["", "Means over the six lead times:", "",
                  "| Method | POD | FAR | CSI | " + " | ".join(fss) + " |", "|---|---|---|---|" + "---|" * len(fss)]
            for m in rows:
                vals = [series(d, m, k, leads).mean() for k in ["POD", "FAR", "CSI"] + fss]
                L.append(f"| {m.replace('@STEPS', ' (PySTEPS run)')} | " + " | ".join(f"{x:.3f}" for x in vals) + " |")
            L.append("")

    L += ["## Headline takeaways (numbers computed from the tables; differences are absolute score differences)", ""]

    def agg(split, a, b, metrics, higher_better=True):
        """a vs b pooled over regions, metrics and lead times of one split -> 'k/N', min, max of a-b."""
        diffs = []
        for region in data[split]:
            d = data[split][region]
            if a not in d or b not in d:
                continue
            bb = pers_ref(d, a) if b == PERS else b
            for metric in (metrics(d) if callable(metrics) else metrics):
                diffs.extend(series(d, a, metric, leads_of(d)) - series(d, bb, metric, leads_of(d)))
        diffs = np.array(diffs)
        wins = (diffs > 0).sum() if higher_better else (diffs < 0).sum()
        return f"{int(wins)}/{len(diffs)}", diffs.min(), diffs.max()

    for split in [s for s in SPLITS if s in data]:
        name = SPLITS[split].capitalize()
        w, lo, hi = agg(split, KA, PERS, ["CSI"])
        wf, lof, hif = agg(split, KA, PERS, fss_metrics)
        L.append(f"- **{name}, full model vs Persistence:** with-KA CSI beats Persistence in {w} region x lead-time cases "
                 f"(difference {lo:+.3f} to {hi:+.3f}); FSS (all four scales) beats Persistence in {wf} cases ({lof:+.3f} to {hif:+.3f}).")
        wp, lop, hip = agg(split, KA, PERS, ["POD"])
        wa, loa, hia = agg(split, KA, PERS, ["FAR"], higher_better=False)
        L.append(f"- **{name}, why:** with-KA POD is above Persistence in {wp} cases ({lop:+.3f} to {hip:+.3f}), but its FAR is "
                 f"lower (better) than Persistence in only {wa} cases (FAR difference {loa:+.3f} to {hia:+.3f}).")
        wp, lop, hip = agg(split, KA, NOKA, ["POD"])
        wa, loa, hia = agg(split, KA, NOKA, ["FAR"], higher_better=False)
        wc, loc, hic = agg(split, KA, NOKA, ["CSI"])
        L.append(f"- **{name}, effect of KA:** KA raises POD in {wp} cases ({lop:+.3f} to {hip:+.3f}), lowers FAR in {wa} cases "
                 f"(FAR change {loa:+.3f} to {hia:+.3f}), and improves CSI in {wc} cases ({loc:+.3f} to {hic:+.3f}).")
        if any(STEPS in data[split][r] for r in data[split]):
            w, lo, hi = agg(split, KA, STEPS, ["CSI"])
            wf, lof, hif = agg(split, KA, STEPS, fss_metrics)
            ws, los, his = agg(split, STEPS, PERS, ["CSI"])
            L.append(f"- **{name}, full model vs PySTEPS:** with-KA CSI beats PySTEPS in {w} cases ({lo:+.3f} to {hi:+.3f}) and FSS in "
                     f"{wf} cases ({lof:+.3f} to {hif:+.3f}). PySTEPS (50-member ensemble mean) beats its own Persistence on CSI in {ws} cases ({los:+.3f} to {his:+.3f}).")
        else:
            L.append(f"- **{name}:** no PySTEPS run exists for this split, so no optical-flow comparison can be made here.")
    L.append("")
    L += ["## Per-dataset detail", ""]
    n = 0
    for split in [s for s in SPLITS if s in data]:
        for region in [r for r in REGIONS if r in data[split]]:
            d = data[split][region]
            tag = f"{REGIONS[region]}, {SPLITS[split]}"
            fss44 = fss_metrics(d)[-2]
            t, lo, hi, mean = compare(d, KA, PERS, "CSI")
            t2, lo2, hi2, _ = compare(d, KA, PERS, fss44)
            n += 1
            L.append(f"{n}. **{tag} - model vs Persistence:** with-KA CSI is above Persistence at {t} "
                     f"(difference {lo:+.3f} to {hi:+.3f}, mean {mean:+.3f}); {fss44} is above Persistence at {t2} ({lo2:+.3f} to {hi2:+.3f}).")
            t, lo, hi, mean = compare(d, KA, NOKA, "CSI")
            tp, lop, hip, _ = compare(d, KA, NOKA, "POD")
            tf, lof, hif, _ = compare(d, KA, NOKA, "FAR", higher_better=False)
            n += 1
            L.append(f"{n}. **{tag} - effect of KA:** KA raises POD at {tp} ({lop:+.3f} to {hip:+.3f}) but lowers FAR at only {tf} "
                     f"(FAR change {lof:+.3f} to {hif:+.3f}); net CSI is higher with KA at {t} ({lo:+.3f} to {hi:+.3f}, mean {mean:+.3f}).")
            tf, lof, hif, _ = compare(d, KA, PERS, "FAR", higher_better=False)
            tp, lop, hip, _ = compare(d, KA, PERS, "POD")
            n += 1
            L.append(f"{n}. **{tag} - detection vs false alarms:** with-KA POD exceeds Persistence at {tp} ({lop:+.3f} to {hip:+.3f}); "
                     f"its FAR is lower than Persistence at {tf} (FAR difference {lof:+.3f} to {hif:+.3f}).")
            if STEPS in d:
                t, lo, hi, mean = compare(d, KA, STEPS, "CSI")
                ts, los, his, _ = compare(d, STEPS, pers_ref(d, STEPS), "CSI")
                tf, lof, hif, _ = compare(d, KA, STEPS, "FAR", higher_better=False)
                n += 1
                L.append(f"{n}. **{tag} - model vs PySTEPS:** with-KA CSI is above PySTEPS at {t} (difference {lo:+.3f} to {hi:+.3f}, "
                         f"mean {mean:+.3f}); with-KA FAR is lower than PySTEPS at {tf} ({lof:+.3f} to {hif:+.3f}). PySTEPS itself beats "
                         f"its Persistence on CSI at {ts} ({los:+.3f} to {his:+.3f}). PySTEPS is a 50-member ensemble mean; the model is a single sample.")
    L += ["", "Reading notes:", "",
          "- No confidence intervals exist in the source tables; differences of a few thousandths should not be called significant.",
          "- The validation and test sets differ strongly in base scores (test-set Persistence CSI is much higher), so the two "
          "splits should not be mixed in one comparison.",
          "- There is no PySTEPS result on the test sets, so the model-vs-PySTEPS statement rests on validation only.", ""]
    with open(os.path.join(OUT, "summary.md"), "w") as fh:
        fh.write("\n".join(L))
    print("wrote summary.md")


def main():
    os.makedirs(OUT, exist_ok=True)
    data = load()
    if not data:
        raise SystemExit(f"no score tables found under {SRC}")
    for split, dsplit in data.items():
        fig_categorical(split, dsplit)
        fig_fss(split, dsplit)
        fig_scorecard(split, dsplit)
    write_csv(data)
    write_md(data)


if __name__ == "__main__":
    main()
