"""Poster methodology figure: the INDRA-Sat-Diff pipeline with real rain maps (South India, 30 Jun 2023 09 UTC).

Inputs -> VAE encoder -> latent -> Earthformer-UNet denoiser (DDIM, knowledge alignment) -> VAE decoder ->
4-member ensemble -> probability matching -> forecast.   usage: python make_method_fig.py
Needs artifacts/results/val_cache (validation forecasts) for the thumbnails.
"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("postprocess", os.path.join(ROOT, "climate_forecast", "diffusion", "postprocess.py"))
_pp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_pp)   # plain numpy, no torch needed
combine = _pp.combine

CASE = os.path.join(ROOT, "artifacts/results/val_cache/south_jul2025/sac_valid_south/ka1/2023-06-30T0900Z.npz")
BOX = ([71.0, 83.8], [5.0, 17.8])
BINS = [0.1, 0.5, 1, 2.5, 5, 7.6, 10, 16, 25, 50, 100]
CMAP = ListedColormap(["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#31a354", "#74c476", "#fed976", "#feb24c", "#fd8d3c", "#e31a1c"])
NORM = BoundaryNorm(BINS, CMAP.N)
INK, MUTED = "#1b2430", "#5b6675"
C = {"data": "#5b6675", "vae": "#2a8c82", "latent": "#3b6fb6", "unet": "#1f4e8c", "ka": "#e8710a", "ens": "#7b3294", "pm": "#7b3294"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "savefig.transparent": True, "mathtext.fontset": "stix"})
mm = lambda u: np.expm1(np.asarray(u, dtype=np.float32) / 255.0 * math.log1p(100.0))


def coast():
    out = []
    for name in ("ne_50m_coastline", "india_states"):
        fc = json.load(open(os.path.join(ROOT, "app", "static", "vendor", f"{name}.geojson")))
        for f in fc["features"]:
            g = f["geometry"]
            for c in ([g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]):
                xy = np.asarray(c)[:, :2]
                if xy[:, 0].max() > 70 and xy[:, 0].min() < 85 and xy[:, 1].max() > 4 and xy[:, 1].min() < 19:
                    out.append((xy, name))
    return out


def thumb(ax, field_mm, x, y, w, geo, edge="#8a94a3", lw=1.2, z=3):
    """Draw a rain map inside axes-coordinates rectangle (x, y, w, w)."""
    sub = ax.inset_axes([x, y, w, w * ASPECT], transform=ax.transData, zorder=z)
    sub.set_facecolor("white")
    sub.imshow(np.ma.masked_less(np.flipud(field_mm.T), 0.1), cmap=CMAP, norm=NORM, extent=[*BOX[0], *BOX[1]],
               interpolation="nearest")
    for xy, n in geo:
        sub.plot(xy[:, 0], xy[:, 1], color="#222" if n.startswith("ne_") else "#777", lw=0.5 if n.startswith("ne_") else 0.3)
    sub.set_xlim(*BOX[0]); sub.set_ylim(*BOX[1]); sub.set_xticks([]); sub.set_yticks([])
    for s in sub.spines.values():
        s.set_color(edge); s.set_linewidth(lw)
    return sub


def box(ax, x, y, w, hgt, color, title, sub=None, fs=17):
    ax.add_patch(FancyBboxPatch((x, y), w, hgt, boxstyle="round,pad=0.012,rounding_size=0.06", fc=color, ec="none", zorder=2))
    ax.text(x + w / 2, y + hgt / 2 + (0.13 if sub else 0), title, ha="center", va="center", color="white",
            fontsize=fs, fontweight="bold", zorder=3)
    if sub:
        ax.text(x + w / 2, y + hgt / 2 - 0.2, sub, ha="center", va="center", color="white", fontsize=fs - 4, zorder=3)


def arrow(ax, x0, y0, x1, y1, color=INK, lw=2.6, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=22, color=color, lw=lw,
                                 linestyle=ls, zorder=1.5, shrinkA=0, shrinkB=0))


def trapezoid(ax, x, y, w, hgt, color, wide_left, title, sub):
    d = hgt * 0.28
    pts = [(x, y), (x + w, y + d), (x + w, y + hgt - d), (x, y + hgt)] if wide_left else \
          [(x, y + d), (x + w, y), (x + w, y + hgt), (x, y + hgt - d)]
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec="none", zorder=2))
    ax.text(x + w / 2, y + hgt / 2 + 0.16, title, ha="center", va="center", color="white", fontsize=20, fontweight="bold", zorder=3)
    ax.text(x + w / 2, y + hgt / 2 - 0.24, sub, ha="center", va="center", color="white", fontsize=16, zorder=3)


ASPECT = 1.0
def main():
    global ASPECT
    z = np.load(CASE)
    ens, truth, last = mm(z["ensemble"]), mm(z["truth"]), mm(z["last"])
    u = lambda a: np.clip(np.log1p(np.clip(a, 0, 100)) / math.log1p(100) * 255, 0, 255) / 255.0
    final = mm(combine(u(ens), u(last), "pmi", 1.0) * 255)
    geo = coast()

    W, H = 20.0, 5.55
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ASPECT = 1.0
    yc = 2.55                                                     # main flow line

    T, S_ = 21, 16                                               # title / subtitle font sizes
    def label(x, y, t, sub=None):
        ax.text(x, y, t, ha="center", fontsize=T - 2, fontweight="bold", color=INK)
        if sub:
            ax.text(x, y - 0.42, sub, ha="center", fontsize=S_, color=MUTED)

    # 1. inputs: stacked frames
    for k, off in enumerate((0.3, 0.15, 0.0)):
        thumb(ax, last, 0.15 + off, yc - 0.9 + off, 1.8, geo, z=3 + k)
    label(1.2, yc - 1.4, "IMERG input", "7 frames · 3.5 h")
    arrow(ax, 2.5, yc + 0.1, 2.88, yc + 0.1)
    # 2. encoder
    trapezoid(ax, 2.95, yc - 0.9, 1.3, 2.0, C["vae"], True, "VAE", "encoder")
    arrow(ax, 4.3, yc + 0.1, 4.62, yc + 0.1)
    # 3. latent
    for k in range(4):
        ax.add_patch(FancyBboxPatch((4.68 + k * 0.12, yc - 0.42 + k * 0.12), 0.72, 0.72, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=C["latent"], ec="white", lw=1.2, alpha=0.55 + 0.15 * k, zorder=2 + k * 0.01))
    label(5.22, yc - 0.95, "latent", "64×16×16")
    arrow(ax, 5.82, yc + 0.1, 6.14, yc + 0.1)
    # 4. denoiser + knowledge alignment
    ax.add_patch(FancyBboxPatch((6.2, yc - 0.8), 3.15, 1.8, boxstyle="round,pad=0.012,rounding_size=0.06", fc=C["unet"], ec="none", zorder=2))
    ax.text(7.78, yc + 0.42, "Earthformer-UNet", ha="center", va="center", color="white", fontsize=T, fontweight="bold", zorder=3)
    ax.text(7.78, yc - 0.28, "turns noise into the\nforecast latent, 20 steps", ha="center", va="center", color="white", fontsize=S_, zorder=3, linespacing=1.15)
    ax.add_patch(FancyBboxPatch((6.2, yc + 1.55), 3.15, 1.15, boxstyle="round,pad=0.012,rounding_size=0.06", fc=C["ka"], ec="none", zorder=2))
    ax.text(7.78, yc + 2.3, "Knowledge Alignment", ha="center", va="center", color="white", fontsize=T - 1, fontweight="bold", zorder=3)
    ax.text(7.78, yc + 1.88, "steer each step to F₀ = 1× input rain", ha="center", va="center", color="white", fontsize=S_ - 1, zorder=3)
    arrow(ax, 7.78, yc + 1.55, 7.78, yc + 1.04, color=C["ka"], lw=3.2)
    ax.text(9.5, yc + 2.12, "physics prior,\nno retraining", fontsize=S_, color=C["ka"], va="center", style="italic")
    ax.text(7.78, yc - 1.25, "run 4× from random noise", ha="center", fontsize=S_, color=C["unet"])
    arrow(ax, 9.42, yc + 0.1, 9.74, yc + 0.1)
    # 5. decoder
    trapezoid(ax, 9.8, yc - 0.9, 1.3, 2.0, C["vae"], False, "VAE", "decoder")
    arrow(ax, 11.15, yc + 0.1, 11.47, yc + 0.1)
    # 6. ensemble 2x2 (lead +90 min)
    for i, (dx, dy) in enumerate(((0, 0.92), (0.92, 0.92), (0, 0), (0.92, 0))):
        thumb(ax, ens[i, 2], 11.53 + dx, yc - 0.82 + dy, 0.86, geo, edge=C["ens"], lw=1.5)
    label(12.4, yc - 1.4, "4 members", "plausible futures")
    arrow(ax, 13.4, yc + 0.1, 13.72, yc + 0.1)
    # 7. probability matching
    ax.add_patch(FancyBboxPatch((13.78, yc - 0.8), 2.45, 1.8, boxstyle="round,pad=0.012,rounding_size=0.06", fc=C["pm"], ec="none", zorder=2))
    ax.text(15.0, yc + 0.5, "Probability\nmatching", ha="center", va="center", color="white", fontsize=T - 1, fontweight="bold", zorder=3, linespacing=1.05)
    ax.text(15.0, yc - 0.38, "where: ensemble\nhow much: obs. T0", ha="center", va="center", color="white", fontsize=S_ - 1, zorder=3, linespacing=1.15)
    arrow(ax, 16.3, yc + 0.1, 16.62, yc + 0.1)
    # 8. forecast
    for j, kk in enumerate((0, 2, 5)):
        x = 16.68 + j * 1.12
        thumb(ax, final[kk], x, yc - 0.42, 1.04, geo, edge=INK, lw=1.5)
        ax.text(x + 0.52, yc + 0.73, f"+{30 * (kk + 1)} min", ha="center", fontsize=S_, color=INK)
    label(18.4, yc - 0.95, "Forecast", "6 steps, +30 to +180 min")

    # colour bar (mm/h) bottom right
    cax = fig.add_axes([0.81, 0.06, 0.16, 0.035])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal", ticks=[0.1, 1, 5, 16, 50])
    cb.ax.tick_params(labelsize=14); cb.set_label("rain rate (mm/h)", fontsize=15)

    out = os.path.join(HERE, "figs", "method_pipeline.png")
    fig.savefig(out, dpi=250, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.08)
    print("saved", out)


if __name__ == "__main__":
    main()
