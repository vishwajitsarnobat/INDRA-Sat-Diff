"""Poster case-study figure from the cached validation forecasts (no GPU needed).

The case is chosen objectively: the test-year sequence with the largest observed rain area (>= 1 mm/h) at
+90 min. Rows: observed, INDRA-Sat-Diff (KA x1, 4 members, probability matching), single sample with KA x2
(SAC protocol), PySTEPS 50-member mean, persistence. Columns: +30 ... +180 min.
usage: python make_case_figs.py MODEL DATASET [--year 2023] [--tag 2023-08-08T0900Z]
"""
import argparse, glob, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from climate_forecast.diffusion.postprocess import combine  # noqa: E402

BINS = [0.1, 0.5, 1, 2.5, 5, 7.6, 10, 16, 25, 50, 100]
COLORS = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#31a354", "#74c476", "#fed976", "#feb24c",
          "#fd8d3c", "#e31a1c"]
CMAP, NORM = ListedColormap(COLORS), BoundaryNorm(BINS, len(COLORS))
BOX = {"sac_valid_south": ([71.0, 83.8], [5.0, 17.8]), "sac_valid_ne": ([86.1, 98.9], [19.1, 31.9])}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "font.size": 20, "savefig.transparent": True})
mm = lambda u: np.expm1(np.asarray(u, dtype=np.float32) / 255.0 * math.log1p(100.0))
u = lambda f: np.asarray(f, dtype=np.float32) / 255.0


def lines():
    out = []
    for name in ("ne_50m_coastline", "india_states"):
        fc = json.load(open(os.path.join(ROOT, "app", "static", "vendor", f"{name}.geojson")))
        for f in fc["features"]:
            g = f["geometry"]
            for c in ([g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]):
                out.append((np.asarray(c)[:, :2], name))
    return out


def show(ax, field_mm, lon, lat, geo):
    img = np.flipud(field_mm.T)                      # stored [lon, lat], lat increasing -> north up
    ax.imshow(np.ma.masked_less(img, BINS[0]), cmap=CMAP, norm=NORM, extent=[*lon, *lat], interpolation="nearest")
    for xy, name in geo:
        ax.plot(xy[:, 0], xy[:, 1], color="k" if name.startswith("ne_") else "#555", lw=0.9 if name.startswith("ne_") else 0.5)
    ax.set_xlim(*lon); ax.set_ylim(*lat); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#999")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("dataset")
    ap.add_argument("--year", type=int, default=2023); ap.add_argument("--tag")
    ap.add_argument("--compact", action="store_true", help="3 rows x 4 lead times, for the poster")
    a = ap.parse_args()
    base = os.path.join(ROOT, "artifacts", "results", "val_cache", a.model, a.dataset)
    tags = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(base, "ka2", f"{a.year}*.npz")))
    tags = [t for t in tags if all(os.path.exists(os.path.join(base, v, t + ".npz")) for v in ("pysteps",))]
    sel_variant = "ka1" if os.path.isdir(os.path.join(base, "ka1")) and \
        all(os.path.exists(os.path.join(base, "ka1", t + ".npz")) for t in tags) else "noka"
    if not a.tag:
        a.tag = max(tags, key=lambda t: (mm(np.load(os.path.join(base, "ka2", t + ".npz"))["truth"][2]) >= 1).sum())
    z_sel, z_old = np.load(os.path.join(base, sel_variant, a.tag + ".npz")), np.load(os.path.join(base, "ka2", a.tag + ".npz"))
    z_st = np.load(os.path.join(base, "pysteps", a.tag + ".npz"))
    fixed = mm(np.clip(combine(u(z_sel["ensemble"]), u(z_sel["last"]), "pm", 1.0) * 255, 0, 255))
    rows = [("Observed (IMERG)", mm(z_sel["truth"])),
            ("INDRA-Sat-Diff\n(fixed)", fixed),
            ("Single sample\n(SAC protocol)", mm(z_old["ensemble"][0])),
            ("PySTEPS\n(50-member mean)", mm(z_st["mean"])),
            ("Persistence", np.repeat(mm(z_sel["last"])[None], 6, 0))]
    leads = list(range(6))
    if a.compact:
        rows = [rows[0], rows[1], rows[3]]
        leads = [0, 1, 3, 5]
    lon, lat = BOX[a.dataset]
    geo = [(xy, n) for xy, n in lines() if (xy[:, 0].max() > lon[0] - 1) and (xy[:, 0].min() < lon[1] + 1)
           and (xy[:, 1].max() > lat[0] - 1) and (xy[:, 1].min() < lat[1] + 1)]
    fig, axes = plt.subplots(len(rows), len(leads), figsize=(3.2 * len(leads) + 2.0, 3.25 * len(rows)),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.06})
    for r, (label, data) in enumerate(rows):
        for c, k in enumerate(leads):
            show(axes[r, c], data[k], lon, lat, geo)
            if r == 0:
                axes[r, c].set_title(f"+{30 * (k + 1)} min", fontsize=22)
        axes[r, 0].set_ylabel(label, fontsize=19, rotation=0, ha="right", va="center", labelpad=12)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, ticks=BINS)
    cb.set_label("Rain rate (mm/h)", fontsize=20); cb.ax.tick_params(labelsize=16)
    t0 = a.tag.replace("T", " ").replace("Z", " UTC")
    axes[0, 0].annotate(f"Forecast from {t0[:13]}:{t0[13:]}", xy=(0, 1.32), xycoords="axes fraction", fontsize=22, ha="left")
    out = os.path.join(HERE, "figs", "results", f"fig_case_{a.dataset.replace('sac_valid_', '')}{'_compact' if a.compact else ''}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    print("case", a.tag, "variant", sel_variant, "->", out)


if __name__ == "__main__":
    main()
