"""Build the InGARSS 2026 A0 poster for INDRA-Sat-Diff from the official template.

Usage: .venv/bin/python docs/ingarss2026/poster/build_poster.py [template.pptx] [out.pptx]
Needs python-pptx. Figures: ../paper/latex/figures (paper figures, kept locally) and ./figs
(make_figs.py, make_validation_figs.py, make_case_figs.py). Export to PDF with LibreOffice:
  soffice --headless --convert-to pdf INDRA-Sat-Diff_InGARSS2026_poster.pptx
Text heights are measured with Liberation Serif (metric-compatible with Times New Roman).
"""
import copy, os, re, sys
from lxml import etree
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Cm, Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_FIGS = os.path.join(HERE, "..", "paper", "latex", "figures")
NEW_FIGS = os.path.join(HERE, "figs")
CACHE = os.path.join(HERE, "figs", "_transparent")
FONT = "Times New Roman"
TTF = {(False, False): "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
       (True, False): "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
       (False, True): "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
       (True, True): "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf"}

# Template sizes (do not change, per template instructions)
H, SUB, BODY, CAP, REF, TBL = 39.86, 36.24, 30.8, 28.99, 25.37, 27.17
LINE = 1.15          # rendered line height factor for Liberation Serif
INS_LR, INS_TB = 0.3275, 0.1637   # template text-box insets (cm)
GREEN = RGBColor(0x33, 0x99, 0x66)

PANELS = {"TL": (2.4, 10.2), "BL": (2.3, 62.1), "TR": (42.7, 10.2), "BR": (42.7, 62.1)}
PW, PH, PAD = 39.2, 50.9, 0.55
KEEP = {84, 86, 87, 98, 99, 100, 101, 111, 112, 113, 114, 117, 120}

_fonts = {}
def font(pt, b=False, i=False):
    k = (round(pt * 10), b, i)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(TTF[(b, i)], size=int(pt * 10))  # 10 px per pt
    return _fonts[k]

def runs_of(text, bold=False):
    """'**x**' -> bold, '_x_' -> italic."""
    out = []
    for part in re.split(r"(\*\*.+?\*\*|_[^_]+?_)", text):
        if not part:
            continue
        if part.startswith("**"):
            out.append((part[2:-2], True, False))
        elif part.startswith("_") and part.endswith("_") and len(part) > 2:
            out.append((part[1:-1], bold, True))
        else:
            out.append((part, bold, False))
    return out

def n_lines(runs, pt, width_cm, indent_cm=0.0):
    width_px = (width_cm - 2 * INS_LR - indent_cm) / 2.54 * 72 * 10
    words = []
    for t, b, i in runs:
        for w in re.split(r"(\s+)", t):
            if w:
                words.append((w, font(pt, b, i)))
    lines, cur = 1, 0.0
    for w, f in words:
        wl = f.getlength(w)
        if w.isspace():
            cur += wl
            continue
        if cur + wl > width_px and cur > 0:
            lines += 1
            cur = wl
        else:
            cur += wl
    return lines

def pt2cm(pt):
    return pt / 72 * 2.54

def set_font(run, pt, b=False, i=False, color=None):
    f = run.font
    f.size, f.bold, f.italic, f.name = Pt(pt), b, i, FONT
    f.color.rgb = color or RGBColor(0, 0, 0)
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rPr, qn(tag))
        el.set("typeface", FONT)

def para_fmt(p, align, before_pt=0, indent_cm=0.0):
    p.alignment = align
    p.line_spacing = 1.0
    p.space_before = Pt(before_pt)
    p.space_after = Pt(0)
    if indent_cm:
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(Cm(indent_cm))))
        pPr.set("indent", str(-int(Cm(indent_cm))))


class Panel:
    def __init__(self, slide, key):
        self.slide = slide
        self.x0, self.y0 = PANELS[key]
        self.x = self.x0 + PAD
        self.w = PW - 2 * PAD
        self.y = self.y0 + 0.35
        self.bottom = self.y0 + PH

    # ---- text ----
    def text(self, items, w=None, x=None, gap=0.15):
        """items: list of (style, text). style in h, sub, body, bullet, num, cap, ref, small."""
        w = w or self.w
        x = self.x if x is None else x
        tb = self.slide.shapes.add_textbox(Cm(x), Cm(self.y), Cm(w), Cm(1))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Cm(INS_LR)
        tf.margin_top = tf.margin_bottom = Cm(INS_TB)
        tf.auto_size = None
        height = 2 * INS_TB
        for k, (style, txt) in enumerate(items):
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            pt, align, ind, before, bold = {
                "h": (H, PP_ALIGN.LEFT, 0, 0, True),
                "sub": (SUB, PP_ALIGN.LEFT, 0, 14, False),
                "body": (BODY, PP_ALIGN.JUSTIFY, 0, 8, False),
                "bullet": (BODY, PP_ALIGN.LEFT, 0.9, 6, False),
                "num": (BODY, PP_ALIGN.LEFT, 1.1, 6, False),
                "cap": (CAP, PP_ALIGN.CENTER, 0, 0, False),
                "capl": (CAP, PP_ALIGN.JUSTIFY, 0, 0, False),
                "ref": (REF, PP_ALIGN.LEFT, 1.1, 2, False),
                "small": (REF, PP_ALIGN.JUSTIFY, 0, 4, False),
            }[style]
            if k == 0:
                before = 0
            if style == "bullet":
                txt = "•\t" + txt
            para_fmt(p, align, before, ind)
            if style in ("bullet", "num", "ref"):
                pPr = p._p.get_or_add_pPr()
                tabs = etree.SubElement(pPr, qn("a:tabLst"))
                tab = etree.SubElement(tabs, qn("a:tab"))
                tab.set("pos", str(int(Cm(ind)))); tab.set("algn", "l")
            rs = runs_of(txt, bold)
            for t, b, i in rs:
                r = p.add_run(); r.text = t
                set_font(r, pt, b, i)
            height += n_lines(rs, pt, w, ind) * pt2cm(pt) * LINE + pt2cm(before)
        tb.height = Cm(height)
        self.y += height + gap
        return tb

    # ---- pictures ----
    def picture(self, path, w=None, x=None, advance=True):
        w = w or self.w
        x = self.x + (self.w - w) / 2 if x is None else x
        with Image.open(path) as im:
            h = w * im.height / im.width
        self.slide.shapes.add_picture(path, Cm(x), Cm(self.y), Cm(w), Cm(h))
        if advance:
            self.y += h + 0.2
        return h

    def table(self, rows, col_w, header=True, x=None):
        x = self.x + (self.w - sum(col_w)) / 2 if x is None else x
        row_h = []
        for r in rows:
            row_h.append(max(n_lines(runs_of(c), TBL, cw + 2 * INS_LR - 0.5) for c, cw in zip(r, col_w))
                         * pt2cm(TBL) * LINE + 0.45)
        gf = self.slide.shapes.add_table(len(rows), len(col_w), Cm(x), Cm(self.y), Cm(sum(col_w)), Cm(sum(row_h)))
        tbl = gf.table
        tblPr = tbl._tbl.tblPr
        for a in ("firstRow", "bandRow"):
            tblPr.set(a, "0")
        sid = tblPr.find(qn("a:tableStyleId"))
        if sid is not None:
            sid.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # 'No Style, Table Grid'
        for j, cw in enumerate(col_w):
            tbl.columns[j].width = Cm(cw)
        for i, r in enumerate(rows):
            tbl.rows[i].height = Cm(row_h[i])
            for j, c in enumerate(r):
                cell = tbl.cell(i, j)
                cell.margin_left = cell.margin_right = Cm(0.25)
                cell.margin_top = cell.margin_bottom = Cm(0.1)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                cell.fill.background()
                p = cell.text_frame.paragraphs[0]
                para_fmt(p, PP_ALIGN.CENTER if (header and i == 0) or j > 0 else PP_ALIGN.LEFT)
                for t, b, it in runs_of(c, bold=(header and i == 0)):
                    run = p.add_run(); run.text = t
                    set_font(run, TBL, b, it)
                tcPr = cell._tc.get_or_add_tcPr()
                for side in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                    ln = etree.SubElement(tcPr, qn(side), w="23000", cap="flat", cmpd="sng")
                    sf = etree.SubElement(ln, qn("a:solidFill"))
                    etree.SubElement(sf, qn("a:srgbClr"), val="339966")
                # schema order: borders must precede fill
                for fill in tcPr.findall(qn("a:noFill")):
                    tcPr.remove(fill); tcPr.append(fill)
        self.y += sum(row_h) + 0.3

    def gap(self, cm):
        self.y += cm

    def check(self, name):
        over = self.y - self.bottom
        print(f"{name}: used to y={self.y:.1f} / bottom {self.bottom:.1f}  ({'OVER by %.1f' % over if over > 0 else 'free %.1f' % -over} cm)")


def relabel_fig6(im):
    """Remove the debug title and code-style row labels from the paper's qualitative figure."""
    im = im.crop((0, 40, im.width, im.height))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 262, im.height - 45), fill=(255, 255, 255, 255))
    f = ImageFont.truetype(TTF[(True, False)], 44)
    labels = [("Input", "(context)"), ("IMERG", "truth"), ("With KA", ""), ("No KA", "")]
    for (a, b), yc in zip(labels, (119, 354, 588, 822)):
        for k, s in enumerate([a, b] if b else [a]):
            w = d.textlength(s, font=f)
            d.text((255 - w, yc - (50 if b else 25) + 52 * k), s, font=f, fill=(0, 0, 0, 255))
    return im

def transparent(name):
    """Flood-fill the white margin of a paper figure from its corners so it blends with the poster,
    then crop to the content."""
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, name)
    if not os.path.exists(out):
        im = Image.open(os.path.join(PAPER_FIGS, name)).convert("RGBA")
        if name.startswith("fig6"):
            im = relabel_fig6(im)
        if name.startswith("fig8"):
            im = im.crop((0, int(im.height * 0.105), im.width, im.height))
        W, Hh = im.size
        for seed in [(0, 0), (W - 1, 0), (0, Hh - 1), (W - 1, Hh - 1)]:
            if min(im.getpixel(seed)[:3]) > 245:
                ImageDraw.floodfill(im, seed, (255, 255, 255, 0), thresh=12)
        bbox = im.getchannel("A").getbbox()
        if bbox:
            pad = 6
            im = im.crop((max(bbox[0] - pad, 0), max(bbox[1] - pad, 0), min(bbox[2] + pad, W), min(bbox[3] + pad, Hh)))
        im.save(out)
    return out

def fig(name):
    return os.path.join(NEW_FIGS, name)

def set_text_keep_style(shape, lines):
    """Replace text of a template box, keeping each paragraph's first-run formatting."""
    paras = shape.text_frame.paragraphs
    for p, txt in zip(paras, lines):
        runs = p.runs
        runs[0].text = txt
        for r in runs[1:]:
            r._r.getparent().remove(r._r)


RESULT_BULLETS = [
    "**South India:** with calibrated inference, CSI exceeds persistence at all lead times (significant up to "
    "+150 min) and PySTEPS at +30 and +60 min; it also leads for heavy rain (≥ 7.6 mm/h) at +60 to +150 min, "
    "with the lowest MAE and RMSE.",
    "**Northeast India:** no gain yet. Calibration raises CSI by 0.04-0.07 over the original protocol, but "
    "scores remain below persistence and PySTEPS. The Northeast model uses the VAE trained on South India; training a "
    "Northeast VAE is the next step.",
    "**Future work:** larger ensembles, finishing the retraining, all ~900 sequences per region, "
    "INSAT-3DS HEM fine-tuning.",
]


def build(template, out):
    prs = Presentation(template)
    s = prs.slides[0]
    for sh in list(s.shapes):
        if sh.shape_id not in KEEP:
            sh._element.getparent().remove(sh._element)
    byid = {sh.shape_id: sh for sh in s.shapes}

    # ---------- header ----------
    t = byid[84]
    set_text_keep_style(t, ["INDRA-Sat-Diff: High-Fidelity Generative Precipitation Nowcasting for the "
                            "Indian Monsoon Using a Satellite-Adapted Latent Diffusion Model"])
    t.left, t.top, t.width, t.height = Cm(6.5), Cm(0.5), Cm(71), Cm(4.3)
    t.text_frame.vertical_anchor = MSO_ANCHOR.TOP
    a = byid[86]
    set_text_keep_style(a, ["Vishwajit Sarnobat¹, Vinayak Yadav¹, Abhishek Chhari², Yateen Vaviya¹",
                            "¹Sardar Patel Institute of Technology (SPIT), Mumbai   ²Space Applications Centre (SAC), ISRO, Ahmedabad"])
    a.left, a.top, a.width = Cm(6.5), Cm(4.9), Cm(71)
    set_text_keep_style(byid[87], ["ID: 1571324784"])
    byid[87].left, byid[87].top, byid[87].width = Cm(1.6), Cm(7.6), Cm(14)
    set_text_keep_style(byid[111], ["Date: 04 Dec 2026"])
    byid[111].left, byid[111].top, byid[111].width = Cm(68.0), Cm(7.6), Cm(15)
    byid[111].text_frame.paragraphs[0].alignment = PP_ALIGN.RIGHT

    # ---------- TL: I + II ----------
    p = Panel(s, "TL")
    p.text([("h", "I. Introduction"),
            ("body", "Monsoon convection over India can exceed **100 mm/h** within minutes. Radar coverage is "
                     "sparse, so satellites give the only continuous view. Warnings need **0-3 h nowcasts**:"),
            ("bullet", "**NWP**: 30-90 min spin-up; 3-10 km grids do not resolve fast convection."),
            ("bullet", "**Optical flow** (PySTEPS) moves rain but cannot create or dissipate it."),
            ("sub", "A. Contributions"),
            ("num", "1.\tFirst **latent diffusion** nowcaster on IMERG for Southern and Northeast India."),
            ("num", "2.\t**Calibrated inference** (KA 1×, 4 members, probability matching), no retraining."),
            ("num", "3.\t**Verification**: tuned on 2022, tested on 2023, PySTEPS on the same sequences."),
            ("num", "4.\t**Open-source framework and web application** for any region.")])
    p.gap(0.2)
    p.text([("h", "II. Literature Review"),
            ("bullet", "**Optical flow** (PySTEPS [1]): advection with stochastic noise."),
            ("bullet", "**Deterministic DL** (ConvLSTM [2], U-Net): MSE loss blurs storm cores."),
            ("bullet", "**Generative** (DGMR [5], DiffCast [6], PreDiff [3]): sharp samples; PreDiff adds "
                       "latent diffusion and KA on Earthformer [4].")])
    p.gap(0.2)
    p.text([("cap", "Table I. Data and compute used")], gap=0.1)
    p.table([["Item", "South India", "Northeast India"],
             ["Source", "IMERG Final V07 [7]", "0.1°, 30 min, 128 × 128"],
             ["Training archive", "35.6 k frames (2000-21)", "84.3 k frames (2000-21)"],
             ["Training sequences", "15.7 k (top 10 % + 5 % random)", "21.3 k (same rule)"],
             ["Training compute", "1 shared 32 GB GPU, 50 epochs", "1 shared 32 GB GPU"],
             ["Validation (unseen)", "JJAS 2022 tune, 2023 test", "JJAS 2022 tune, 2023 test"],
             ["Inference", "4 members × 20 steps, fp16", "4 GB laptop GPU, ≈ 40 s"]],
            [10.6, 13.2, 13.2])
    p.gap(0.2)
    p.text([("h", "References")], gap=0.0)
    yref = p.y
    half = (p.w - 0.4) / 2
    p.text([("ref", "[1] S. Pulkkinen et al., “Pysteps,” _GMD_, 2019."),
            ("ref", "[2] X. Shi et al., “ConvLSTM,” _NeurIPS_, 2015."),
            ("ref", "[3] Z. Gao et al., “PreDiff,” _NeurIPS_, 2023."),
            ("ref", "[4] Z. Gao et al., “Earthformer,” _NeurIPS_, 2022.")], w=half)
    yl = p.y
    p.y = yref
    p.text([("ref", "[5] S. Ravuri et al., “DGMR,” _Nature_, 2021."),
            ("ref", "[6] D. Yu et al., “DiffCast,” _CVPR_, 2024."),
            ("ref", "[7] G. J. Huffman et al., “IMERG V07,” GES DISC, 2023."),
            ("ref", "[8] E. E. Ebert, “Ensemble-based QPF,” _MWR_, 2001.")], w=half, x=p.x + half + 0.4)
    p.y = max(p.y, yl)
    p.check("TL")

    # ---------- BL: III ----------
    p = Panel(s, "BL")
    p.text([("h", "III. Problem Statement"),
            ("body", "Given **7 half-hourly** satellite rainfall frames (210 min) on a 128 × 128 grid (≈ 1400 km), "
                     "forecast the **next 6 frames** (180 min) sharply and reliably, using only satellite data.")])
    w1 = 11.4
    x1 = p.x + (p.w - 2 * w1 - 1.0) / 2
    h1 = p.picture(transparent("fig1a_northeast.png"), w=w1, x=x1, advance=False)
    h2 = p.picture(transparent("fig1b_south.png"), w=w1, x=x1 + w1 + 1.0, advance=False)
    p.y += max(h1, h2) + 0.15
    p.text([("cap", "Fig. 1. Domains: (a) NE India 19.1-31.9°N, 86.1-98.9°E; (b) S. India 5-17.8°N, 71-83.8°E.")])
    p.picture(fig("results/fig_case_south_compact.png"), w=37)
    p.text([("capl", "Fig. 2. Test case with the most rain at +90 min, 30 Jun 2023 (chosen automatically). "
                     "INDRA-Sat-Diff retains the coastal convective cells, which the PySTEPS ensemble mean smooths out.")])
    qr = 5.0
    ystrip = p.bottom - qr - 0.6
    p.y = max(p.y + 0.2, ystrip)
    p.slide.shapes.add_picture(fig("qr_github.png"), Cm(p.x + 0.3), Cm(p.y), Cm(qr), Cm(qr))
    p.text([("small", "**Code, app and results:** github.com/vishwajitsarnobat/INDRA-Sat-Diff"),
            ("small", "**Acknowledgment:** We thank Dr. Prashant Kumar (SAC/ISRO) for his guidance, Shri Nilesh M. "
                      "Desai, Director SAC/ISRO, for this opportunity, and SPIT Mumbai for academic support.")],
           w=p.w - qr - 0.8, x=p.x + qr + 0.8)
    p.check("BL")

    # ---------- TR: IV ----------
    p = Panel(s, "TR")
    p.text([("h", "IV. Methodology")])
    p.picture(fig("method_pipeline.png"), w=36)
    p.text([("cap", "Fig. 3. INDRA-Sat-Diff pipeline, shown on a real case (South India, 30 Jun 2023, 09 UTC).")])
    p.text([("sub", "A. Two-stage latent diffusion"),
            ("bullet", "**VAE** compresses each 128×128 frame to a 64×16×16 latent."),
            ("bullet", "**Earthformer-UNet** (cuboid attention + skip links) denoises the 6-frame latent forecast, "
                       "conditioned on the 7 input latents."),
            ("bullet", "**VAE decoder** returns rain fields; each run with new noise is one ensemble member.")])
    p.text([("sub", "B. Knowledge Alignment (KA)")], gap=0.1)
    p.picture(transparent("fig5_ka_inference.png"), w=18.5)
    p.text([("cap", "Fig. 4. KA: denoising × physics alignment term (after PreDiff [3]).")])
    p.text([("body", "Each sampling step is steered toward the target mean intensity **F₀**, without retraining. "
                     "We set **F₀** to the mean input intensity (1×). The earlier evaluation used 2×, which "
                     "increased false alarms.")])
    p.text([("sub", "C. Calibrated inference"),
            ("bullet", "**4 members**, 20 DDIM steps each, fp16."),
            ("bullet", "**Probability matching** [8]: cells ranked by ensemble mean receive the rain rates "
                       "observed at T0, keeping the observed rain area and intensities.")])
    p.check("TR")

    # ---------- BR: V, VI, refs ----------
    p = Panel(s, "BR")
    p.text([("h", "V. Results (JJAS 2023 test data, ≥ 1 mm/h)")])
    p.picture(fig("results/fig_validation_fix.png"), w=37)
    p.text([("capl", "Fig. 5. (a, c) CSI vs lead time; (b, d) CSI minus persistence (above 0: better than "
                     "persistence; shaded: 95 % CI). 75 test sequences per region, settings chosen on 2022. PySTEPS "
                     "(50 members) loses skill more slowly than persistence, so its gain rises with lead time.")])
    p.gap(0.2)
    p.text([("h", "VI. Conclusions & Future Scope")] + [("bullet", b) for b in RESULT_BULLETS])
    p.check("BR")

    prs.save(out)
    print("saved", out)


if __name__ == "__main__":
    template = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "InGARSS2026_Poster_Template_2.pptx")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "INDRA-Sat-Diff_InGARSS2026_poster.pptx")
    build(template, out)
