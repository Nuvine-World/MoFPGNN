import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

TEX = "main.tex"
OUT = "figures"
N_SEEDS = 10

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
C_VS_GRAPH, C_VS_FP = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False,
})

METRICS = [("R2", r"$R^2$"), ("EF", "EF@5%"), ("NDCG", "NDCG@5%")]
METRIC_KEYS = {"R^2": "R2", "RMSE": "RMSE", "EF@5": "EF", "NDCG@5": "NDCG",
               "HitRate@5": "HitRate"}


def _num(field):
    return float(field.strip().strip("$").strip())


def parse_table(label):
    """Return {(baseline, hybrid, metric): (delta, t, ns)} for one table.

    baseline is "graph" for the upper panel and "fp" for the lower one;
    hybrid is "CMF" or "BMF" (left and right column groups).
    """
    tex = open(TEX).read()
    at = tex.index(r"\label{" + label + "}")
    block = tex[tex.rfind(r"\begin{table}", 0, at): tex.index(r"\end{table}", at)]
    out, panel = {}, None
    for line in block.split("\n"):
        if r"\emph{vs." in line:
            panel = "fp" if "fingerprint branch" in line else "graph"
            continue
        cells = line.rstrip().rstrip("\\").split("&")
        if len(cells) != 7 or panel is None:
            continue
        metric = next((v for k, v in METRIC_KEYS.items() if k in cells[0]), None)
        if metric is None:
            continue
        for hyb, (d, t, p) in (("CMF", cells[1:4]), ("BMF", cells[4:7])):
            out[(panel, hyb, metric)] = (_num(d), _num(t), "n.s." in p)
    assert len(out) == 20, f"{label}: expected 20 cells, parsed {len(out)}"
    return out


def eye(ax, y, delta, t, color, half_height, ns):
    if t == 0:
        # Delta = 0, t = 0: no standard error to draw, so show the point alone.
        ax.plot(delta, y, "o", ms=4.2, color=color, mec=SURFACE, mew=0.8, zorder=4)
        ax.text(delta, y + half_height * 0.9, " n.s.", fontsize=6.8, color=MUTED,
                va="center", ha="left")
        return
    se = delta / t
    tcrit = stats.t.ppf(0.975, N_SEEDS - 1)
    lo, hi = sorted((delta - tcrit * se, delta + tcrit * se))
    x = np.linspace(delta - 4 * abs(se), delta + 4 * abs(se), 300)
    h = half_height * stats.t.pdf((x - delta) / abs(se), N_SEEDS - 1) \
        / stats.t.pdf(0, N_SEEDS - 1)
    ax.fill_between(x, y - h, y + h, color=color, alpha=0.22, lw=0, zorder=2)
    ax.plot(x, y + h, color=color, lw=0.6, zorder=2)
    ax.plot(x, y - h, color=color, lw=0.6, zorder=2)
    ax.hlines(y, lo, hi, color=color, lw=1.6, zorder=3)
    ax.plot(delta, y, "o", ms=4.2, color=color, mec=SURFACE, mew=0.8, zorder=4)
    if ns:
        ax.text(hi, y + half_height * 0.9, " n.s.", fontsize=6.8, color=MUTED,
                va="center", ha="left")


def main():
    fams = [("KPGT", parse_table("tab:hypothesis")),
            ("Chemprop", parse_table("tab:hypothesis2"))]

    # one pair of rows per hybrid: upper = vs graph model, lower = vs fingerprint
    pair_gap, fam_gap, offset, half = 1.0, 0.7, 0.2, 0.16
    rows, y = [], 0.0
    for fi, (fam, cells) in enumerate(fams):
        if fi:
            y -= fam_gap
        for hyb in ("CMF", "BMF"):
            rows.append((fam, cells, hyb, y))
            y -= pair_gap

    fig, axes = plt.subplots(1, len(METRICS), figsize=(6.1, 3.6), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    for ax, (mkey, mlabel) in zip(axes, METRICS):
        ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
        for fam, cells, hyb, yc in rows:
            for panel, color, dy in (("graph", C_VS_GRAPH, offset),
                                     ("fp", C_VS_FP, -offset)):
                d, t, ns = cells[(panel, hyb, mkey)]
                eye(ax, yc + dy, d, t, color, half, ns)
        ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_title(mlabel, color=INK)
        ax.set_xlabel(r"$\Delta$ (hybrid $-$ baseline)")
        sep = rows[1][3] - (pair_gap + fam_gap) / 2
        ax.axhline(sep, color=GRID, lw=0.8)

    axes[0].set_yticks([r[3] for r in rows])
    axes[0].set_yticklabels(
        [f"{fam} + {hyb}" for fam, _, hyb, _ in rows], color=INK)
    trans = mtransforms.blended_transform_factory(axes[0].transAxes,
                                                  axes[0].transData)
    for fam, first in (("KPGT", rows[0][3]), ("Chemprop", rows[2][3])):
        axes[0].text(-0.02, first + 0.55, fam, transform=trans, ha="right",
                     va="center", fontsize=8, style="italic", color=INK2)
    axes[0].set_ylim(rows[-1][3] - 0.5, rows[0][3] + 0.75)

    handles = [Line2D([], [], color=C_VS_GRAPH, marker="o", lw=1.6, ms=4.2,
                      mec=SURFACE, label="vs. the graph model alone (KPGT or Chemprop)"),
               Line2D([], [], color=C_VS_FP, marker="o", lw=1.6, ms=4.2,
                      mec=SURFACE, label="vs. its own fingerprint branch")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.09))

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_fusion_effects.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}/fig_fusion_effects.pdf and .png")


if __name__ == "__main__":
    main()
