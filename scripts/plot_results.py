"""Turns results/results.csv into slide-ready PNG charts in results/plots/.

    py scripts/plot_results.py [results/results.csv]
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "results.csv")
OUT = os.path.join(ROOT, "results", "plots")

# Fixed colour per kernel (categorical slots in fixed order), neutral grey for the CPU.
COLORS = {
    "gpu_bvh_opt": "#2a78d6",
    "gpu_bvh_persistent": "#eb6834",
    "gpu_bvh_naive": "#1baf7a",
    "gpu_brute": "#eda100",
    "cpu": "#8a8984",
}
MARKERS = {"gpu_bvh_opt": "o", "gpu_bvh_persistent": "s", "gpu_bvh_naive": "^", "gpu_brute": "D"}
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "lines.linewidth": 2, "lines.markersize": 8,
    "legend.frameon": False,
})


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for k in ("tris", "width", "height", "ao", "runs"):
                r[k] = int(r[k])
            for k in ("build_ms", "kernel_ms", "e2e_ms", "upload_ms", "download_ms", "mrays_per_s"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def group(rows, key):
    """{key(row): {mode: row}} keeping the last row for duplicates."""
    g = defaultdict(dict)
    for r in rows:
        g[key(r)][r["mode"]] = r
    return g


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def speedup_plot(rows, exp, xkey, xlabel, name, title, xlog=False, xticks=None):
    sel = [r for r in rows if r["experiment"] == exp]
    if not sel:
        return
    g = group(sel, xkey)
    xs = sorted(g)
    pos = {x: i for i, x in enumerate(xs)}  # evenly spaced categories read better than a log axis
    fig, ax = plt.subplots(figsize=(7, 4.2))
    modes = [m for m in COLORS if m != "cpu" and any(m in g[x] for x in xs)]
    for k, m in enumerate(modes):
        pts = [(x, g[x]["cpu"]["kernel_ms"] / g[x][m]["kernel_ms"]) for x in xs if m in g[x] and "cpu" in g[x]]
        if not pts:
            continue
        ax.plot([pos[p[0]] for p in pts], [p[1] for p in pts], color=COLORS[m], marker=MARKERS[m], label=f"{m} (kernel)")
        last = pts[-1]
        ax.annotate(f"{last[1]:.0f}x", (pos[last[0]], last[1]), textcoords="offset points", xytext=(6, 7 if k % 2 == 0 else -7), va="center",
                    color=TEXT2)
    # End-to-end speedup of the main kernel, dashed.
    pts = [(x, g[x]["cpu"]["kernel_ms"] / g[x]["gpu_bvh_opt"]["e2e_ms"]) for x in xs
           if "gpu_bvh_opt" in g[x] and "cpu" in g[x]]
    if pts:
        ax.plot([pos[p[0]] for p in pts], [p[1] for p in pts], color=COLORS["gpu_bvh_opt"], linestyle="--",
                marker="o", markerfacecolor=SURFACE, label="gpu_bvh_opt (end to end)")
    labels = dict(xticks or [])
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([labels.get(x, str(x)) for x in xs])
    ax.set_xlim(-0.3, len(xs) - 0.6)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("speedup over 1-thread CPU (x)")
    ax.set_title(title, loc="left")
    ax.legend(loc="best")
    save(fig, name)


def ablation_plot(rows):
    sel = [r for r in rows if r["experiment"] == "ablation"]
    brute = [r for r in rows if r["experiment"] == "ablation_brute"]
    panels = [p for p in ((sel, "Dragon, 1080p, 4 AO"), (brute, "Bunny, 320x240, 4 AO")) if p[0]]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.2), squeeze=False)
    for ax, (data, title) in zip(axes[0], panels):
        order = [m for m in ("cpu", "gpu_brute", "gpu_bvh_naive", "gpu_bvh_opt", "gpu_bvh_persistent")
                 if any(r["mode"] == m for r in data)]
        byk = {r["mode"]: r for r in data}
        vals = [byk[m]["mrays_per_s"] for m in order]
        bars = ax.bar(range(len(order)), vals, color=[COLORS[m] for m in order], width=0.7,
                      edgecolor=SURFACE, linewidth=2)
        for b, v, m in zip(bars, vals, order):
            rate = f"{v:,.0f}" if v >= 10 else f"{v:.2f}"
            ax.annotate(f"{rate}\n({byk[m]['kernel_ms']:,.1f} ms)", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 4), ha="center", va="bottom", color=TEXT2,
                        fontsize=9)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([m.replace("gpu_", "") for m in order], rotation=0)
        ax.set_yscale("log" if "gpu_brute" in order else "linear")
        ax.set_ylabel("M rays / s (higher is better)")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.25)
    save(fig, "ablation.png")


def block_plot(rows):
    sel = [r for r in rows if r["experiment"] == "block"]
    if not sel:
        return
    blocks = []
    for r in sel:
        if r["block"] not in blocks:
            blocks.append(r["block"])
    g = group(sel, lambda r: r["block"])
    modes = [m for m in ("gpu_bvh_naive", "gpu_bvh_opt", "gpu_bvh_persistent") if any(m in g[b] for b in blocks)]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.8 / len(modes)
    for i, m in enumerate(modes):
        xs = [j + (i - (len(modes) - 1) / 2) * w for j in range(len(blocks))]
        ax.bar(xs, [g[b][m]["kernel_ms"] if m in g[b] else 0 for b in blocks], width=w, color=COLORS[m],
               label=m, edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(range(len(blocks)))
    ax.set_xticklabels(blocks)
    ax.set_xlabel("block size (threads)")
    ax.set_ylabel("kernel time, ms (lower is better)")
    ax.set_title("Block-size tuning, dragon 1080p, 4 AO", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend()
    save(fig, "block_size.png")


def breakdown_plot(rows):
    sel = [r for r in rows if r["experiment"] == "size" and r["mode"] == "gpu_bvh_opt"]
    if not sel:
        return
    sel.sort(key=lambda r: r["tris"])
    labels = [f"{r['model']}\n{r['tris'] / 1e6:.2f}M tris" for r in sel]
    parts = [("BVH build (CPU)", "build_ms", "#8a8984"), ("upload", "upload_ms", "#6250d6"),
             ("kernel", "kernel_ms", "#2a78d6"), ("download", "download_ms", "#1baf7a")]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    left = [0.0] * len(sel)
    for name, key, color in parts:
        vals = [r[key] for r in sel]
        ax.barh(range(len(sel)), vals, left=left, color=color, label=name, edgecolor=SURFACE, linewidth=2,
                height=0.6)
        left = [a + b for a, b in zip(left, vals)]
    for i, total in enumerate(left):
        ax.annotate(f"{total:,.0f} ms", (total, i), textcoords="offset points", xytext=(6, 0), va="center",
                    color=TEXT2)
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("time, ms")
    ax.set_title("Where the time goes (gpu_bvh_opt, 1080p, 4 AO)", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncol=2)
    ax.margins(x=0.15)
    save(fig, "time_breakdown.png")


def summary_table(rows):
    print("\n%-14s %-8s %10s %-20s %10s %10s %9s" % ("experiment", "model", "tris", "mode", "kernel ms",
                                                    "e2e ms", "speedup"))
    g = group(rows, lambda r: (r["experiment"], r["model"], r["width"], r["height"], r["ao"], r["block"]))
    cpu_by = {(r["experiment"], r["model"], r["width"], r["height"], r["ao"]): r["kernel_ms"]
              for r in rows if r["mode"] == "cpu"}
    for r in rows:
        cpu = cpu_by.get((r["experiment"], r["model"], r["width"], r["height"], r["ao"]))
        sp = f"{cpu / r['kernel_ms']:.1f}x" if cpu and r["mode"] != "cpu" else ""
        print("%-14s %-8s %10d %-20s %10.2f %10.2f %9s" % (r["experiment"], r["model"], r["tris"], r["mode"],
                                                           r["kernel_ms"], r["e2e_ms"], sp))
    del g


def main():
    rows = load(CSV)
    speedup_plot(rows, "size", lambda r: r["tris"], "mesh (triangles)", "speedup_vs_size.png",
                 "Speedup vs mesh size (1080p, 4 AO)", xlog=True,
                 xticks=sorted({(r["tris"], f"{r['model']}\n{r['tris'] / 1e6:.2f}M") for r in rows
                                if r["experiment"] == "size"}))
    speedup_plot(rows, "ao", lambda r: r["ao"] + 2, "rays per pixel (primary + shadow + AO)",
                 "speedup_vs_workload.png", "Speedup vs workload (dragon, 1080p)", xlog=True,
                 xticks=[(a + 2, f"{a + 2}\n({a} AO)") for a in sorted({r["ao"] for r in rows
                                                                      if r["experiment"] == "ao"})])
    speedup_plot(rows, "resolution", lambda r: r["width"] * r["height"], "image size",
                 "speedup_vs_resolution.png", "Speedup vs resolution (dragon, 4 AO)", xlog=True,
                 xticks=sorted({(r["width"] * r["height"], f"{r['width']}x{r['height']}") for r in rows
                                if r["experiment"] == "resolution"}))
    ablation_plot(rows)
    block_plot(rows)
    breakdown_plot(rows)
    summary_table(rows)


if __name__ == "__main__":
    main()
