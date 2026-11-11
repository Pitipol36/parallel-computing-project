"""Charts for the GPU BVH construction experiments, from results/build*.csv,
results/traversal.csv and results/dynamic.csv, into results/plots/.

    py scripts/plot_build.py
"""
import csv
import glob
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "plots")

COLORS = {"cpu_sah": "#52514e", "cpu_lbvh": "#eb6834", "gpu_lbvh": "#2a78d6", "cpu_ploc": "#7a5bd6",
          "gpu_ploc": "#1baf7a", "cpu_trbvh": "#8a8984", "gpu_trbvh": "#d6457a", "gpu_refit": "#eda100",
          "cpu_sah_rebuild": "#52514e", "gpu_lbvh_rebuild": "#2a78d6", "gpu_ploc_rebuild": "#1baf7a",
          "gpu_trbvh_rebuild": "#d6457a"}
STAGE_COLORS = {"bounds": "#1baf7a", "morton": "#eda100", "sort": "#eb6834", "hierarchy": "#2a78d6",
                "fit": "#7a5bd6", "cluster": "#d6457a", "restructure": "#10172a", "convert": "#8a8984"}
STAGES = ["bounds", "morton", "sort", "hierarchy", "fit", "cluster", "restructure", "convert"]
LABELS = {"cpu_sah": "CPU binned SAH", "cpu_lbvh": "CPU LBVH (1 thread)", "gpu_lbvh": "GPU LBVH",
          "cpu_ploc": "CPU PLOC (1 thread)", "gpu_ploc": "GPU PLOC",
          "cpu_trbvh": "CPU TRBVH (1 thread)", "gpu_trbvh": "GPU TRBVH", "gpu_trbvh_rebuild": "GPU TRBVH rebuild",
          "cpu_sah_rebuild": "CPU SAH rebuild", "gpu_lbvh_rebuild": "GPU LBVH rebuild",
          "gpu_ploc_rebuild": "GPU PLOC rebuild", "gpu_refit": "GPU refit (frame-0 topology)"}
TREE_LABELS = {"cpu_sah": "SAH tree", "gpu_lbvh": "LBVH tree", "gpu_ploc": "PLOC tree", "gpu_trbvh": "TRBVH tree"}
BUILDERS = ["cpu_sah", "cpu_lbvh", "cpu_ploc", "cpu_trbvh", "gpu_lbvh", "gpu_ploc", "gpu_trbvh"]
MODEL_ORDER = ["bunny", "dragon", "buddha", "lucy"]
MODEL_MARKERS = {"bunny": "o", "dragon": "s", "buddha": "D", "lucy": "^"}
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "lines.linewidth": 2, "lines.markersize": 7,
    "legend.frameon": False,
})


def read(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if v not in ("", None) else None
            except ValueError:
                pass
    return rows


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def models_in(rows):
    present = {r["model"] for r in rows}
    return [m for m in MODEL_ORDER if m in present]


def tick(model, rows):
    tris = next(int(r["tris"]) for r in rows if r["model"] == model)
    return f"{model}\n{tris / 1e3:,.0f}K tris" if tris < 1e6 else f"{model}\n{tris / 1e6:.2f}M tris"


def fmt_ms(v):
    return f"{v:,.1f} ms" if v < 100 else f"{v:,.0f} ms"


# ---------------------------------------------------------------- build time vs size

def build_time_plot(build):
    rows = [r for r in build if r["experiment"] == "size"]
    if not rows:
        return
    models = models_in(rows)
    by = {(r["model"], r["builder"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for b in BUILDERS:
        pts = [(i, by[(m, b)]["build_ms"]) for i, m in enumerate(models) if (m, b) in by]
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=COLORS[b], label=LABELS[b],
                linestyle="--" if b.startswith("cpu_") and b != "cpu_sah" else "-")
        if b in ("cpu_sah", "gpu_lbvh", "gpu_ploc", "gpu_trbvh"):
            for i, v in pts:
                ax.annotate(fmt_ms(v), (i, v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9,
                            color=COLORS[b])
    last = models[-1]
    lines = []
    for b in ("gpu_lbvh", "gpu_ploc", "gpu_trbvh"):
        if (last, b) in by and (last, "cpu_sah") in by:
            lines.append(f"{LABELS[b]}: {by[(last, 'cpu_sah')]['build_ms'] / by[(last, b)]['build_ms']:.0f}x faster "
                         f"than CPU SAH")
    ax.text(0.99, 0.04, f"{last}:\n" + "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
            color=TEXT2)
    ax.set_title("BVH build time vs mesh size", loc="left")
    ax.set_yscale("log")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([tick(m, rows) for m in models])
    ax.set_xlim(-0.3, len(models) - 0.4)
    ax.set_ylabel("build time (ms, log scale)")
    ax.set_ylim(top=ax.get_ylim()[1] * 2.5)
    ax.legend(loc="center right", fontsize=9.5)
    save(fig, "build_time_vs_size.png")


# ---------------------------------------------------------------- stage breakdown

def stage_plot(build):
    rows = [r for r in build if r["experiment"] == "size"]
    if not rows:
        return
    models = models_in(rows)
    panels = [b for b in ("gpu_lbvh", "gpu_ploc", "gpu_trbvh", "cpu_lbvh", "cpu_ploc", "cpu_trbvh")
              if any(r["builder"] == b for r in rows)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for ax, b in zip(axes.flat, panels):
        by = {r["model"]: r for r in rows if r["builder"] == b}
        bottom = [0.0] * len(models)
        for s in STAGES:
            vals = [(by[m].get(f"{s}_ms") or 0.0) if m in by else 0.0 for m in models]
            if not any(vals):
                continue
            ax.bar(range(len(models)), vals, bottom=bottom, color=STAGE_COLORS[s], label=s, width=0.6,
                   edgecolor=SURFACE, linewidth=1)
            bottom = [a + v for a, v in zip(bottom, vals)]
        for i, tot in enumerate(bottom):
            ax.annotate(fmt_ms(tot), (i, tot), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9,
                        color=TEXT2)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([tick(m, rows) for m in models], fontsize=9.5)
        ax.set_ylabel("time (ms)")
        ax.set_title(LABELS[b] + ": time per stage", loc="left", fontsize=12)
        ax.set_ylim(top=max(bottom) * 1.18)
        ax.legend(loc="upper left", ncol=2, fontsize=8.5)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    save(fig, "build_stages.png")


# ---------------------------------------------------------------- quality trade-off

def quality_plot(build, render, trav):
    rows = [r for r in build if r["experiment"] == "size"]
    if not rows:
        return
    models = models_in(rows)
    trees = [b for b in ("cpu_sah", "gpu_lbvh", "gpu_ploc", "gpu_trbvh") if any(r["builder"] == b for r in rows)]
    by = {(r["model"], r["builder"]): r for r in rows}
    rend = {(r["model"], r["experiment"].replace("quality_", ""), r["mode"]): r for r in render
            if str(r["experiment"]).startswith("quality_")}
    tv = {(r["model"], r["builder"]): r for r in trav}
    panels = [("SAH cost (% of CPU SAH)", lambda m, b: 100 * by[(m, b)]["sah_cost"] / by[(m, "cpu_sah")]["sah_cost"]
               if (m, b) in by and (m, "cpu_sah") in by else None)]
    if tv:
        panels.append(("box tests per ray", lambda m, b: tv[(m, b)]["box_tests_per_ray"] if (m, b) in tv else None))
        panels.append(("triangle tests per ray", lambda m, b: tv[(m, b)]["tri_tests_per_ray"] if (m, b) in tv else None))
    if rend:
        panels.append(("gpu_bvh_opt kernel time (ms)",
                       lambda m, b: rend[(m, b, "gpu_bvh_opt")]["kernel_ms"] if (m, b, "gpu_bvh_opt") in rend else None))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.4), squeeze=False)
    w = 0.8 / len(trees)
    for ax, (title, fn) in zip(axes[0], panels):
        for j, b in enumerate(trees):
            vals = [fn(m, b) for m in models]
            xs = [i + (j - (len(trees) - 1) / 2) * w for i in range(len(models))]
            keep = [(x, v) for x, v in zip(xs, vals) if v is not None]
            bars = ax.bar([k[0] for k in keep], [k[1] for k in keep], width=w, color=COLORS[b], label=TREE_LABELS[b],
                          edgecolor=SURFACE, linewidth=1)
            for bar, (_, v) in zip(bars, keep):
                ax.annotate(f"{v:.0f}" if v >= 10 else f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5, color=TEXT2)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models)
        ax.set_title(title, loc="left", fontsize=11)
        ax.margins(y=0.15)
    axes[0][0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Tree quality by builder (1080p, 4 AO)", x=0.01, ha="left", fontweight="bold")
    save(fig, "build_quality.png")


# ---------------------------------------------------------------- PLOC radius sweep

def radius_plot(build):
    rows = sorted([r for r in build if r["experiment"] == "radius"], key=lambda r: r["ploc_radius"])
    sah = {r["model"]: r for r in build if r["experiment"] == "size" and r["builder"] == "cpu_sah"}
    lb = {r["model"]: r for r in build if r["experiment"] == "size" and r["builder"] == "gpu_lbvh"}
    if not rows:
        return
    model = rows[0]["model"]
    ref = sah[model]["sah_cost"] if model in sah else None
    xs = list(range(len(rows)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(xs, [r["build_ms"] for r in rows], marker="o", color=COLORS["gpu_ploc"], label="GPU PLOC")
    for x, r in zip(xs, rows):
        axes[0].annotate(f"{r['build_ms']:.1f}", (x, r["build_ms"]), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=9, color=TEXT2)
    if model in lb:
        axes[0].axhline(lb[model]["build_ms"], color=COLORS["gpu_lbvh"], linestyle="--", linewidth=1.5,
                        label=f"GPU LBVH ({lb[model]['build_ms']:.1f} ms)")
    axes[0].set_ylabel("build time (ms)")
    axes[0].set_title("Build time", loc="left")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="upper left", fontsize=9)
    if ref:
        ys = [100 * r["sah_cost"] / ref for r in rows]
        axes[1].plot(xs, ys, marker="o", color=COLORS["gpu_ploc"], label="GPU PLOC")
        for x, y in zip(xs, ys):
            axes[1].annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9,
                             color=TEXT2)
        if model in lb:
            axes[1].axhline(100 * lb[model]["sah_cost"] / ref, color=COLORS["gpu_lbvh"], linestyle="--",
                            linewidth=1.5, label="GPU LBVH")
        axes[1].axhline(100, color=COLORS["cpu_sah"], linestyle=":", linewidth=1.5, label="CPU SAH = 100%")
        axes[1].set_ylabel("SAH cost (% of CPU SAH)")
        axes[1].set_title("Tree quality", loc="left")
        axes[1].legend(loc="upper right", fontsize=9)
        axes[1].margins(y=0.25)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{int(r['ploc_radius'])}" for r in rows])
        ax.set_xlabel("PLOC search radius r (clusters on each side)")
    fig.suptitle(f"PLOC radius sweep ({model})", x=0.01, ha="left", fontweight="bold")
    save(fig, "ploc_radius.png")


# ---------------------------------------------------------------- Pareto: build time vs quality

def pareto_plot(build):
    rows = [r for r in build if r["experiment"] == "size"]
    if not rows:
        return
    sah = {r["model"]: r["sah_cost"] for r in rows if r["builder"] == "cpu_sah"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for b in BUILDERS:
        pts = [(r["build_ms"], 100 * r["sah_cost"] / sah[r["model"]], r["model"]) for r in rows
               if r["builder"] == b and r["model"] in sah]
        for i, (x, y, m) in enumerate(pts):
            ax.scatter([x], [y], color=COLORS[b], marker=MODEL_MARKERS.get(m, "o"), s=60, zorder=3,
                       label=LABELS[b] if i == 0 else None, edgecolor=SURFACE, linewidth=0.8)
        big = [p for p in pts if p[2] == "dragon"]
        if big:
            ax.annotate(LABELS[b], (big[0][0], big[0][1]), textcoords="offset points", xytext=(8, 6), fontsize=9,
                        color=COLORS[b])
    for m, mk in MODEL_MARKERS.items():
        if m in sah:
            ax.scatter([], [], color=TEXT2, marker=mk, label=m)
    ax.set_xscale("log")
    ax.set_xlabel("build time (ms, log scale)")
    ax.set_ylabel("SAH cost (% of CPU SAH)")
    ax.set_title("Build speed vs tree quality (lower left is better)", loc="left")
    ax.legend(loc="lower left", fontsize=8.5, ncol=2)
    ax.set_ylim(bottom=ax.get_ylim()[0] - 12)
    save(fig, "pareto.png")


# ---------------------------------------------------------------- dynamic scene

def dynamic_plots(dyn):
    if not dyn:
        return
    model = dyn[0]["model"]
    series = defaultdict(list)
    for r in dyn:
        series[r["variant"]].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for v, rs in series.items():
        rs.sort(key=lambda r: r["frame"])
        f = [r["frame"] for r in rs]
        axes[0].plot(f, [r["sah_cost"] for r in rs], color=COLORS[v], label=LABELS[v])
        axes[1].plot(f, [r["total_ms"] for r in rs], color=COLORS[v], label=LABELS[v])
    for ax, yl, t in ((axes[0], "SAH cost (log scale)", "Tree quality per frame"),
                      (axes[1], "update + render (ms, log scale)", "Time per frame")):
        ax.set_yscale("log")
        ax.set_xlabel("frame")
        ax.set_ylabel(yl)
        ax.set_title(t, loc="left")
    axes[1].legend(loc="upper left", fontsize=9)
    fig.suptitle(f"Exploding {model}: refit vs rebuild under incoherent motion", x=0.01, ha="left", fontweight="bold")
    save(fig, "dynamic_refit_vs_rebuild.png")


def explode_gif():
    frames = sorted(glob.glob(os.path.join(ROOT, "renders", "explode_*.png")))
    if len(frames) < 2:
        return
    try:
        from PIL import Image
    except ImportError:
        return
    imgs = []
    for p in frames:
        im = Image.open(p).convert("RGB")
        im.thumbnail((640, 360))
        imgs.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
    path = os.path.join(OUT, "explode.gif")
    imgs[0].save(path, save_all=True, append_images=imgs[1:] + [imgs[-1]] * 8, duration=60, loop=0, optimize=True)
    print("wrote", os.path.relpath(path, ROOT))


# ---------------------------------------------------------------- time to image vs rays per build

def time_to_image_plot(build, render):
    sweep = [r for r in render if str(r["experiment"]).startswith("rays_") and r["mode"] == "gpu_bvh_opt"]
    size = {(r["model"], r["builder"]): r for r in build if r["experiment"] == "size"}
    if not sweep:
        return
    model = sweep[0]["model"]
    fits = {}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for b in ("cpu_sah", "gpu_lbvh", "gpu_ploc", "gpu_trbvh"):
        pts = sorted(((r["width"] * r["height"] * (2 + r["ao"]), r) for r in sweep if r["experiment"] == f"rays_{b}"),
                     key=lambda p: p[0])
        if len(pts) < 2 or (model, b) not in size:
            continue
        # Least-squares line: render time = a + c * rays (per frame).
        xs = [p[0] for p in pts]
        ys = [p[1]["e2e_ms"] if b == "cpu_sah" else p[1]["kernel_ms"] + p[1]["download_ms"] for p in pts]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        c = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        a = my - c * mx
        s = size[(model, b)]
        # cpu_sah: CPU build, then render_gpu uploads the tree (inside e2e).
        # GPU builders: mesh upload + GPU build; the tree is already resident.
        fixed = s["build_ms"] + (s["upload_ms"] if b != "cpu_sah" else 0.0)
        fits[b] = (fixed + a, c)
        rays = [1e6 * 10 ** (k / 20) for k in range(0, 81)]  # 1M .. 10G rays per build
        ax.plot(rays, [(fixed + a + c * r) / 1e3 for r in rays], color=COLORS[b], label=TREE_LABELS[b])
        ax.scatter(xs, [(fixed + y) / 1e3 for y in ys], color=COLORS[b], s=22, zorder=3)
    notes = []
    for b1, b2 in (("gpu_lbvh", "cpu_sah"), ("gpu_trbvh", "cpu_sah")):
        if b1 in fits and b2 in fits:
            (f1, c1), (f2, c2) = fits[b1], fits[b2]
            if c1 > c2 and f2 > f1:
                rstar = (f2 - f1) / (c1 - c2)
                notes.append((b1, b2, rstar))
    for k, (b1, b2, rstar) in enumerate(notes):
        ax.axvline(rstar, color=COLORS[b2], linestyle=":", linewidth=1.2)
        ax.annotate(f"{TREE_LABELS[b2]} overtakes {TREE_LABELS[b1]}\nat {rstar / 1e9:.2g}G rays "
                    f"(~{rstar / (1920 * 1080 * 6):.0f} frames, 1080p 4 AO)", (rstar, 0.02 * 3 ** k),
                    textcoords="offset points", xytext=(-6, 0), ha="right", fontsize=8.5, color=TEXT2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rays traced with one tree (log scale)")
    ax.set_ylabel("time to image: build + transfers + render (s)")
    ax.set_title(f"Time to image vs rays per build ({model})", loc="left")
    ax.legend(loc="upper left")
    save(fig, "time_to_image.png")


def trbvh_plot(build):
    rows = sorted([r for r in build if r["experiment"] == "trbvh_passes"], key=lambda r: r["trbvh_passes"])
    sah = {r["model"]: r["sah_cost"] for r in build if r["experiment"] == "size" and r["builder"] == "cpu_sah"}
    if not rows or rows[0]["model"] not in sah:
        return
    model = rows[0]["model"]
    ref = sah[model]
    extra = {r["experiment"]: r for r in build if r["experiment"] in ("trbvh_treelet", "trbvh_thread")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = [int(r["trbvh_passes"]) for r in rows]
    axes[0].plot(xs, [r["build_ms"] for r in rows], marker="o", color=COLORS["gpu_trbvh"], label="GPU TRBVH, 7-leaf, warp")
    for x, r in zip(xs, rows):
        axes[0].annotate(f"{r['build_ms']:.1f}", (x, r["build_ms"]), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=9, color=TEXT2)
    ys = [100 * r["sah_cost"] / ref for r in rows]
    axes[1].plot(xs, ys, marker="o", color=COLORS["gpu_trbvh"], label="GPU TRBVH, 7-leaf treelets")
    for x, y in zip(xs, ys):
        axes[1].annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9,
                         color=TEXT2)
    if "trbvh_treelet" in extra:
        r = extra["trbvh_treelet"]
        p = int(r["trbvh_passes"])
        axes[0].scatter([p], [r["build_ms"]], marker="s", color=COLORS["gpu_ploc"], zorder=3, label="5-leaf treelets")
        axes[1].scatter([p], [100 * r["sah_cost"] / ref], marker="s", color=COLORS["gpu_ploc"], zorder=3,
                        label="5-leaf treelets")
    if "trbvh_thread" in extra:
        r = extra["trbvh_thread"]
        axes[0].scatter([int(r["trbvh_passes"])], [r["build_ms"]], marker="D", color=COLORS["cpu_sah"], zorder=3,
                        label=f"one thread per treelet ({r['build_ms']:.0f} ms)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("build time (ms, log scale)")
    axes[0].set_title("Build time", loc="left")
    axes[1].axhline(100, color=COLORS["cpu_sah"], linestyle=":", linewidth=1.5, label="CPU SAH = 100%")
    axes[1].set_ylabel("SAH cost (% of CPU SAH)")
    axes[1].set_title("Tree quality", loc="left")
    axes[1].margins(y=0.2)
    axes[0].legend(loc="lower right", fontsize=8.5)
    axes[1].legend(loc="upper right", fontsize=8.5)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xlabel("restructuring passes (0 = plain LBVH)")
    fig.suptitle(f"TRBVH: passes, treelet size and warp cooperation ({model})", x=0.01, ha="left", fontweight="bold")
    save(fig, "trbvh_passes.png")


def main():
    build, render = read("build.csv"), read("build_render.csv")
    trav, dyn = read("traversal.csv"), read("dynamic.csv")
    build_time_plot(build)
    stage_plot(build)
    quality_plot(build, render, trav)
    radius_plot(build)
    trbvh_plot(build)
    pareto_plot(build)
    time_to_image_plot(build, render)
    dynamic_plots(dyn)
    explode_gif()


if __name__ == "__main__":
    main()
