"""Charts for the GPU BVH construction experiments (Goal 1), from results/build*.csv,
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

COLORS = {"cpu_sah": "#52514e", "cpu_lbvh": "#eb6834", "gpu_lbvh": "#2a78d6", "gpu_refit": "#eda100",
          "cpu_sah_rebuild": "#52514e", "gpu_lbvh_rebuild": "#2a78d6"}
STAGE_COLORS = {"bounds": "#1baf7a", "morton": "#eda100", "sort": "#eb6834", "hierarchy": "#2a78d6",
                "fit": "#7a5bd6", "convert": "#8a8984"}
LABELS = {"cpu_sah": "CPU binned SAH", "cpu_lbvh": "CPU LBVH (1 thread)", "gpu_lbvh": "GPU LBVH",
          "cpu_sah_rebuild": "CPU SAH rebuild", "gpu_lbvh_rebuild": "GPU LBVH rebuild",
          "gpu_refit": "GPU refit (frame-0 topology)"}
MODEL_ORDER = ["bunny", "dragon", "buddha", "lucy"]
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


# ---------------------------------------------------------------- 1. build time vs size

def build_time_plot(build):
    rows = [r for r in build if r["experiment"] == "size"]
    if not rows:
        return
    models = models_in(rows)
    by = {(r["model"], r["builder"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for b in ("cpu_sah", "cpu_lbvh", "gpu_lbvh"):
        pts = [(i, by[(m, b)]["build_ms"]) for i, m in enumerate(models) if (m, b) in by]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=COLORS[b], label=LABELS[b])
        for i, v in pts:
            ax.annotate(f"{v:,.1f} ms" if v < 100 else f"{v:,.0f} ms", (i, v), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9, color=COLORS[b])
    last = models[-1]
    if (last, "gpu_lbvh") in by and (last, "cpu_sah") in by:
        g, s, c = by[(last, "gpu_lbvh")]["build_ms"], by[(last, "cpu_sah")]["build_ms"], by[(last, "cpu_lbvh")]["build_ms"]
        ax.text(0.99, 0.04, f"{last}: GPU LBVH is {c / g:.0f}x faster than CPU LBVH\n"
                f"and {s / g:.0f}x faster than CPU SAH",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color=TEXT2)
    ax.set_title("BVH build time vs mesh size", loc="left")
    ax.set_yscale("log")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([tick(m, rows) for m in models])
    ax.set_xlim(-0.3, len(models) - 0.4)
    ax.set_ylabel("build time (ms, log scale)")
    ax.legend(loc="center right")
    ax.set_ylim(top=ax.get_ylim()[1] * 2.5)
    save(fig, "build_time_vs_size.png")


# ---------------------------------------------------------------- 2. GPU stage breakdown

def stage_plot(build):
    rows = [r for r in build if r["experiment"] == "size" and r["builder"] in ("gpu_lbvh", "cpu_lbvh")]
    if not rows:
        return
    models = models_in(rows)
    stages = ["bounds", "morton", "sort", "hierarchy", "fit", "convert"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, b in zip(axes, ("gpu_lbvh", "cpu_lbvh")):
        by = {r["model"]: r for r in rows if r["builder"] == b}
        bottom = [0.0] * len(models)
        for s in stages:
            vals = [by[m][f"{s}_ms"] or 0.0 for m in models]
            ax.bar(range(len(models)), vals, bottom=bottom, color=STAGE_COLORS[s], label=s, width=0.6,
                   edgecolor=SURFACE, linewidth=1)
            bottom = [a + v for a, v in zip(bottom, vals)]
        for i, tot in enumerate(bottom):
            ax.annotate(f"{tot:,.1f} ms" if tot < 100 else f"{tot:,.0f} ms", (i, tot), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=9, color=TEXT2)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([tick(m, rows) for m in models])
        ax.set_ylabel("time (ms)")
        ax.set_title(LABELS[b] + ": time per stage", loc="left")
        ax.set_ylim(top=max(bottom) * 1.15)
    axes[0].legend(loc="upper left", ncol=2, fontsize=9)
    save(fig, "build_stages.png")


# ---------------------------------------------------------------- 3. quality trade-off

def quality_plot(build, render, trav):
    rows = [r for r in build if r["experiment"] == "size"]
    if not rows:
        return
    models = models_in(rows)
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
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2), squeeze=False)
    w = 0.38
    for ax, (title, fn) in zip(axes[0], panels):
        for j, b in enumerate(("cpu_sah", "gpu_lbvh")):
            vals = [fn(m, b) for m in models]
            xs = [i + (j - 0.5) * w for i in range(len(models))]
            bars = ax.bar([x for x, v in zip(xs, vals) if v is not None], [v for v in vals if v is not None],
                          width=w, color=COLORS[b], label="SAH tree" if b == "cpu_sah" else "LBVH tree",
                          edgecolor=SURFACE, linewidth=1)
            for bar, v in zip(bars, [v for v in vals if v is not None]):
                ax.annotate(f"{v:.0f}" if v >= 10 else f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5, color=TEXT2)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models)
        ax.set_title(title, loc="left", fontsize=11)
        ax.margins(y=0.15)
    axes[0][0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Tree quality: LBVH vs SAH (1080p, 4 AO)", x=0.01, ha="left", fontweight="bold")
    save(fig, "build_quality.png")


# ---------------------------------------------------------------- 4. dynamic scene

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
    axes[0].set_yscale("log")
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("SAH cost (log scale)")
    axes[0].set_title("Tree quality per frame", loc="left")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("update + render (ms, log scale)")
    axes[1].set_title("Time per frame", loc="left")
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


# ---------------------------------------------------------------- 5. time-to-image vs rays per build

def time_to_image_plot(build, render):
    sweep = [r for r in render if str(r["experiment"]).startswith("rays_") and r["mode"] == "gpu_bvh_opt"]
    size = {(r["model"], r["builder"]): r for r in build if r["experiment"] == "size"}
    if not sweep:
        return
    model = sweep[0]["model"]
    fits = {}
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for b in ("cpu_sah", "gpu_lbvh"):
        pts = sorted(((r["width"] * r["height"] * (2 + r["ao"]), r) for r in sweep if r["experiment"] == f"rays_{b}"),
                     key=lambda p: p[0])
        if len(pts) < 2 or (model, b) not in size:
            continue
        # Least-squares line: render time = a + c * rays (per frame, kernel + transfers).
        xs = [p[0] for p in pts]
        ys = [p[1]["e2e_ms"] if b == "cpu_sah" else p[1]["kernel_ms"] + p[1]["download_ms"] for p in pts]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        c = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        a = my - c * mx
        s = size[(model, b)]
        # cpu_sah: CPU build, then render_gpu uploads the tree (inside e2e).
        # gpu_lbvh: mesh upload + GPU build; the tree is already resident.
        fixed = s["build_ms"] + (s["upload_ms"] if b == "gpu_lbvh" else 0.0)
        fits[b] = (fixed + a, c)
        rays = [1e6 * 10 ** (k / 20) for k in range(0, 81)]  # 1M .. 10G rays per build
        ax.plot(rays, [(fixed + a + c * r) / 1e3 for r in rays], color=COLORS[b], label=LABELS[b] + " tree")
        ax.scatter(xs, [(fixed + y) / 1e3 for y in ys], color=COLORS[b], s=22, zorder=3)
    if len(fits) == 2:
        (f1, c1), (f2, c2) = fits["cpu_sah"], fits["gpu_lbvh"]
        if c2 > c1:
            rstar = (f1 - f2) / (c2 - c1)
            t = (f1 + c1 * rstar) / 1e3
            ax.axvline(rstar, color=TEXT2, linestyle=":", linewidth=1.2)
            frames1080 = rstar / (1920 * 1080 * 6)
            ax.annotate(f"crossover: {rstar / 1e9:.1f}G rays per build\n"
                        f"(~{frames1080:.0f} frames at 1080p, 4 AO)", (rstar, t), textcoords="offset points",
                        xytext=(-10, -160), ha="right", fontsize=9, color=TEXT2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rays traced with one tree (log scale)")
    ax.set_ylabel("time to image: build + transfers + render (s)")
    ax.set_title(f"Time to image vs rays per build ({model})", loc="left")
    ax.legend(loc="upper left")
    save(fig, "time_to_image.png")


def main():
    build, render = read("build.csv"), read("build_render.csv")
    trav, dyn = read("traversal.csv"), read("dynamic.csv")
    build_time_plot(build)
    stage_plot(build)
    quality_plot(build, render, trav)
    time_to_image_plot(build, render)
    dynamic_plots(dyn)
    explode_gif()


if __name__ == "__main__":
    main()
