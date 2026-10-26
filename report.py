"""
Consolidated visual report for the 3-stage pipeline.

Produces a self-contained review pack you can open in a browser or attach to a
ticket: reports/review_pack.html. All images are embedded as base64, so the
single HTML file is portable — no folder of loose PNGs to lose.

    python report.py                  # auto-discovers whatever stages have run
    python report.py --open           # also open it in the default browser

Inputs it reads (all optional — missing stages are simply skipped):
    logs/<stage>/summary.json         written by train.py
    logs/<stage>/training_log.csv     per-epoch curves
    logs/<stage>/classification_report.txt   written by evaluate.py
    checkpoints/<stage>/class_mapping.json
    the datasets themselves (class distributions)

Charts written to reports/:
    curves_<stage>.png       per-stage loss / accuracy curves
    comparison.png           best val accuracy per stage
    dataset_<name>.png       class distribution per dataset
"""

import argparse
import base64
import csv
import io
import json
import os
import sys
import webbrowser

import config

STAGE_ORDER = ["scratch", "pretrain", "finetune"]

STAGE_TITLES = {
    "scratch": "Stage 1 — Custom CNN from scratch",
    "pretrain": "Stage 2 — Domain pretraining on PlantVillage",
    "finetune": "Stage 3 — Transfer to PlantDoc field images",
}

STAGE_NOTES = {
    "scratch": "Random initialisation. Proves the custom architecture learns on "
               "clean, controlled lab images.",
    "pretrain": "ImageNet backbone adapted to leaf imagery. This stage exists to "
                "produce features, not a headline number.",
    "finetune": "The headline result: same backbone, new 27-class head, trained "
                "on real field photographs.",
}

PALETTE = ["#2f6f4e", "#c8892a", "#3b6ea5"]


# ─── Collection ──────────────────────────────────────────────────────────────
def load_summary(stage):
    path = os.path.join(config.BASE_DIR, "logs", stage, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_curves(stage):
    path = os.path.join(config.BASE_DIR, "logs", stage, "training_log.csv")
    if not os.path.exists(path):
        return None
    cols = {"epoch": [], "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [], "lr": []}
    with open(path) as f:
        for row in csv.DictReader(f):
            for k in cols:
                try:
                    cols[k].append(float(row[k]))
                except (KeyError, ValueError):
                    cols[k].append(0.0)
    return cols if cols["epoch"] else None


def load_text_report(stage):
    path = os.path.join(config.BASE_DIR, "logs", stage, "classification_report.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read()


def collect():
    stages = []
    for stage in STAGE_ORDER:
        summary = load_summary(stage)
        if not summary:
            continue
        stages.append({
            "stage": stage,
            "title": STAGE_TITLES.get(stage, stage),
            "note": STAGE_NOTES.get(stage, ""),
            "summary": summary,
            "curves": load_curves(stage),
            "report_text": load_text_report(stage),
        })
    return stages


# ─── Charts ──────────────────────────────────────────────────────────────────
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt = _plt()
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def chart_curves(stage_entry, out_dir):
    curves = stage_entry["curves"]
    if not curves:
        return None
    plt = _plt()
    stage = stage_entry["stage"]
    color = PALETTE[STAGE_ORDER.index(stage) % len(PALETTE)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))

    ax1.plot(curves["epoch"], curves["train_loss"], color=color,
             linewidth=2, label="Train")
    ax1.plot(curves["epoch"], curves["val_loss"], color="#c0392b",
             linewidth=2, linestyle="--", label="Validation")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"{stage} — loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(curves["epoch"], [a * 100 for a in curves["train_acc"]],
             color=color, linewidth=2, label="Train")
    ax2.plot(curves["epoch"], [a * 100 for a in curves["val_acc"]],
             color="#c0392b", linewidth=2, linestyle="--", label="Validation")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{stage} — accuracy"); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 100)

    path = os.path.join(out_dir, f"curves_{stage}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    b64 = _b64(fig)
    return {"path": path, "b64": b64}


def chart_comparison(stages, out_dir):
    if not stages:
        return None
    plt = _plt()

    names = [s["stage"] for s in stages]
    accs = [s["summary"]["best_val_acc"] * 100 for s in stages]
    colors = [PALETTE[STAGE_ORDER.index(n) % len(PALETTE)] for n in names]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar(names, accs, color=colors, width=0.55)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 1.5,
                f"{acc:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Accuracy by stage")
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.5, -0.22,
            "Not directly comparable: each stage validates on its own dataset.",
            transform=ax.transAxes, ha="center", fontsize=8, style="italic",
            color="#666")

    path = os.path.join(out_dir, "comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    b64 = _b64(fig)
    return {"path": path, "b64": b64}


def chart_dataset(root, out_dir):
    try:
        import dataset as ds
        from prepare_datasets import _plot_distribution
    except ImportError:
        return None
    try:
        stats = ds.dataset_stats(root)
    except Exception as exc:                              # dataset not present
        print(f"  (skipped dataset chart for {root}: {exc})")
        return None

    name = os.path.basename(str(root).rstrip("/\\"))
    path = _plot_distribution(stats, name, out_dir)
    if not path:
        return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return {"path": path, "b64": b64, "stats": stats}


# ─── Console scorecard ───────────────────────────────────────────────────────
def print_scorecard(stages):
    if not stages:
        print("No completed stages found. Run:  python train.py --stage scratch")
        return

    print("\n" + "=" * 96)
    print("  STAGE SCORECARD")
    print("=" * 96)
    hdr = (f"  {'stage':<10} {'architecture':<18} {'dataset':<22} "
           f"{'classes':>7} {'train':>7} {'val':>6} {'acc':>7}")
    print(hdr)
    print("  " + "-" * 92)
    for s in stages:
        m = s["summary"]
        data = os.path.basename(str(m.get("dataset", "")).rstrip("/\\"))[:22]
        print(f"  {m['stage']:<10} {m.get('architecture', '?'):<18} {data:<22} "
              f"{m['num_classes']:>7} {m['num_train']:>7} {m['num_val']:>6} "
              f"{m['best_val_acc']*100:>6.2f}%")
    print("  " + "-" * 92)
    print("  Note: each stage validates on its own dataset, so the accuracies are")
    print("        not directly comparable. The headline number is the finetune")
    print("        stage's accuracy on PlantDoc's held-out test split.\n")

    # Markdown table, ready to paste into a report
    print("  Markdown:\n")
    print("  | Stage | Architecture | Dataset | Classes | Train | Acc |")
    print("  |---|---|---|---|---|---|")
    for s in stages:
        m = s["summary"]
        data = os.path.basename(str(m.get("dataset", "")).rstrip("/\\"))
        print(f"  | {m['stage']} | {m.get('architecture', '?')} | {data} | "
              f"{m['num_classes']} | {m['num_train']:,} | "
              f"{m['best_val_acc']*100:.2f}% |")
    print()


# ─── HTML review pack ────────────────────────────────────────────────────────
CSS = """
:root { --ink:#1a1a1a; --muted:#6b6b6b; --line:#dcdcdc;
        --green:#2f6f4e; --amber:#c8892a; --blue:#3b6ea5; --bg:#fafaf8; }
* { box-sizing: border-box; }
body { margin:0; padding:36px 32px 72px; background:var(--bg); color:var(--ink);
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:1120px; margin:0 auto; }
h1 { font-size:26px; margin:0 0 4px; letter-spacing:-0.2px; }
.sub { color:var(--muted); font-size:13px; margin-bottom:28px; }
h2 { font-size:17px; margin:38px 0 6px; padding-bottom:6px;
     border-bottom:2px solid var(--line); }
h3 { font-size:14px; margin:20px 0 8px; color:var(--muted);
     text-transform:uppercase; letter-spacing:0.6px; }
.kpis { display:flex; gap:14px; flex-wrap:wrap; margin:18px 0 8px; }
.kpi { flex:1 1 150px; background:#fff; border:1px solid var(--line);
       border-radius:10px; padding:14px 16px; }
.kpi .v { font-size:26px; font-weight:600; letter-spacing:-0.5px; }
.kpi .l { font-size:11px; color:var(--muted); text-transform:uppercase;
          letter-spacing:0.7px; margin-top:2px; }
table { border-collapse:collapse; width:100%; background:#fff; font-size:13px;
        border:1px solid var(--line); border-radius:8px; overflow:hidden; }
th,td { padding:8px 11px; text-align:left; border-bottom:1px solid #eee; }
th { background:#f2f2ef; font-size:11px; text-transform:uppercase;
     letter-spacing:0.6px; color:var(--muted); }
tr:last-child td { border-bottom:none; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.note { font-size:12px; color:var(--muted); font-style:italic; margin:8px 0 0; }
.stage { background:#fff; border:1px solid var(--line); border-radius:10px;
         padding:18px 20px; margin:16px 0; }
.stage h3 { margin-top:0; color:var(--ink); text-transform:none;
            font-size:15px; letter-spacing:0; }
.badge { display:inline-block; font-size:11px; padding:2px 9px; border-radius:20px;
         font-weight:600; letter-spacing:0.3px; }
.b-scratch { background:#e3efe8; color:var(--green); }
.b-pretrain { background:#f8eeda; color:var(--amber); }
.b-finetune { background:#e2eaf5; color:var(--blue); }
img { max-width:100%; border:1px solid var(--line); border-radius:8px;
      margin-top:10px; background:#fff; }
pre { background:#1f2320; color:#e8e8e4; padding:14px 16px; border-radius:8px;
      font-size:11.5px; line-height:1.45; overflow-x:auto; }
footer { margin-top:48px; padding-top:16px; border-top:1px solid var(--line);
         font-size:11px; color:var(--muted); }
"""


def _esc(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def build_html(stages, charts):
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Crop disease pipeline — review pack</title>",
        f"<style>{CSS}</style></head><body><div class='wrap'>",
        "<h1>Crop disease classification — review pack</h1>",
        f"<div class='sub'>Stage: <b>{_esc(config.ACTIVE_STAGE)}</b> config &middot; "
        "generated by report.py</div>",
    ]

    # KPI row
    if stages:
        best = max(stages, key=lambda s: s["summary"]["best_val_acc"])
        total_train = sum(s["summary"]["num_train"] for s in stages)
        total_classes = max(s["summary"]["num_classes"] for s in stages)
        total_time = sum(s["summary"].get("elapsed_min", 0) for s in stages)
        parts.append("<div class='kpis'>")
        for value, label in [
            (f"{len(stages)}", "stages completed"),
            (f"{best['summary']['best_val_acc']*100:.1f}%", f"best acc ({best['stage']})"),
            (f"{total_train:,}", "training images"),
            (f"{total_classes}", "max classes"),
            (f"{total_time:.0f}m", "total compute"),
        ]:
            parts.append(f"<div class='kpi'><div class='v'>{value}</div>"
                         f"<div class='l'>{label}</div></div>")
        parts.append("</div>")

    # Comparison table
    parts.append("<h2>Stage comparison</h2>")
    parts.append("<table><thead><tr><th>Stage</th><th>Architecture</th>"
                 "<th>Dataset</th><th>Layout</th><th>Classes</th>"
                 "<th>Train</th><th>Val</th><th>Best acc</th><th>Time</th>"
                 "</tr></thead><tbody>")
    for s in stages:
        m = s["summary"]
        data = os.path.basename(str(m.get("dataset", "")).rstrip("/\\"))
        parts.append(
            f"<tr><td><span class='badge b-{s['stage']}'>{m['stage']}</span></td>"
            f"<td>{_esc(m.get('architecture', '?'))}</td>"
            f"<td>{_esc(data)}</td>"
            f"<td>{_esc(m.get('layout', '?'))}</td>"
            f"<td class='num'>{m['num_classes']}</td>"
            f"<td class='num'>{m['num_train']:,}</td>"
            f"<td class='num'>{m['num_val']:,}</td>"
            f"<td class='num'><b>{m['best_val_acc']*100:.2f}%</b></td>"
            f"<td class='num'>{m.get('elapsed_min', 0):.0f}m</td></tr>")
    parts.append("</tbody></table>")
    parts.append("<p class='note'>Each stage validates on its own dataset, so "
                 "these accuracies are not directly comparable. The headline "
                 "result is the finetune stage on PlantDoc's held-out split.</p>")

    if charts.get("comparison"):
        parts.append(f"<img src='data:image/png;base64,{charts['comparison']['b64']}' "
                     "alt='accuracy by stage'>")

    # Per-stage detail
    for s in stages:
        m = s["summary"]
        parts.append("<h2>" + _esc(s["title"]) + "</h2>")
        parts.append(f"<p class='note'>{_esc(s['note'])}</p>")
        parts.append("<div class='stage'>")
        parts.append("<h3>Run quantities</h3>")
        rows = [
            ("Architecture", m.get("architecture", "?")
             + (" (ImageNet init)" if m.get("pretrained") else "")),
            ("Warm start", os.path.basename(str(m.get("warm_start") or "-"))),
            ("Dataset", m.get("dataset", "?")),
            ("Classes", m["num_classes"]),
            ("Train / Val", f"{m['num_train']:,} / {m['num_val']:,}"),
            ("Epochs run", f"{m.get('epochs_run', '?')} (best {m.get('best_epoch', '?')})"),
            ("Image size", m.get("img_size", "?")),
            ("Batch size", m.get("batch_size", "?")),
            ("Head / backbone LR",
             f"{m.get('head_lr', '?')} / {m.get('backbone_lr', '?')}"),
            ("Freeze backbone", f"{m.get('freeze_backbone_epochs', 0)} epoch(s)"),
            ("Scheduler", m.get("scheduler", "?")),
            ("Best val loss", f"{m.get('best_val_loss', 0):.4f}"),
            ("Best val acc", f"{m['best_val_acc']*100:.2f}%"),
            ("Wall clock", f"{m.get('elapsed_min', 0):.1f} min"),
        ]
        parts.append("<table><tbody>")
        for k, v in rows:
            parts.append(f"<tr><td style='width:190px;color:#6b6b6b'>{_esc(k)}</td>"
                         f"<td class='num' style='text-align:left'>{_esc(v)}</td></tr>")
        parts.append("</tbody></table>")

        if charts.get(f"curves_{s['stage']}"):
            curve = charts[f"curves_{s['stage']}"]
            parts.append(f"<img src='data:image/png;base64,{curve['b64']}' "
                         "alt='training curves'>")

        if s["report_text"]:
            parts.append("<h3>Classification report</h3>")
            parts.append(f"<pre>{_esc(s['report_text'])}</pre>")
        parts.append("</div>")

    # Dataset sections
    for key, chart in charts.items():
        if not key.startswith("dataset_"):
            continue
        st = chart["stats"]
        parts.append(f"<h2>Dataset — {_esc(os.path.basename(str(st['root']).rstrip('/\\\\')))}</h2>")
        parts.append("<div class='kpis'>")
        for value, label in [
            (f"{st['num_total']:,}", "images"),
            (f"{st['num_classes']}", "classes"),
            (f"{st['num_crops']}", "crops"),
            (f"{st['num_train']:,}", "train"),
            (f"{st['num_val']:,}", "val"),
            (f"{st['imbalance_ratio']:.1f}x", "imbalance"),
        ]:
            parts.append(f"<div class='kpi'><div class='v'>{value}</div>"
                         f"<div class='l'>{label}</div></div>")
        parts.append("</div>")
        parts.append(f"<img src='data:image/png;base64,{chart['b64']}' "
                     "alt='class distribution'>")

    parts.append("<footer>Generated by report.py &middot; "
                 "figures reproduce from logs/ and checkpoints/ &middot; "
                 "single self-contained file</footer>")
    parts.append("</div></body></html>")
    return "\n".join(parts)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Build the visual review pack.")
    p.add_argument("--open", action="store_true", help="Open in a browser")
    p.add_argument("--no-datasets", action="store_true",
                   help="Skip dataset distribution charts (faster)")
    args = p.parse_args()

    out_dir = os.path.join(config.BASE_DIR, "reports")
    os.makedirs(out_dir, exist_ok=True)

    stages = collect()
    print_scorecard(stages)

    charts = {}

    legend = {"comparison": "best accuracy by stage"}
    if stages:
        comp = chart_comparison(stages, out_dir)
        if comp:
            charts["comparison"] = comp
        for s in stages:
            cur = chart_curves(s, out_dir)
            if cur:
                charts[f"curves_{s['stage']}"] = cur

    if not args.no_datasets:
        roots = []
        if stages:
            roots = [s["summary"].get("dataset") for s in stages]
        roots.append(config.DATASET_ROOT)
        seen = set()
        for root in roots:
            if not root or root in seen:
                continue
            seen.add(root)
            chart = chart_dataset(root, out_dir)
            if chart:
                charts[f"dataset_{len(seen)}"] = chart

    html = build_html(stages, charts)
    pack = os.path.join(out_dir, "review_pack.html")
    with open(pack, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Charts written to : {out_dir}")
    for key, chart in charts.items():
        print(f"  {os.path.basename(chart['path'])}")
    print(f"Review pack       : {pack}")

    if args.open:
        webbrowser.open("file://" + os.path.abspath(pack))


if __name__ == "__main__":
    main()
