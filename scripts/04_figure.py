#!/usr/bin/env python
"""The headline figure: coverage per exceedance-probability bin.

Reads config.COVERAGE_TABLE_PATH (written by 03_verify.py) and draws raw vs
marginal-ACI coverage against the 1 - alpha target, with the per-bin sample
count printed on the figure.

    python scripts/04_figure.py
    python scripts/04_figure.py --table /some/other/coverage_by_bin.csv --out fig.pdf

Form. Paired dots with a target rule, not grouped bars. The quantity is a
coverage *level* near 0.9, and the whole claim is its deviation from the rule;
bars would force a zero baseline (making a 0.90-vs-0.72 gap invisible) or a
truncated axis (which misstates the ratios). Dots carry no baseline obligation,
so the axis can show the range where the effect lives without lying about it.

Coverage is area-weighted by cos(latitude) upstream in 03_verify.py: on a
regular lat-lon grid a box at 75N covers about a quarter of the area of one at
the equator, so an unweighted mean reports a number dominated by the poles,
where there are far more gridpoints than there is planet. The counts printed on
the axes are RAW sample counts, so weighting changes the estimate but never
inflates how much data it appears to rest on.

If a c_trajectories CSV sits next to the coverage table, a second figure is
written showing c_t at the six probe gridpoints. Over a full year that should
oscillate seasonally -- the controller tracking seasonal miscalibration -- which
is a result rather than an artefact.

TODO(lucas): [Sat] Sign off on the figure form before it goes in the paper, and
set --ylim once the real numbers are in -- the default autoscale is fine for
looking at results but a hand-set range is better for a printed figure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from xconformal import config  # noqa: E402

# Palette: slots 1 and 2 of the validated categorical order, assigned in fixed
# order and never cycled. Ink and grid are text tokens, not series colors -- the
# coverage numbers stay in ink so identity is carried by the marks beside them.
SERIES_COLORS = ("#2a78d6", "#eb6834")  # blue = raw, orange = ACI
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8a85"
SURFACE = "#ffffff"


def style() -> None:
    """Print-oriented rcParams: recessive axes, small type, vector-safe fonts."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK_PRIMARY,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "font.size": 9,
        "pdf.fonttype": 42,  # embed TrueType, not Type 3 -- some venues reject Type 3
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def load_table(path: Path) -> pd.DataFrame:
    """Read the tidy coverage table written by 03_verify.py."""
    if not path.is_file():
        raise FileNotFoundError(
            f"No coverage table at {path}. Run `python scripts/03_verify.py` first."
        )
    table = pd.read_csv(path)
    missing = {"method", "bin_label", "coverage", "count"} - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return table


def draw(table: pd.DataFrame, ylim: tuple[float, float] | None = None) -> plt.Figure:
    """Draw per-bin coverage for each method against the target."""
    labels = list(dict.fromkeys(table["bin_label"]))
    methods = list(dict.fromkeys(table["method"]))
    if len(methods) > len(SERIES_COLORS):
        raise ValueError(
            f"{len(methods)} methods but only {len(SERIES_COLORS)} categorical slots; "
            "fold the extras into a facet rather than inventing a hue"
        )
    x = np.arange(len(labels), dtype=float)
    offsets = np.linspace(-0.11, 0.11, len(methods)) if len(methods) > 1 else [0.0]

    fig, ax = plt.subplots(figsize=(5.6, 3.3))

    # Target rule first, so the marks sit on top of it.
    ax.axhline(config.TARGET_COVERAGE, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.annotate(
        f"target $1-\\alpha$ = {config.TARGET_COVERAGE:g}",
        xy=(1.0, config.TARGET_COVERAGE), xycoords=("axes fraction", "data"),
        xytext=(-2, 4), textcoords="offset points",
        ha="right", va="bottom", fontsize=8, color=INK_SECONDARY,
    )

    for (method, color, dx) in zip(methods, SERIES_COLORS, offsets):
        rows = table[table["method"] == method].set_index("bin_label").reindex(labels)
        y = rows["coverage"].to_numpy(dtype=float)
        # Drop the stem to the target rule: shows the miscoverage as a length.
        ax.vlines(x + dx, config.TARGET_COVERAGE, y, color=color, lw=1.6, alpha=0.55, zorder=2)
        # 2px surface ring keeps overlapping marks readable.
        ax.plot(x + dx, y, "o", ms=8, color=color, mec=SURFACE, mew=2.0,
                ls="none", label=method, zorder=3)
        for xi, yi in zip(x + dx, y):
            if np.isnan(yi):
                continue  # empty bin: annotated once per bin below, not once per method
            va, pad = ("bottom", 9) if yi >= config.TARGET_COVERAGE else ("top", -9)
            ax.annotate(f"{yi:.3f}", xy=(xi, yi), xytext=(0, pad),
                        textcoords="offset points", ha="center", va=va,
                        fontsize=7.5, color=INK_PRIMARY, zorder=4)

    # Per-bin counts, printed ON the figure: the high-p_t bins are rare by
    # construction and the reader must be able to see how rare.
    counts = (table.groupby("bin_label", sort=False)["count"].max().reindex(labels))
    for xi, n in zip(x, counts.to_numpy()):
        n = 0 if not np.isfinite(n) else int(n)
        ax.annotate(f"n = {n:,}", xy=(xi, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY)
        if n == 0:
            # One "no data" per empty bin, centred on the group -- not one per
            # method, which would print two overlapping labels on the same spot.
            ax.annotate("no data", xy=(xi, 0.5), xycoords=("data", "axes fraction"),
                        ha="center", va="center", fontsize=7.5, color=INK_MUTED,
                        style="italic")

    ax.set_xticks(x, labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        # Headroom below the lowest mark, so its value label clears the count
        # row pinned to the bottom of the axes. The target rule must stay in
        # view, so the span always includes it.
        values = table["coverage"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        lo = min(finite.min(), config.TARGET_COVERAGE)
        hi = max(finite.max(), config.TARGET_COVERAGE)
        span = max(hi - lo, 1e-3)
        ax.set_ylim(lo - 0.22 * span, hi + 0.10 * span)
    ax.set_xlabel("ensemble exceedance probability $p_t$ "
                  f"(members above climatological {config.CLIM_PERCENTILE:g}th pct)")
    ax.set_ylabel("empirical coverage")
    ax.yaxis.grid(True, color=INK_MUTED, lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # Legend above the axes: the per-bin count row occupies the bottom strip, and
    # an in-axes legend lands on top of it at exactly the bin that matters least.
    ax.legend(frameon=False, ncols=len(methods), loc="lower left",
              bbox_to_anchor=(0.0, 1.0, 1.0, 0.12), mode=None,
              handletextpad=0.4, borderaxespad=0.0, columnspacing=1.4)
    fig.tight_layout()
    return fig


def draw_c_trajectories(frame: pd.DataFrame) -> plt.Figure:
    """c_t at the six probe gridpoints over the evaluation window."""
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    probes = list(dict.fromkeys(frame["probe"]))
    # More than the categorical palette holds, so these are drawn as one family
    # in ink with direct labels rather than eight invented hues.
    colors = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7")
    ax.axhline(0.0, color=INK_MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    for probe, color in zip(probes, colors):
        rows = frame[frame["probe"] == probe]
        times = pd.to_datetime(rows["init_time"])
        ax.plot(times, rows["c"], lw=1.4, color=color, label=probe.replace("_", " "),
                zorder=2)
    ax.set_xlabel("init date")
    ax.set_ylabel("conformal correction $c_t$ (standardized)")
    ax.yaxis.grid(True, color=INK_MUTED, lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, ncols=3, loc="lower left",
              bbox_to_anchor=(0.0, 1.0, 1.0, 0.12), handletextpad=0.4,
              borderaxespad=0.0, columnspacing=1.4)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table", type=Path, default=config.COVERAGE_TABLE_PATH)
    parser.add_argument("--out", type=Path, default=config.FIGURE_PATH)
    parser.add_argument("--ylim", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"))
    parser.add_argument("--c-table", type=Path, default=None,
                        help="c_trajectories CSV; inferred from --table if omitted")
    parser.add_argument("--label", default="",
                        help="banner drawn on the figure, e.g. a provenance warning")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    style()
    table = load_table(args.table)
    fig = draw(table, ylim=tuple(args.ylim) if args.ylim else None)
    if args.label:
        fig.text(0.5, 0.5, args.label, ha="center", va="center", fontsize=15,
                 color="#e34948", alpha=0.30, rotation=18, zorder=10,
                 fontweight="bold")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    png = args.out.with_suffix(".png")
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"wrote {args.out}\nwrote {png}")

    # The exact numbers behind the figure, next to it.
    csv = args.out.with_suffix(".csv")
    table.to_csv(csv, index=False)
    print(f"wrote {csv}")

    c_table = args.c_table
    if c_table is None:
        stem = args.table.stem.replace("coverage_by_bin", "c_trajectories")
        c_table = args.table.with_name(stem + ".csv")
    if c_table.is_file():
        c_fig = draw_c_trajectories(pd.read_csv(c_table))
        c_out = args.out.with_name(args.out.stem + "_ct" + args.out.suffix)
        c_fig.savefig(c_out)
        c_fig.savefig(c_out.with_suffix(".png"), dpi=200)
        plt.close(c_fig)
        print(f"wrote {c_out}")
    else:
        print(f"(no c trajectories at {c_table}; skipping that panel)")

    # A table view of the same numbers, so identity is never colour-alone.
    print()
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
