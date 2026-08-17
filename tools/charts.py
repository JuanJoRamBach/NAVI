"""
tools/charts.py

Deterministic chart rendering for /graph-data. The model's only job is to
produce the underlying numbers (via the render_chart tool call below) —
the actual pixels are drawn by matplotlib, not the model, so the chart
can never hallucinate a wrong-looking trend. This is the "our code draws
it, not the AI" approach agreed on for this command specifically, as
opposed to /create-image, which is the right place for a model to
generate an illustrative image from a text prompt.

Uses the non-interactive "Agg" backend since this runs headless on a
server with no display.
"""

import io

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402 — already a matplotlib dependency, no new install
from matplotlib.colors import to_rgb  # noqa: E402

# Dark, clean, "futuristic but grounded" theme — deliberately not just a
# color swap: no chart-box border, minimal single-axis gridlines, generous
# spacing, a soft ambient glow behind the plot, and vertical gradients on
# bars are doing as much of the work as the palette is.
_BG = "#12141c"
_TEXT = "#e5e7eb"
_MUTED = "#7d8394"
_GRID = "#2a2d3a"
_ACCENTS = ["#22d3ee", "#f472b6", "#4ade80", "#fbbf24", "#a78bfa", "#fb7185"]


def _draw_ambient_glow(fig) -> None:
    """Soft diffused radial glow blobs behind the whole figure — plain
    numpy math (no scipy/blur dependency), just a smooth Gaussian falloff
    per blob, alpha-blended over the dark background."""
    bg_ax = fig.add_axes((0, 0, 1, 1), zorder=-10)
    bg_ax.set_facecolor(_BG)
    bg_ax.axis("off")

    size = 300
    yy, xx = np.mgrid[0:size, 0:size]
    base = np.array(to_rgb(_BG))
    canvas = np.tile(base, (size, size, 1))

    blobs = [(0.12, 0.9, _ACCENTS[0], 0.20), (0.92, 0.08, _ACCENTS[1], 0.15)]
    for cx, cy, color_hex, strength in blobs:
        cx_px, cy_px = cx * size, (1 - cy) * size
        dist = np.sqrt((xx - cx_px) ** 2 + (yy - cy_px) ** 2)
        falloff = np.exp(-(dist ** 2) / (2 * (size * 0.42) ** 2)) * strength
        color = np.array(to_rgb(color_hex))
        canvas = canvas * (1 - falloff[..., None]) + color * falloff[..., None]

    bg_ax.imshow(canvas, extent=(0, 1, 0, 1), origin="lower", aspect="auto")


def _gradient_bars(ax, positions: list[float], heights: list[float], width: float, color_hex: str) -> None:
    """Draws each bar as a vertical gradient (base accent color at the
    bottom, fading toward a lighter tint at the top) instead of a flat
    fill — clips a gradient image to each bar's exact rectangle."""
    base = np.array(to_rgb(color_hex))
    light = base + (1 - base) * 0.6
    grad = np.linspace(0, 1, 256).reshape(256, 1, 1)
    grad_rgb = light.reshape(1, 1, 3) * grad + base.reshape(1, 1, 3) * (1 - grad)
    grad_rgba = np.concatenate([grad_rgb, np.full((256, 1, 1), 0.88)], axis=2)

    for xpos, h in zip(positions, heights):
        if h == 0:
            continue
        y0, y1 = (h, 0) if h < 0 else (0, h)
        im = ax.imshow(
            grad_rgba, extent=(xpos - width / 2, xpos + width / 2, y0, y1),
            origin="lower", aspect="auto", zorder=3,
        )
        rect = plt.Rectangle((xpos - width / 2, y0), width, y1 - y0, transform=ax.transData)
        im.set_clip_path(rect)

CHART_TOOL_NAME = "render_chart"

CHART_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": CHART_TOOL_NAME,
        "description": "Render a chart from structured data. This is the ONLY way to "
                        "produce a chart — call this with the real numbers, don't "
                        "describe a chart in prose.",
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "pie"],
                    "description": "bar/line for comparing or trending numeric values, "
                                    "pie for parts of a whole.",
                },
                "title": {"type": "string"},
                "x_label": {"type": "string", "description": "Axis label (ignored for pie)."},
                "y_label": {"type": "string", "description": "Axis label (ignored for pie)."},
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Category labels, e.g. [\"Q1\", \"Q2\", \"Q3\"].",
                },
                "series": {
                    "type": "array",
                    "description": "One or more data series plotted against `labels`.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "values": {"type": "array", "items": {"type": "number"}},
                        },
                        "required": ["name", "values"],
                    },
                },
            },
            "required": ["chart_type", "title", "labels", "series"],
        },
    },
}


# Forces the model to call render_chart specifically, rather than leaving
# it to "auto" and hoping the model picks it — passed as the tool_choice
# argument to provider.chat().
CHART_TOOL_CHOICE = {"type": "function", "function": {"name": CHART_TOOL_NAME}}


class ChartError(Exception):
    pass


def render_chart(
    chart_type: str,
    title: str,
    labels: list[str],
    series: list[dict],
    x_label: str = "",
    y_label: str = "",
) -> bytes:
    """Renders the chart and returns PNG bytes. Raises ChartError on bad
    input (e.g. mismatched label/value counts) rather than producing a
    misleading chart."""
    if not labels or not series:
        raise ChartError("Chart needs at least one label and one data series.")
    for s in series:
        if len(s.get("values", [])) != len(labels):
            raise ChartError(
                f"Series '{s.get('name', '?')}' has {len(s.get('values', []))} values "
                f"but there are {len(labels)} labels — they must match."
            )

    fig = plt.figure(figsize=(8, 5), dpi=140)
    fig.patch.set_facecolor(_BG)
    _draw_ambient_glow(fig)
    ax = fig.add_axes((0.09, 0.12, 0.85, 0.72))
    ax.set_facecolor("none")

    if chart_type == "pie":
        values = series[0]["values"]
        wedges, _, autotexts = ax.pie(
            values, labels=labels, autopct="%1.1f%%",
            colors=_ACCENTS, startangle=90,
            wedgeprops={"linewidth": 2, "edgecolor": _BG, "alpha": 0.92},
            textprops={"color": _TEXT, "fontsize": 10},
        )
        for at in autotexts:
            at.set_color(_BG)
            at.set_fontweight("bold")
        ax.axis("equal")
    elif chart_type == "line":
        for i, s in enumerate(series):
            color = _ACCENTS[i % len(_ACCENTS)]
            ax.plot(labels, s["values"], marker="o", markersize=5, linewidth=2.2,
                     color=color, alpha=0.9, label=s["name"])
            ax.fill_between(range(len(labels)), s["values"], alpha=0.08, color=color)
        _style_axes(ax, x_label, y_label, len(series) > 1)
    elif chart_type == "bar":
        x = list(range(len(labels)))
        width = 0.8 / max(len(series), 1)
        for i, s in enumerate(series):
            offsets = [xi + i * width for xi in x]
            _gradient_bars(ax, offsets, s["values"], width * 0.88, _ACCENTS[i % len(_ACCENTS)])
            ax.plot([], [], color=_ACCENTS[i % len(_ACCENTS)], linewidth=6, label=s["name"])  # legend swatch
        ax.set_xticks([xi + width * (len(series) - 1) / 2 for xi in x])
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylim(0, max(v for s in series for v in s["values"]) * 1.12)
        _style_axes(ax, x_label, y_label, len(series) > 1)
    else:
        plt.close(fig)
        raise ChartError(f"Unknown chart_type: {chart_type}")

    ax.set_title(title, color=_TEXT, fontsize=14, fontweight="bold", pad=16, loc="left")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=_BG)
    plt.close(fig)
    return buf.getvalue()


def _style_axes(ax, x_label: str, y_label: str, show_legend: bool) -> None:
    """Shared dark-theme styling for bar/line axes: no boxed border, only
    a faint horizontal grid, muted tick labels — the details that make it
    read as designed rather than as matplotlib's defaults."""
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)

    ax.tick_params(colors=_MUTED, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(_MUTED)

    ax.set_xlabel(x_label, color=_MUTED, fontsize=10)
    ax.set_ylabel(y_label, color=_MUTED, fontsize=10)

    ax.grid(True, axis="y", color=_GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)

    if show_legend:
        legend = ax.legend(frameon=False, labelcolor=_TEXT, fontsize=9)
        legend.get_frame().set_facecolor(_BG)
