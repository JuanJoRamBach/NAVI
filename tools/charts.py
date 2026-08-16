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

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)

    if chart_type == "pie":
        values = series[0]["values"]
        ax.pie(values, labels=labels, autopct="%1.1f%%")
        ax.axis("equal")
    elif chart_type == "line":
        for s in series:
            ax.plot(labels, s["values"], marker="o", label=s["name"])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if len(series) > 1:
            ax.legend()
        ax.grid(True, alpha=0.3)
    elif chart_type == "bar":
        x = range(len(labels))
        width = 0.8 / max(len(series), 1)
        for i, s in enumerate(series):
            offsets = [xi + i * width for xi in x]
            ax.bar(offsets, s["values"], width=width, label=s["name"])
        ax.set_xticks([xi + width * (len(series) - 1) / 2 for xi in x])
        ax.set_xticklabels(labels)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if len(series) > 1:
            ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
    else:
        plt.close(fig)
        raise ChartError(f"Unknown chart_type: {chart_type}")

    ax.set_title(title)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
