"""Shared matplotlib plot styling for consistent, readable plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

DEFAULT_SAVE_DPI = 180

# -- Color palette (VS Code-inspired dark gray) --
BG_COLOR = "#1e1e1e"
AXES_BG = "#252525"
TEXT_COLOR = "#d4d4d4"
TICK_COLOR = "#b0b0b0"
AXES_EDGE = "#444444"
GRID_COLOR = "#3a3a3a"
LEGEND_BG = "#2d2d2d"
LEGEND_EDGE = "#444444"
IMAGE_BORDER = "#555555"

# Widget colors
WIDGET_BG = "#353535"
WIDGET_HOVER = "#4a4a4a"
WIDGET_TEXT = "#d4d4d4"
WIDGET_ACTIVE = "#505050"

BASE_PLOT_RC: dict[str, float] = {
    "figure.dpi": 150,
    "savefig.dpi": 180,
    "font.size": 11.0,
    "axes.titlesize": 15.0,
    "axes.labelsize": 13.0,
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,
    "legend.fontsize": 11.0,
    "legend.title_fontsize": 12.0,
    "figure.titlesize": 18.0,
}

DARK_COLORS_RC: dict[str, str] = {
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": AXES_BG,
    "axes.edgecolor": AXES_EDGE,
    "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "xtick.color": TICK_COLOR,
    "ytick.color": TICK_COLOR,
    "grid.color": GRID_COLOR,
    "legend.facecolor": LEGEND_BG,
    "legend.edgecolor": LEGEND_EDGE,
    "savefig.facecolor": BG_COLOR,
    "lines.color": "#9cdcfe",
    "patch.edgecolor": AXES_EDGE,
}


def build_plot_rc(font_scale: float = 1.0) -> dict[str, float]:
    """Build matplotlib rcParams dict, optionally scaled."""
    scale = max(font_scale, 0.5)
    return {key: value * scale for key, value in BASE_PLOT_RC.items()}


def scale_figsize(
    width: float, height: float, figure_scale: float = 1.0
) -> tuple[float, float]:
    """Scale a base figure size."""
    scale = max(figure_scale, 0.5)
    return width * scale, height * scale


def apply_plot_style(font_scale: float = 1.0) -> None:
    """Apply custom dark style with readable font sizes and good contrast.

    Uses a dark-gray palette instead of pure black so that dark image
    content is visually distinct from the figure background.
    """
    plt.rcParams.update(DARK_COLORS_RC)
    plt.rcParams.update(build_plot_rc(font_scale))
    # Use dark_background's color cycle without its pure-black bg.
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=[
            "#8dd3c7", "#feffb3", "#bfbbd9", "#fa8174",
            "#81b1d2", "#fdb462", "#b3de69", "#bc82bd",
            "#ccebc4", "#ffed6f",
        ]
    )


def style_widget(widget: Button | Slider) -> None:
    """Apply dark-theme colors to a matplotlib Button or Slider."""
    if isinstance(widget, Button):
        widget.color = WIDGET_BG
        widget.hovercolor = WIDGET_HOVER
        widget.ax.set_facecolor(WIDGET_BG)
        widget.label.set_color(WIDGET_TEXT)
    elif isinstance(widget, Slider):
        widget.track.set_facecolor(WIDGET_BG)
        widget.poly.set_facecolor(WIDGET_ACTIVE)
        widget.label.set_color(WIDGET_TEXT)
        widget.valtext.set_color(WIDGET_TEXT)


def style_image_axes(ax: plt.Axes) -> None:
    """Add a subtle border around an image axes for visibility."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(IMAGE_BORDER)
        spine.set_linewidth(1.0)


def enable_slider_scroll(slider: Slider) -> None:
    """Allow mouse-wheel to change a Slider value when hovering over it."""

    def _on_scroll(event):
        if event.inaxes is not slider.ax:
            return
        step = slider.valstep if slider.valstep else (slider.valmax - slider.valmin) / 100
        new_val = slider.val + step * event.step
        new_val = max(slider.valmin, min(slider.valmax, new_val))
        slider.set_val(new_val)

    slider.ax.figure.canvas.mpl_connect("scroll_event", _on_scroll)
