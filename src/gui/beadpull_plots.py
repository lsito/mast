from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def record_label(bdata) -> str:
    """
    Return the filename-only label for a bead-pull record.
    """
    if bdata.filename is None:
        return "record"

    return Path(bdata.filename).name


def remove_extra_axes(ax, colorbar_ax=None) -> None:
    """
    Remove old dynamic colorbar axes while preserving fixed axes.
    """
    figure = ax.figure
    allowed_axes = {ax}

    if colorbar_ax is not None:
        allowed_axes.add(colorbar_ax)

    for extra_ax in list(figure.axes):
        if extra_ax not in allowed_axes:
            extra_ax.remove()


def clear_colorbar_axis(colorbar_ax) -> None:
    """
    Clear and hide the fixed colorbar axes.
    """
    if colorbar_ax is None:
        return

    colorbar_ax.clear()
    colorbar_ax.set_visible(False)


def style_axes(ax) -> None:
    """
    Apply common plot styling.
    """
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.35,
    )

    ax.tick_params(
        axis="both",
        labelsize=11,
        width=0.8,
        colors="#374151",
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#cbd5e1")

    ax.set_facecolor("#ffffff")


def plot_records(
    ax,
    records: Iterable,
    plot_key: str,
    legend: bool = False,
    colorbar_ax=None,
) -> None:
    """
    Plot the requested bead-pull records on one axes.
    """
    records = list(records)

    remove_extra_axes(ax, colorbar_ax=colorbar_ax)
    clear_colorbar_axis(colorbar_ax)

    ax.clear()
    ax.set_aspect("auto")
    style_axes(ax)

    colorbar_mappable = None

    for bdata in records:
        mappable = plot_single_record(ax, bdata, plot_key)

        if mappable is not None:
            colorbar_mappable = mappable

    finish_plot(ax, plot_key)

    if colorbar_mappable is not None and colorbar_ax is not None:
        colorbar_ax.set_visible(True)
        colorbar = ax.figure.colorbar(colorbar_mappable, cax=colorbar_ax)
        colorbar.set_label("Normalized bead position", fontsize=11)
        colorbar.ax.tick_params(labelsize=10)

    if legend and len(records) > 0:
        ax.legend(
            fontsize=8,
            frameon=True,
            framealpha=0.9,
            edgecolor="#cbd5e1",
            facecolor="#ffffff",
        )


def plot_single_record(ax, bdata, plot_key: str):
    """
    Plot one bead-pull record using the selected plot type.
    """
    label = record_label(bdata)

    if plot_key == "df_to_tune":
        plot_cell_array(ax, bdata.df2tune, scale=1e-6, label=label)

    elif plot_key == "phase_advance":
        plot_line_array(ax, bdata.phiadv, scale=1.0, label=label)

    elif plot_key == "s11_beadpull":
        return plot_complex_signal(
            ax=ax,
            bdata=bdata,
            label=label,
            signal=get_s11_signal(bdata),
        )

    elif plot_key == "ds11_bp":
        return plot_complex_signal(
            ax=ax,
            bdata=bdata,
            label=label,
            signal=get_ds11_bp_signal(bdata),
        )

    elif plot_key == "abs_ds11_bp":
        return plot_abs_ds11_bp_samples(ax, bdata, label)

    elif plot_key == "abs_ds11_bp_z":
        return plot_abs_ds11_bp_z(ax, bdata, label)

    elif plot_key == "mag_e":
        if bdata.Ebp is not None:
            plot_cell_array(ax, np.abs(bdata.Ebp), scale=1.0, label=label)

    elif plot_key == "mag_peaks_e":
        if bdata.dref is not None:
            plot_cell_array(ax, np.sqrt(np.abs(bdata.dref)), scale=1.0, label=label)

    elif plot_key == "zero_line":
        plot_zero_line(ax, bdata, label)

    elif plot_key == "pm_abs_ds11":
        plot_pm_abs_ds11(ax, bdata, label)

    elif plot_key == "phi_vs_freq":
        if bdata.gamma is not None:
            plot_cell_array(ax, np.angle(bdata.gamma), scale=1.0, label=label)

    elif plot_key == "local_s11":
        plot_local_s11(ax, bdata, label)

    elif plot_key == "local_s11_cell":
        if bdata.s11local is not None:
            plot_cell_array(ax, np.abs(bdata.s11local), scale=1e3, label=label)

    elif plot_key == "wbn":
        if bdata.B is not None:
            plot_cell_array(
                ax,
                np.abs(bdata.B),
                scale=1.0,
                label=label,
                start_at_zero=True,
            )

    elif plot_key == "wfn":
        if bdata.A is not None:
            plot_cell_array(
                ax,
                np.abs(bdata.A),
                scale=1.0,
                label=label,
                start_at_zero=True,
            )

    elif plot_key == "arg_ds11_bp_z":
        if bdata.ds11global is not None:
            plot_cell_array(ax, np.angle(bdata.ds11global), scale=1.0, label=label)

    elif plot_key == "abs_arg_ds11_bp_z":
        if bdata.ds11global is not None:
            plot_cell_array(
                ax,
                np.abs(np.angle(bdata.ds11global)),
                scale=1.0,
                label=label,
            )

    return None


def get_s11_signal(bdata):
    """
    Return the best available raw S11 bead-pull signal.
    """
    if getattr(bdata, "sorg", None) is not None:
        return bdata.sorg

    if getattr(bdata, "aorg", None) is not None:
        return bdata.aorg

    if getattr(bdata, "a", None) is not None:
        return bdata.a

    return None


def get_ds11_bp_signal(bdata):
    """
    Return the best available zero-line-subtracted bead-pull signal.
    """
    if getattr(bdata, "a", None) is not None:
        return bdata.a

    if getattr(bdata, "atp", None) is not None:
        return bdata.atp

    if getattr(bdata, "sorg", None) is not None:
        return bdata.sorg

    return None


def plot_complex_signal(ax, bdata, label: str, signal):
    """
    Plot a complex bead-pull trajectory.

    The full trajectory is shown as a line. Detected peak positions are shown
    as colored open circles.
    """
    if signal is None:
        return None

    signal = np.asarray(signal, dtype=np.complex128)

    ax.plot(
        np.real(signal),
        np.imag(signal),
        linewidth=0.9,
        alpha=0.95,
        label=label,
    )

    scatter = plot_peak_markers_on_complex_signal(ax, bdata, signal)

    ax.set_aspect("equal", adjustable="datalim")

    return scatter


def plot_peak_markers_on_complex_signal(ax, bdata, signal):
    """
    Plot colored peak markers on a complex trajectory.
    """
    if bdata.locpk is None or len(bdata.locpk) == 0:
        return None

    locpk = valid_peak_indices(bdata.locpk, len(signal))

    if len(locpk) == 0:
        return None

    peak_signal = signal[locpk]
    colors = np.linspace(0.0, 1.0, len(peak_signal))

    scatter = ax.scatter(
        np.real(peak_signal),
        np.imag(peak_signal),
        c=colors,
        cmap="jet",
        s=42,
        marker="o",
        facecolors="none",
        linewidths=1.4,
        zorder=4,
    )

    return scatter


def plot_abs_ds11_bp_samples(ax, bdata, label: str):
    """
    Plot |dS11 BP| against raw sample index.
    """
    signal = get_ds11_bp_signal(bdata)

    if signal is None:
        return None

    signal = np.asarray(signal, dtype=np.complex128)
    x = np.arange(len(signal))
    y = np.abs(signal)

    ax.plot(
        x,
        y,
        linewidth=1.0,
        alpha=0.95,
        label=label,
    )

    if bdata.locpk is None or len(bdata.locpk) == 0:
        return None

    locpk = valid_peak_indices(bdata.locpk, len(signal))

    if len(locpk) == 0:
        return None

    colors = np.linspace(0.0, 1.0, len(locpk))

    scatter = ax.scatter(
        locpk,
        y[locpk],
        c=colors,
        cmap="jet",
        s=38,
        marker="o",
        facecolors="none",
        linewidths=1.3,
        zorder=4,
    )

    return scatter


def plot_abs_ds11_bp_z(ax, bdata, label: str):
    """
    Plot |dS11 BP| against bead-pull cell coordinate z.

    The detected peak positions define integer cell coordinates.
    """
    signal = get_ds11_bp_signal(bdata)

    if signal is None:
        return None

    signal = np.asarray(signal, dtype=np.complex128)
    y = np.abs(signal)

    if bdata.locpk is None or len(bdata.locpk) < 2:
        x = np.arange(len(y))
        ax.plot(x, y, linewidth=1.0, alpha=0.95, label=label)
        return None

    locpk = valid_peak_indices(bdata.locpk, len(signal))

    if len(locpk) < 2:
        x = np.arange(len(y))
        ax.plot(x, y, linewidth=1.0, alpha=0.95, label=label)
        return None

    z = sample_index_to_cell_coordinate(
        sample_indices=np.arange(len(y)),
        peak_indices=locpk,
    )

    ax.plot(
        z,
        y,
        linewidth=1.0,
        alpha=0.95,
        label=label,
    )

    colors = np.linspace(0.0, 1.0, len(locpk))
    cell_positions = np.arange(1, len(locpk) + 1)

    scatter = ax.scatter(
        cell_positions,
        y[locpk],
        c=colors,
        cmap="jet",
        s=38,
        marker="o",
        facecolors="none",
        linewidths=1.3,
        zorder=4,
    )

    ax.set_xlim(0, len(locpk) + 1)
    ax.set_xticks(np.arange(0, len(locpk) + 2, 1))
    ax.tick_params(axis="x", labelrotation=90)

    return scatter


def sample_index_to_cell_coordinate(
    sample_indices: np.ndarray,
    peak_indices: np.ndarray,
) -> np.ndarray:
    """
    Convert raw sample indices to a bead-pull cell coordinate.
    """
    peak_indices = np.asarray(peak_indices, dtype=float)
    cell_positions = np.arange(1, len(peak_indices) + 1, dtype=float)

    z = np.interp(sample_indices, peak_indices, cell_positions)

    left_slope = (cell_positions[1] - cell_positions[0]) / (
        peak_indices[1] - peak_indices[0]
    )
    right_slope = (cell_positions[-1] - cell_positions[-2]) / (
        peak_indices[-1] - peak_indices[-2]
    )

    left_mask = sample_indices < peak_indices[0]
    right_mask = sample_indices > peak_indices[-1]

    z[left_mask] = cell_positions[0] + (
        sample_indices[left_mask] - peak_indices[0]
    ) * left_slope

    z[right_mask] = cell_positions[-1] + (
        sample_indices[right_mask] - peak_indices[-1]
    ) * right_slope

    return z


def valid_peak_indices(locpk, signal_length: int) -> np.ndarray:
    """
    Return valid integer peak indices.
    """
    locpk = np.asarray(locpk, dtype=int)
    locpk = locpk[(locpk >= 0) & (locpk < signal_length)]

    return locpk


def plot_cell_array(
    ax,
    values,
    scale: float = 1.0,
    label: str | None = None,
    start_at_zero: bool = False,
) -> None:
    """
    Plot an array against cell number.
    """
    if values is None:
        return

    y = np.asarray(values) * scale

    if start_at_zero:
        x = np.arange(len(y))
    else:
        x = np.arange(1, len(y) + 1)

    ax.plot(
        x,
        y,
        marker="o",
        markersize=6.5,
        markeredgewidth=1.2,
        linestyle="None",
        markerfacecolor="none",
        label=label,
    )


def plot_line_array(
    ax,
    values,
    scale: float = 1.0,
    label: str | None = None,
    start_at_zero: bool = False,
) -> None:
    """
    Plot an array against cell number using a connected line.
    """
    if values is None:
        return

    y = np.asarray(values) * scale

    if start_at_zero:
        x = np.arange(len(y))
    else:
        x = np.arange(1, len(y) + 1)

    ax.plot(
        x,
        y,
        marker="o",
        markersize=6.5,
        markeredgewidth=1.2,
        linewidth=1.3,
        label=label,
    )


def plot_zero_line(ax, bdata, label: str) -> None:
    """
    Plot zero-line diagnostic data.
    """
    if bdata.aorg is not None:
        x = np.arange(len(bdata.aorg))
        ax.plot(x, np.real(bdata.aorg), label=f"{label} real(aorg)")

    if bdata.a_zero is not None:
        x = np.arange(len(bdata.a_zero))
        ax.plot(x, np.real(bdata.a_zero), label=f"{label} real(a_zero)")


def plot_pm_abs_ds11(ax, bdata, label: str) -> None:
    """
    Plot positive and negative absolute dS11.
    """
    if bdata.ds11 is None:
        return

    y = np.abs(bdata.ds11) * 1e3
    x = np.arange(1, len(y) + 1)

    ax.plot(
        x,
        y,
        marker="o",
        markersize=6.5,
        markeredgewidth=1.2,
        linestyle="None",
        markerfacecolor="none",
        label=f"{label} +",
    )

    ax.plot(
        x,
        -y,
        marker="o",
        markersize=6.5,
        markeredgewidth=1.2,
        linestyle="None",
        markerfacecolor="none",
        label=f"{label} -",
    )


def plot_local_s11(ax, bdata, label: str) -> None:
    """
    Plot local S11 real and imaginary parts.
    """
    if bdata.s11local is None:
        return

    x = np.arange(1, len(bdata.s11local) + 1)

    ax.plot(
        x,
        np.real(bdata.s11local) * 1e3,
        marker="o",
        markersize=6.5,
        markeredgewidth=1.2,
        linestyle="None",
        markerfacecolor="none",
        label=f"{label} real",
    )

    ax.plot(
        x,
        np.imag(bdata.s11local) * 1e3,
        marker="o",
        markersize=6.5,
        markeredgewidth=1.2,
        linestyle="None",
        markerfacecolor="none",
        label=f"{label} imag",
    )


def finish_plot(ax, plot_key: str) -> None:
    """
    Apply title and axis labels.
    """
    titles = {
        "df_to_tune": ("df to tune", "Cell", "df to tune [MHz]"),
        "phase_advance": ("phase advance", "Cell interval", "Phase advance [deg]"),
        "s11_beadpull": ("S11 bead-pull", "Real(S11)", "Imag(S11)"),
        "ds11_bp": ("dS11 BP", "Real(dS11 BP)", "Imag(dS11 BP)"),
        "abs_ds11_bp": ("|dS11| BP", "Sample", "|dS11 BP|"),
        "abs_ds11_bp_z": ("|dS11| BP (z)", "Cell coordinate z", "|dS11 BP|"),
        "mag_e": ("Mag(E)", "Cell", "|E|"),
        "mag_peaks_e": ("Mag(peaks(E))", "Cell", "sqrt(|dref|)"),
        "zero_line": ("0-Line", "Sample", "Real part"),
        "pm_abs_ds11": ("+/-|dS11|", "Cell", "+/-|dS11| [mU]"),
        "phi_vs_freq": ("phi v.s. freq", "Cell", "arg(gamma) [rad]"),
        "local_s11": ("local S11", "Cell", "local S11 [mU]"),
        "local_s11_cell": ("local S11(cell)", "Cell", "|local S11| [mU]"),
        "wbn": ("wbn", "Cell boundary", "|wbn|"),
        "wfn": ("wfn", "Cell boundary", "|wfn|"),
        "arg_ds11_bp_z": ("arg(dS11) BP (z)", "Cell", "arg(dS11 global) [rad]"),
        "abs_arg_ds11_bp_z": ("|arg(dS11) BP(z)|", "Cell", "|arg(dS11 global)| [rad]"),
    }

    title, xlabel, ylabel = titles.get(plot_key, ("Plot", "x", "y"))

    ax.set_title(title, fontsize=14, fontweight="600", pad=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    style_axes(ax)