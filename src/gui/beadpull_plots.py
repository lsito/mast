from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np


MAIN_AXES_RECT = [0.10, 0.12, 0.74, 0.78]

STACKED_AXES_RECTS = [
    [0.10, 0.70, 0.74, 0.22],
    [0.10, 0.40, 0.74, 0.22],
    [0.10, 0.10, 0.74, 0.22],
]

ZERO_LINE_AXES_RECTS = [
    [0.08, 0.57, 0.37, 0.30],
    [0.08, 0.14, 0.37, 0.30],
    [0.56, 0.25, 0.34, 0.48],
]

STACKED_PLOT_KEYS = {
    "local_s11_cell",
    "wbn",
    "wfn",
}


def record_label(bdata) -> str:
    """
    Return the filename-only label for a bead-pull record.
    """
    if bdata.filename is None:
        return "record"

    return Path(bdata.filename).name


def get_record_array(bdata, *names):
    """
    Return the first existing non-empty array from a bead-pull record.

    This supports both notebook-style names and application-style names.
    """
    for name in names:
        if hasattr(bdata, name):
            value = getattr(bdata, name)

            if value is not None:
                return value

    if hasattr(bdata, "info") and isinstance(bdata.info, dict):
        for name in names:
            value = bdata.info.get(name)

            if value is not None:
                return value

    return None


def plot_records(
    ax,
    records: Iterable,
    plot_key: str,
    legend: bool = False,
    colorbar_ax=None,
) -> None:
    """
    Plot all selected bead-pull records.

    Single-axes plots use the fixed main axes. Abs/Re/Im plots use three fixed
    stacked axes. The 0-Line plot uses a MATLAB-like three-panel layout.
    """
    records = list(records)

    if plot_key == "zero_line":
        zero_line_axes = prepare_zero_line_axes(ax, colorbar_ax)
        plot_zero_line_records(zero_line_axes, records, legend)
        return

    if plot_key in STACKED_PLOT_KEYS:
        stacked_axes = prepare_stacked_axes(ax, colorbar_ax)
        plot_stacked_records(stacked_axes, records, plot_key, legend)
        return

    prepare_single_axes(ax, colorbar_ax)

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


def prepare_single_axes(ax, colorbar_ax=None) -> None:
    """
    Prepare the figure for a single-axes plot.
    """
    figure = ax.figure

    for extra_ax in list(figure.axes):
        if extra_ax not in {ax, colorbar_ax}:
            extra_ax.remove()

    ax.set_visible(True)
    ax.set_position(MAIN_AXES_RECT)
    ax.clear()
    ax.set_aspect("auto")
    style_axes(ax)

    if colorbar_ax is not None:
        colorbar_ax.clear()
        colorbar_ax.set_visible(False)


def prepare_stacked_axes(ax, colorbar_ax=None) -> list:
    """
    Prepare the figure for a three-panel Abs/Re/Im plot.
    """
    figure = ax.figure

    for extra_ax in list(figure.axes):
        if extra_ax not in {ax, colorbar_ax}:
            extra_ax.remove()

    ax.clear()
    ax.set_visible(False)

    if colorbar_ax is not None:
        colorbar_ax.clear()
        colorbar_ax.set_visible(False)

    stacked_axes = []

    for rect in STACKED_AXES_RECTS:
        stacked_ax = figure.add_axes(rect)
        style_axes(stacked_ax)
        stacked_axes.append(stacked_ax)

    return stacked_axes


def prepare_zero_line_axes(ax, colorbar_ax=None) -> list:
    """
    Prepare the figure for the MATLAB-like 0-Line diagnostic layout.
    """
    figure = ax.figure

    for extra_ax in list(figure.axes):
        if extra_ax not in {ax, colorbar_ax}:
            extra_ax.remove()

    ax.clear()
    ax.set_visible(False)

    if colorbar_ax is not None:
        colorbar_ax.clear()
        colorbar_ax.set_visible(False)

    zero_line_axes = []

    for rect in ZERO_LINE_AXES_RECTS:
        zero_ax = figure.add_axes(rect)
        style_axes(zero_ax)
        zero_line_axes.append(zero_ax)

    return zero_line_axes


def style_axes(ax) -> None:
    """
    Apply common axes styling.
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
        return plot_complex_trajectory(
            ax=ax,
            bdata=bdata,
            label=label,
            signal=get_s11_signal(bdata),
        )

    elif plot_key == "ds11_bp":
        return plot_complex_trajectory(
            ax=ax,
            bdata=bdata,
            label=label,
            signal=get_ds11_bp_signal(bdata),
        )

    elif plot_key == "abs_ds11_bp":
        plot_sample_signal(
            ax=ax,
            label=label,
            signal=get_ds11_bp_signal(bdata),
            transform=np.abs,
        )

    elif plot_key == "abs_ds11_bp_z":
        plot_z_signal(
            ax=ax,
            bdata=bdata,
            label=label,
            signal=get_ds11_bp_signal(bdata),
            transform=np.abs,
            integer_ticks=True,
        )

    elif plot_key == "mag_e":
        plot_sample_signal(
            ax=ax,
            label=label,
            signal=get_ds11_bp_signal(bdata),
            transform=lambda values: np.sqrt(np.abs(values)),
        )

    elif plot_key == "mag_peaks_e":
        plot_mag_peaks_e(ax, bdata, label)

    elif plot_key == "pm_abs_ds11":
        plot_signed_ds11(ax, bdata, label)

    elif plot_key == "phi_vs_freq":
        plot_phi_vs_frequency(ax, bdata, label)

    elif plot_key == "local_s11":
        plot_complex_cells(ax, bdata.s11local, label)

    elif plot_key == "arg_ds11_bp_z":
        plot_z_signal(
            ax=ax,
            bdata=bdata,
            label=label,
            signal=get_ds11_bp_signal(bdata),
            transform=unwrapped_phase_degrees,
            integer_ticks=True,
            line_color="#8e44ad",
        )

    elif plot_key == "abs_arg_ds11_bp_z":
        plot_z_signal(
            ax=ax,
            bdata=bdata,
            label=label,
            signal=get_ds11_bp_signal(bdata),
            transform=wrapped_negative_phase_degrees,
            integer_ticks=True,
        )

    return None


def plot_zero_line_records(axes, records: list, legend: bool = False) -> None:
    """
    Plot the MATLAB-like 0-Line diagnostic.

    Top-left panel shows Re(S11) and Im(S11) against sample coordinate.
    Bottom-left panel shows Abs(S11) against sample coordinate.
    Right panel shows the complex S11 trajectory.
    """
    re_im_ax, abs_ax, complex_ax = axes

    for bdata in records:
        signal = get_s11_signal(bdata)

        if signal is None:
            continue

        signal = np.asarray(signal, dtype=np.complex128)
        x = np.arange(len(signal))

        re_im_ax.plot(
            x,
            np.real(signal),
            linewidth=0.9,
            color="blue",
            label="Re",
        )

        re_im_ax.plot(
            x,
            np.imag(signal),
            linewidth=0.9,
            color="magenta",
            label="Im",
        )

        abs_ax.plot(
            x,
            np.abs(signal),
            linewidth=0.9,
            color="red",
            label="abs(aorg)",
        )

        complex_ax.plot(
            np.real(signal),
            np.imag(signal),
            linewidth=0.9,
            color="limegreen",
            label="aorg",
        )

        if getattr(bdata, "a_zero", None) is not None:
            zero_line = np.asarray(bdata.a_zero, dtype=np.complex128)

            if len(zero_line) == len(signal):
                re_im_ax.plot(
                    x,
                    np.real(zero_line),
                    linewidth=1.0,
                    linestyle="--",
                    color="green",
                    alpha=0.85,
                    label="Re zero-line",
                )

                re_im_ax.plot(
                    x,
                    np.imag(zero_line),
                    linewidth=1.0,
                    linestyle="--",
                    color="red",
                    alpha=0.85,
                    label="Im zero-line",
                )

                abs_ax.plot(
                    x,
                    np.abs(zero_line),
                    linewidth=1.0,
                    linestyle="--",
                    color="orange",
                    alpha=0.85,
                    label="abs(a_zero)",
                )

                complex_ax.plot(
                    np.real(zero_line),
                    np.imag(zero_line),
                    linewidth=1.0,
                    linestyle="--",
                    color="orange",
                    alpha=0.85,
                    label="a_zero",
                )

    re_im_ax.set_title("")
    abs_ax.set_title("")
    complex_ax.set_title("")

    re_im_ax.set_xlabel("z", fontsize=11)
    re_im_ax.set_ylabel("Re(S11) and Im(S11)", fontsize=11)

    abs_ax.set_xlabel("z", fontsize=11)
    abs_ax.set_ylabel("Abs(S11)", fontsize=11)

    complex_ax.set_xlabel("Re(S11)", fontsize=11)
    complex_ax.set_ylabel("Im(S11)", fontsize=11)
    complex_ax.set_aspect("equal", adjustable="datalim")

    for axis in axes:
        style_axes(axis)

    re_im_ax.legend(
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cbd5e1",
        facecolor="#ffffff",
    )

    abs_ax.legend(
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cbd5e1",
        facecolor="#ffffff",
    )

    complex_ax.legend(
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cbd5e1",
        facecolor="#ffffff",
    )


def plot_stacked_records(axes, records: list, plot_key: str, legend: bool = False) -> None:
    """
    Plot Abs/Re/Im panels for all selected records.
    """
    plotted_anything = False

    for bdata in records:
        label = record_label(bdata)
        values, scale, unit, start_at_zero = stacked_values_for_key(bdata, plot_key)

        if values is None:
            continue

        values = np.asarray(values)

        if values.size == 0:
            continue

        plotted_anything = True

        plot_abs_re_im_stack(
            axes=axes,
            values=values,
            scale=scale,
            unit=unit,
            label=label,
            start_at_zero=start_at_zero,
        )

    title, xlabel = stacked_title_for_key(plot_key)

    axes[0].set_title(title, fontsize=14, fontweight="600", pad=8)
    axes[2].set_xlabel(xlabel, fontsize=12)

    for index, axis in enumerate(axes):
        axis.tick_params(axis="both", labelsize=10)

        if index < 2:
            axis.tick_params(labelbottom=False)

    if not plotted_anything:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "No data available for this plot",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="#6b7280",
            )

    if legend and plotted_anything:
        axes[0].legend(
            fontsize=8,
            frameon=True,
            framealpha=0.9,
            edgecolor="#cbd5e1",
            facecolor="#ffffff",
        )


def stacked_values_for_key(bdata, plot_key: str):
    """
    Return the complex values and display format for stacked plots.

    GUI names:
    local S11(cell) -> bdata.s11local
    wbn             -> bdata.B, fallback bdata.wbn
    wfn             -> bdata.A, fallback bdata.wfn
    """
    if plot_key == "local_s11_cell":
        values = get_record_array(
            bdata,
            "s11local",
            "s11local_org",
        )
        return values, 100.0, "%", False

    if plot_key == "wbn":
        values = get_record_array(
            bdata,
            "B",
            "wbn",
        )
        return values, 1e3, "mU", True

    if plot_key == "wfn":
        values = get_record_array(
            bdata,
            "A",
            "wfn",
        )
        return values, 1e3, "mU", True

    return None, 1.0, "", False


def stacked_title_for_key(plot_key: str) -> tuple[str, str]:
    """
    Return title and x-axis label for stacked plots.
    """
    titles = {
        "local_s11_cell": ("local S11(cell)", "Cell"),
        "wbn": ("B / wbn", "Cell boundary"),
        "wfn": ("A / wfn", "Cell boundary"),
    }

    return titles.get(plot_key, ("Stacked plot", "Index"))


def plot_abs_re_im_stack(
    axes,
    values,
    scale: float,
    unit: str,
    label: str,
    start_at_zero: bool,
) -> None:
    """
    Plot Abs/Re/Im of a complex array on three stacked axes.
    """
    values = np.asarray(values, dtype=np.complex128) * scale

    if values.size == 0:
        return

    if start_at_zero:
        x = np.arange(len(values))
    else:
        x = np.arange(1, len(values) + 1)

    y_values = [
        np.abs(values),
        np.real(values),
        np.imag(values),
    ]

    y_labels = [
        f"Abs [{unit}]",
        f"Re [{unit}]",
        f"Im [{unit}]",
    ]

    for axis, y, y_label in zip(axes, y_values, y_labels):
        axis.plot(
            x,
            y,
            marker="o",
            markersize=5.8,
            markeredgewidth=1.0,
            linewidth=1.1,
            markerfacecolor="none",
            label=label,
        )
        axis.set_ylabel(y_label, fontsize=11)
        style_axes(axis)


def get_s11_signal(bdata):
    """
    Return the best available raw S11 bead-pull signal.

    Priority:
    1. `sorg`
    2. `aorg`
    3. `a`
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

    Priority:
    1. `a`
    2. `atp`
    3. `sorg`
    """
    if getattr(bdata, "a", None) is not None:
        return bdata.a

    if getattr(bdata, "atp", None) is not None:
        return bdata.atp

    if getattr(bdata, "sorg", None) is not None:
        return bdata.sorg

    return None


def plot_complex_trajectory(ax, bdata, label: str, signal):
    """
    Plot a complex trajectory with colored peak markers.
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


def plot_sample_signal(
    ax,
    label: str,
    signal,
    transform: Callable,
    line_color=None,
) -> None:
    """
    Plot a transformed complex signal against raw sample index.
    """
    if signal is None:
        return

    signal = np.asarray(signal, dtype=np.complex128)
    x = np.arange(len(signal))
    y = transform(signal)

    plot_kwargs = {
        "linewidth": 1.0,
        "alpha": 0.95,
        "label": label,
    }

    if line_color is not None:
        plot_kwargs["color"] = line_color

    ax.plot(x, y, **plot_kwargs)


def plot_z_signal(
    ax,
    bdata,
    label: str,
    signal,
    transform: Callable,
    integer_ticks: bool = False,
    line_color=None,
) -> None:
    """
    Plot a transformed complex signal against bead-pull cell coordinate z.
    """
    if signal is None:
        return

    signal = np.asarray(signal, dtype=np.complex128)
    y = transform(signal)

    if bdata.locpk is None or len(bdata.locpk) < 2:
        x = np.arange(len(y))
    else:
        locpk = valid_peak_indices(bdata.locpk, len(signal))

        if len(locpk) < 2:
            x = np.arange(len(y))
        else:
            x = sample_index_to_cell_coordinate(
                sample_indices=np.arange(len(y)),
                peak_indices=locpk,
            )

    plot_kwargs = {
        "linewidth": 1.0,
        "alpha": 0.95,
        "label": label,
    }

    if line_color is not None:
        plot_kwargs["color"] = line_color

    ax.plot(x, y, **plot_kwargs)

    if integer_ticks and bdata.locpk is not None:
        number_of_cells = len(bdata.locpk)
        ax.set_xlim(0, number_of_cells + 1)
        ax.set_xticks(np.arange(0, number_of_cells + 2, 1))
        ax.tick_params(axis="x", labelrotation=90)


def sample_index_to_cell_coordinate(
    sample_indices: np.ndarray,
    peak_indices: np.ndarray,
) -> np.ndarray:
    """
    Convert raw sample indices to a bead-pull cell coordinate.

    Peak 1 maps to z=1, peak 2 maps to z=2, and so on.
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


def unwrapped_phase_degrees(signal: np.ndarray) -> np.ndarray:
    """
    Return unwrapped phase in degrees, starting near zero.
    """
    phase = np.rad2deg(np.unwrap(np.angle(signal)))
    phase = phase - phase[0]

    if len(phase) > 1 and np.nanmean(np.diff(phase)) > 0:
        phase = -phase

    return phase


def wrapped_negative_phase_degrees(signal: np.ndarray) -> np.ndarray:
    """
    Return wrapped phase in degrees, displayed mostly in the negative branch.
    """
    phase = np.rad2deg(np.angle(signal))
    phase = phase - phase[0]
    phase = np.where(phase > 20.0, phase - 360.0, phase)

    return phase


def valid_peak_indices(locpk, signal_length: int) -> np.ndarray:
    """
    Return valid integer peak indices.
    """
    locpk = np.asarray(locpk, dtype=int)
    locpk = locpk[(locpk >= 0) & (locpk < signal_length)]

    return locpk


def plot_mag_peaks_e(ax, bdata, label: str) -> None:
    """
    Plot magnitude of the electric field at detected peaks.
    """
    if bdata.Ebp is not None:
        values = np.abs(bdata.Ebp)
    elif bdata.dref is not None:
        values = np.sqrt(np.abs(bdata.dref))
    else:
        return

    plot_line_array(ax, values, scale=1.0, label=label)


def plot_signed_ds11(ax, bdata, label: str) -> None:
    """
    Plot signed dS11 values over cell number.
    """
    if bdata.ds11 is None:
        return

    plot_line_array(ax, bdata.ds11, scale=1.0, label=label)


def plot_phi_vs_frequency(ax, bdata, label: str) -> None:
    """
    Plot mean phase advance versus bead-pull frequency.
    """
    x = bdata.f0 / 1e9

    if bdata.phimean is not None and np.isfinite(bdata.phimean):
        y = bdata.phimean
    elif bdata.phiadv is not None and len(bdata.phiadv) > 0:
        y = float(np.nanmean(bdata.phiadv))
    else:
        return

    ax.plot(
        x,
        y,
        marker="o",
        markersize=7.0,
        markeredgewidth=1.2,
        linestyle="None",
        markerfacecolor="none",
        label=label,
    )

    ax.set_xlim(x - 1.5, x + 1.0)
    ax.set_ylim(y - 1.5, y + 1.0)


def plot_complex_cells(ax, values, label: str) -> None:
    """
    Plot one complex cell array in the complex plane.
    """
    if values is None:
        return

    values = np.asarray(values, dtype=np.complex128)

    ax.plot(
        np.real(values),
        np.imag(values),
        marker="o",
        markersize=6.5,
        markeredgewidth=1.2,
        linestyle="None",
        markerfacecolor="none",
        label=label,
    )


def plot_cell_array(
    ax,
    values,
    scale: float = 1.0,
    label: str | None = None,
    start_at_zero: bool = False,
) -> None:
    """
    Plot an array against cell number using markers only.
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
        markerfacecolor="none",
        label=label,
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
        "mag_e": ("Mag(E)", "Sample", "|E|"),
        "mag_peaks_e": ("Mag(peaks(E))", "Cell", "|E peak|"),
        "zero_line": ("0-Line", "Sample", "Real part"),
        "pm_abs_ds11": ("+/-|dS11|", "Cell", "sgn*|dS11|"),
        "phi_vs_freq": ("phi v.s. freq", "Frequency [GHz]", "Mean phase advance [deg]"),
        "local_s11": ("local S11", "Real(local S11)", "Imag(local S11)"),
        "local_s11_cell": ("local S11(cell)", "Cell", "local S11"),
        "wbn": ("B / wbn", "Cell boundary", "B"),
        "wfn": ("A / wfn", "Cell boundary", "A"),
        "arg_ds11_bp_z": ("arg(dS11) BP (z)", "Cell coordinate z", "Unwrapped phase [deg]"),
        "abs_arg_ds11_bp_z": ("|arg(dS11) BP(z)|", "Cell coordinate z", "Wrapped phase [deg]"),
    }

    title, xlabel, ylabel = titles.get(plot_key, ("Plot", "x", "y"))

    ax.set_title(title, fontsize=14, fontweight="600", pad=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    style_axes(ax)