from __future__ import annotations
from pathlib import Path
import numpy as np


def write_vna_csv(
    filename: str,
    frequency_hz,
    trace_names: list[str],
    trace_data,
) -> None:
    """
    Save VNA trace data in the same semicolon-separated format as the MATLAB code.

    Parameters
    ----------
    filename:
        Output CSV filename.

    frequency_hz:
        Frequency vector with shape (n_points,).

    trace_names:
        List of trace names, for example ["S11", "S21", "S12", "S22"].

    trace_data:
        Complex array with shape (n_points, n_traces).
    """
    frequency_hz = np.asarray(frequency_hz, dtype=float).reshape(-1)
    trace_data = np.asarray(trace_data, dtype=np.complex128)

    if trace_data.ndim != 2:
        raise ValueError("trace_data must have shape (n_points, n_traces).")

    n_points, n_traces = trace_data.shape

    if len(frequency_hz) != n_points:
        raise ValueError("frequency_hz length does not match trace_data.")

    if len(trace_names) != n_traces:
        raise ValueError("trace_names length does not match trace_data.")

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    header_columns = ["trigger"]

    for name in trace_names:
        header_columns.append(f"Re({name})")
        header_columns.append(f"Im({name})")

    with open(path, "w", encoding="utf-8", newline="") as file:
        file.write("#Read via python\n")
        file.write("#\n")
        file.write(";".join(header_columns) + "\n")

        for point_index in range(n_points):
            row = [f"{frequency_hz[point_index]:19.15e}"]

            for trace_index in range(n_traces):
                value = trace_data[point_index, trace_index]
                row.append(f"{value.real: 19.15e}")
                row.append(f"{value.imag: 19.15e}")

            file.write(";".join(row) + "\n")