from __future__ import annotations
import numpy as np

def read_vna_measurement(
    ip_address: str = "128.11.11.11",
    port: int = 1601,
):
    """
    Read VNA data.

    This is the only place that should contain VNA-specific communication.

    For now this is a placeholder/interface. Once we know the Python library or
    SCPI commands you want to use, this function should return:

        trace_names, trace_data, frequency_hz

    where:
        trace_names  -> list[str]
        trace_data   -> complex array, shape (n_points, n_traces)
        frequency_hz -> float array, shape (n_points,)
    """
    raise NotImplementedError(
        "Connect this function to your Rohde & Schwarz VNA communication layer."
    )


def validate_vna_data(trace_names, trace_data, frequency_hz) -> None:
    """
    Validate VNA data before saving.
    """
    frequency_hz = np.asarray(frequency_hz)
    trace_data = np.asarray(trace_data)

    if len(trace_names) == 0:
        raise ValueError("No trace names were returned by the VNA.")

    if trace_data.size == 0:
        raise ValueError("No trace data were returned by the VNA.")

    if trace_data.ndim != 2:
        raise ValueError("trace_data must have shape (n_points, n_traces).")

    if trace_data.shape[1] != len(trace_names):
        raise ValueError("Number of traces does not match trace names.")

    if trace_data.shape[0] != len(frequency_hz):
        raise ValueError("Number of frequency points does not match trace data.")
    
def trigger_vna(
    ip_address: str = "128.11.11.11",
    port: int = 1601,
    trigger_command: str = "*TRG\n",
    timeout_s: float = 3.0,
) -> None:
    """
    Send a trigger command to the VNA over a raw TCP socket.

    The default command is the standard SCPI trigger command `*TRG`.
    If your VNA helper server expects another command, change
    `trigger_command`.
    """
    import socket

    with socket.create_connection((ip_address, int(port)), timeout=timeout_s) as sock:
        sock.sendall(trigger_command.encode("ascii"))