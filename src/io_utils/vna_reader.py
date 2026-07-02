from __future__ import annotations

import numpy as np

from src.core.vna_control import RSVNASocketController


def read_vna_measurement(
    ip_address: str = "128.11.11.11",
    port: int = 5025,
    channel: int = 1,
    trigger: bool = False,
    trigger_wait_s: float = 5.0,
):
    """
    Read VNA data through the raw socket controller.

    Returns
    -------
    trace_names, trace_data, frequency_hz

    trace_data has shape:
        (n_points, n_traces)
    """
    with RSVNASocketController(
        ip_address=ip_address,
        port=port,
        timeout_s=120,
        release_on_close=True,
        release_after_measurement=True,
    ) as vna:
        data = vna.acquire_trace_data(
            channel=channel,
            trigger=trigger,
            trigger_wait_s=trigger_wait_s,
            trigger_use_opc=False,
            use_measurement_names=True,
        )

    # RSVNASocketController returns traces as [trace][point].
    # The old writer expects [point][trace].
    trace_data = np.array(data.complex_traces, dtype=np.complex128).T
    frequency_hz = np.array(data.frequency_hz, dtype=float)

    return data.trace_names, trace_data, frequency_hz


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
    port: int = 5025,
    channel: int = 1,
    trigger_wait_s: float = 5.0,
    timeout_s: float = 120.0,
) -> None:
    """
    Trigger one VNA sweep over the raw TCP socket and release the VNA cleanly.

    This avoids VISA/PyVISA and avoids *OPC? by using a fixed wait.
    """
    with RSVNASocketController(
        ip_address=ip_address,
        port=port,
        timeout_s=timeout_s,
        release_on_close=True,
        release_after_measurement=True,
    ) as vna:
        vna.trigger_single_sweep(
            channel=channel,
            wait_s=trigger_wait_s,
            use_opc=False,
        )
