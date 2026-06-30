from __future__ import annotations

from datetime import datetime

from src.io_utils.vna_csv_writer import write_vna_csv
from src.io_utils.vna_reader import read_vna_measurement, validate_vna_data


def read_vna_to_csv(
    filename: str,
    ip_address: str = "128.11.11.11",
    port: int = 1601,
) -> bool:
    """
    Read VNA traces and save them to a CSV file.
    """
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    try:
        print(f"Connecting to VNA at {ip_address}:{port}", flush=True)
        print("Starting to get data from machine...", flush=True)

        trace_names, trace_data, frequency_hz = read_vna_measurement(
            ip_address=ip_address,
            port=port,
        )

        validate_vna_data(trace_names, trace_data, frequency_hz)

        print("Success to acquire data!", flush=True)
        print(f"Try to save file to {filename}", flush=True)

        write_vna_csv(
            filename=filename,
            frequency_hz=frequency_hz,
            trace_names=trace_names,
            trace_data=trace_data,
        )

        print("All the data were written to file successfully!", flush=True)

        return True

    except Exception as exc:
        print("VNA read failed.", flush=True)
        print(exc, flush=True)
        return False