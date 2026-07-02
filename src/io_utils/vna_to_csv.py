from __future__ import annotations

from datetime import datetime

from src.core.vna_control import RSVNASocketController


def read_vna_to_csv(
    filename: str,
    ip_address: str = "128.11.11.11",
    port: int = 5025,
    channel: int = 1,
    trigger: bool = False,
    trigger_wait_s: float = 5.0,
) -> bool:
    """
    Read VNA traces through the raw socket controller and save a local CSV.

    This is the GUI entry point used by BeadpullFileDialog.
    It does not require VISA/PyVISA.

    By default trigger=False because the safest workflow is to read the
    current displayed/completed VNA data. Set trigger=True later if you want
    this button to trigger a fresh sweep before saving.
    """
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    try:
        print(f"Connecting to VNA at {ip_address}:{port}", flush=True)
        print("Starting to get data from machine...", flush=True)

        with RSVNASocketController(
            ip_address=ip_address,
            port=port,
            timeout_s=120,
            release_on_close=True,
            release_after_measurement=True,
        ) as vna:
            print(vna.status(), flush=True)

            saved_path = vna.save_matlab_style_csv(
                filename=filename,
                channel=channel,
                trigger=trigger,
                trigger_wait_s=trigger_wait_s,
                trigger_use_opc=False,
                use_measurement_names=True,
                header_source="#Read via matlab",
                release_after=True,
            )

        print("Success to acquire data!", flush=True)
        print(f"All the data were written to file successfully: {saved_path}", flush=True)
        return True

    except Exception as exc:
        print("VNA read failed.", flush=True)
        print(exc, flush=True)
        return False
