from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


VNADataFormat = Literal["sdata", "fdata"]


@dataclass(slots=True)
class VNAStatus:
    """
    Current VNA status.
    """

    connected: bool = False
    instrument_id: str = ""
    resource_name: str = ""
    last_error: str = ""


class RSVNAController:
    """
    Small wrapper around an R&S ZVA/ZVB/ZVT vector network analyzer.

    Intended first target:
        R&S ZVA24 over Ethernet/LAN.

    Communication:
        RsInstrument over VISA LAN.

    Typical VISA resource:
        TCPIP::192.168.0.10::INSTR
        TCPIP0::192.168.0.10::inst0::INSTR
    """

    def __init__(
        self,
        ip_address: str = "192.168.0.10",
        resource_name: str | None = None,
        timeout_ms: int = 60_000,
        reset_on_connect: bool = False,
        id_query_on_connect: bool = True,
        simulate: bool = False,
    ) -> None:
        """
        Initialize the VNA controller wrapper.

        Parameters
        ----------
        ip_address:
            IP address of the VNA.

        resource_name:
            Optional full VISA resource string.
            If None, TCPIP::{ip_address}::INSTR is used.

        timeout_ms:
            VISA timeout in milliseconds.

        reset_on_connect:
            If True, sends *RST during connection.
            Usually keep False if you want to preserve saved settings.

        id_query_on_connect:
            If True, checks *IDN? when connecting.

        simulate:
            If True, no hardware communication is attempted.
        """
        self.ip_address = str(ip_address)
        self.resource_name = (
            str(resource_name)
            if resource_name is not None
            else f"TCPIP::{self.ip_address}::INSTR"
        )

        self.timeout_ms = int(timeout_ms)
        self.reset_on_connect = bool(reset_on_connect)
        self.id_query_on_connect = bool(id_query_on_connect)
        self.simulate = bool(simulate)

        self.instrument = None
        self.connected = False
        self.instrument_id = ""
        self.last_error = ""

    def connect(self) -> VNAStatus:
        """
        Connect to the VNA.
        """
        if self.connected:
            return self.status()

        if self.simulate:
            self.connected = True
            self.instrument_id = "SIMULATED R&S ZVA24"
            return self.status()

        try:
            from RsInstrument import RsInstrument
        except ImportError as exc:
            raise RuntimeError(
                "RsInstrument is not installed. Install it with:\n"
                "    pip install RsInstrument\n"
                "You may also need R&S VISA or NI-VISA installed."
            ) from exc

        try:
            self.instrument = RsInstrument(
                self.resource_name,
                self.id_query_on_connect,
                self.reset_on_connect,
            )
            self.instrument.visa_timeout = self.timeout_ms

            self.connected = True

            try:
                self.instrument_id = self.query("*IDN?").strip()
            except Exception:
                self.instrument_id = ""

            self.clear_status()
            return self.status()

        except Exception:
            self.connected = False
            self.instrument = None
            raise

    def close(self) -> None:
        """
        Close the VNA connection.
        """
        if self.instrument is not None:
            try:
                self.instrument.close()
            except Exception:
                pass

        self.instrument = None
        self.connected = False

    def status(self) -> VNAStatus:
        """
        Return current VNA connection status.
        """
        return VNAStatus(
            connected=self.connected,
            instrument_id=self.instrument_id,
            resource_name=self.resource_name,
            last_error=self.last_error,
        )

    def write(self, command: str) -> None:
        """
        Send a SCPI command.
        """
        self._require_connected()

        if self.simulate:
            print(f"[SIM VNA WRITE] {command}")
            return

        self.instrument.write(command)

    def query(self, command: str) -> str:
        """
        Send a SCPI query and return the answer as text.
        """
        self._require_connected()

        if self.simulate:
            print(f"[SIM VNA QUERY] {command}")
            if command.strip().upper() == "*IDN?":
                return "Rohde&Schwarz,ZVA24,SIMULATED,0.0"
            if command.strip().upper() == "SYST:ERR?":
                return '0,"No error"'
            return ""

        return str(self.instrument.query(command))

    def clear_status(self) -> None:
        """
        Clear status/event registers and error queue.
        """
        self.write("*CLS")

    def check_error(self) -> str:
        """
        Query the instrument error queue once.
        """
        try:
            error = self.query("SYST:ERR?").strip()
        except Exception as exc:
            error = repr(exc)

        self.last_error = error
        return error

    def wait_operation_complete(self, timeout_ms: int | None = None) -> None:
        """
        Wait until the current operation is complete.

        Uses *OPC?, which is the standard SCPI operation-complete query.
        """
        self._require_connected()

        if self.simulate:
            return

        old_timeout = self.instrument.visa_timeout

        if timeout_ms is not None:
            self.instrument.visa_timeout = int(timeout_ms)

        try:
            answer = self.query("*OPC?").strip()
            if answer not in {"1", "+1"}:
                raise RuntimeError(f"Unexpected *OPC? response: {answer!r}")
        finally:
            self.instrument.visa_timeout = old_timeout

    def load_state(self, state_path_on_instrument: str, slot: int = 1) -> None:
        """
        Load a saved instrument state from the VNA filesystem.

        Important:
        The path must be visible to the VNA, for example a file already saved
        on the analyzer's internal disk or a USB drive connected to the analyzer.

        Example:
            C:\\R_S\\Instr\\user\\my_setup.zvx

        Depending on the exact firmware/file type, the ZVA may expect a state
        file created from the analyzer's own save/recall function.
        """
        self._require_connected()

        path = self._quote_path(state_path_on_instrument)

        # Common R&S SCPI form for recalling an instrument state.
        # If your ZVA firmware expects a slightly different file type or syntax,
        # this is the one line to adjust.
        self.write(f"MMEM:LOAD:STAT {int(slot)}, {path}")
        self.wait_operation_complete(timeout_ms=self.timeout_ms)

        error = self.check_error()
        if not error.startswith("0"):
            raise RuntimeError(f"VNA reported error after loading state: {error}")

    def trigger_single_sweep(
        self,
        channel: int = 1,
        wait: bool = True,
        timeout_ms: int | None = None,
    ) -> None:
        """
        Trigger one single sweep/measurement.

        This uses a common VNA SCPI pattern:
            INITiate<channel>:CONTinuous OFF
            INITiate<channel>:IMMediate
            *OPC?

        The measurement is triggered by software over Ethernet.
        """
        self._require_connected()

        channel = int(channel)

        self.write(f"INIT{channel}:CONT OFF")
        self.write(f"INIT{channel}:IMM")

        if wait:
            self.wait_operation_complete(timeout_ms=timeout_ms or self.timeout_ms)

        error = self.check_error()
        if not error.startswith("0"):
            raise RuntimeError(f"VNA reported error after trigger: {error}")

    def continuous_sweep(self, enabled: bool = True, channel: int = 1) -> None:
        """
        Enable or disable continuous sweeping.
        """
        self._require_connected()

        channel = int(channel)
        value = "ON" if enabled else "OFF"

        self.write(f"INIT{channel}:CONT {value}")

    def read_trace_data(
        self,
        channel: int = 1,
        trace: int = 1,
        data_format: VNADataFormat = "sdata",
    ) -> list[float]:
        """
        Read trace data from the VNA.

        data_format:
            "sdata" -> complex data as interleaved real/imag values.
            "fdata" -> formatted trace data, usually what is displayed.

        Returns
        -------
        list[float]
            Numeric values returned by the VNA.
        """
        self._require_connected()

        channel = int(channel)
        trace = int(trace)

        # Select trace if supported by current setup.
        # If your setup uses trace names instead of numeric traces,
        # this may need to be adapted.
        try:
            self.write(f"CALC{channel}:PAR:MNUM {trace}")
        except Exception:
            pass

        fmt = data_format.lower().strip()
        if fmt == "sdata":
            command = f"CALC{channel}:DATA? SDATA"
        elif fmt == "fdata":
            command = f"CALC{channel}:DATA? FDATA"
        else:
            raise ValueError("data_format must be 'sdata' or 'fdata'.")

        if self.simulate:
            return []

        raw = self.query(command).strip()

        if not raw:
            return []

        return [float(x) for x in raw.replace("\n", "").split(",") if x.strip()]

    def save_touchstone(
        self,
        output_path_on_instrument: str,
        ports: tuple[int, ...] = (1, 2),
    ) -> None:
        """
        Save S-parameter data to a Touchstone file on the VNA filesystem.

        This is intentionally conservative; exact syntax can vary by firmware
        and by how the active channels/traces are configured.
        """
        self._require_connected()

        path = self._quote_path(output_path_on_instrument)

        # Common R&S style command for storing Touchstone/S-parameter data.
        # If your firmware complains, we can adapt this after checking SYST:ERR?.
        port_list = ",".join(str(int(p)) for p in ports)
        self.write(f"MMEM:STOR:TRAC:PORT {path}, {port_list}")
        self.wait_operation_complete(timeout_ms=self.timeout_ms)

        error = self.check_error()
        if not error.startswith("0"):
            raise RuntimeError(f"VNA reported error after saving Touchstone: {error}")

    def _require_connected(self) -> None:
        """
        Raise if the VNA is not connected.
        """
        if not self.connected:
            raise RuntimeError("VNA is not connected.")

    @staticmethod
    def _quote_path(path: str) -> str:
        """
        Quote a SCPI file path.
        """
        clean = str(path).replace("/", "\\")
        clean = clean.replace('"', "")
        return f'"{clean}"'

    def __enter__(self) -> RSVNAController:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False