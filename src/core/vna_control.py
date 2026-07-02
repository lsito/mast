from __future__ import annotations

import csv
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class VNAStatus:
    """
    Current VNA connection status.
    """

    connected: bool = False
    instrument_id: str = ""
    ip_address: str = ""
    port: int = 5025
    last_error: str = ""


@dataclass(slots=True)
class VNATraceData:
    """
    Acquired VNA trace data.

    complex_traces is arranged as:
        complex_traces[trace_index][frequency_point]
    """

    trace_names: list[str]
    frequency_hz: list[float]
    complex_traces: list[list[complex]]


class RSVNASocketController:
    """
    Minimal Rohde & Schwarz ZVA/ZVB/ZVT controller using raw SCPI over TCP.

    Target instrument:
        R&S ZVA24 over Ethernet.

    Requirements:
        No VISA.
        No PyVISA.
        No external Python packages.

    Typical workflow:
        1. Connect to the VNA on TCP port 5025.
        2. Optionally trigger a sweep.
        3. Read active trace catalog.
        4. Read frequency axis.
        5. Read complex SDATA for each active trace.
        6. Save a MATLAB-style semicolon CSV locally on the PC.

    Cleanup behavior:
        close() tries to send:
            ABOR
            *CLS
            SYST:LOC

        This helps avoid leaving the VNA stuck in remote/busy state.
    """

    def __init__(
        self,
        ip_address: str = "128.11.11.11",
        port: int = 5025,
        timeout_s: float = 120.0,
        connect_timeout_s: float = 5.0,
        id_query_on_connect: bool = True,
        release_on_close: bool = True,
        release_after_measurement: bool = True,
    ) -> None:
        self.ip_address = str(ip_address)
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.connect_timeout_s = float(connect_timeout_s)
        self.id_query_on_connect = bool(id_query_on_connect)
        self.release_on_close = bool(release_on_close)
        self.release_after_measurement = bool(release_after_measurement)

        self.sock: socket.socket | None = None
        self.connected = False
        self.instrument_id = ""
        self.last_error = ""
        self.last_command = ""

    # ------------------------------------------------------------------
    # Connection and cleanup
    # ------------------------------------------------------------------

    def connect(self) -> VNAStatus:
        """
        Open TCP socket connection to the VNA.
        """
        if self.connected:
            return self.status()

        try:
            self.sock = socket.create_connection(
                (self.ip_address, self.port),
                timeout=self.connect_timeout_s,
            )
            self.sock.settimeout(self.timeout_s)
            self.connected = True

            if self.id_query_on_connect:
                self.instrument_id = self.query("*IDN?", timeout_s=5.0).strip()

            try:
                self.clear_status()
            except Exception:
                pass

            return self.status()

        except Exception:
            self.close(release_to_local=False)
            raise

    def close(self, release_to_local: bool | None = None) -> None:
        """
        Close TCP socket connection.

        By default this also tries to release the VNA cleanly using:
            ABOR
            *CLS
            SYST:LOC

        Set release_to_local=False if you only want to close the socket.
        """
        if release_to_local is None:
            release_to_local = self.release_on_close

        if release_to_local:
            self.release_control(
                abort=True,
                clear_status=True,
                go_local=True,
            )

        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass

        self.sock = None
        self.connected = False

    def status(self) -> VNAStatus:
        """
        Return current VNA status.
        """
        return VNAStatus(
            connected=self.connected,
            instrument_id=self.instrument_id,
            ip_address=self.ip_address,
            port=self.port,
            last_error=self.last_error,
        )

    def safe_write(self, command: str, timeout_s: float = 2.0) -> None:
        """
        Send one SCPI command with a short timeout.

        Intended for cleanup commands. This method never raises.
        """
        if not self.connected or self.sock is None:
            return

        old_timeout = self.sock.gettimeout()

        try:
            self.sock.settimeout(float(timeout_s))
            self.sock.sendall((command.rstrip() + "\n").encode("ascii"))
        except Exception:
            pass
        finally:
            try:
                self.sock.settimeout(old_timeout)
            except Exception:
                pass

    def release_control(
        self,
        abort: bool = True,
        clear_status: bool = True,
        go_local: bool = True,
    ) -> None:
        """
        Try to leave the VNA in a clean, usable state.

        Raw-socket equivalent of:
            abort current operation
            clear error/status queue
            return to local/front-panel control

        This method never raises.
        """
        if abort:
            self.safe_write("ABOR", timeout_s=2.0)
            time.sleep(0.05)

        if clear_status:
            self.safe_write("*CLS", timeout_s=2.0)
            time.sleep(0.05)

        if go_local:
            self.safe_write("SYST:LOC", timeout_s=2.0)
            time.sleep(0.05)

    # ------------------------------------------------------------------
    # Low-level SCPI
    # ------------------------------------------------------------------

    def write(self, command: str) -> None:
        """
        Send one SCPI command.
        """
        self._require_connected()
        self.last_command = command
        self.sock.sendall((command.rstrip() + "\n").encode("ascii"))

    def query(self, command: str, timeout_s: float | None = None) -> str:
        """
        Send one SCPI query and return the ASCII response.
        """
        self._require_connected()

        old_timeout = self.sock.gettimeout()

        if timeout_s is not None:
            self.sock.settimeout(float(timeout_s))

        try:
            self.write(command)
            return self._read_ascii_response()
        except socket.timeout as exc:
            raise TimeoutError(
                f"Timeout while waiting for response to SCPI command: "
                f"{command!r}"
            ) from exc
        finally:
            try:
                self.sock.settimeout(old_timeout)
            except Exception:
                pass

    def clear_status(self) -> None:
        """
        Clear status/event registers and error queue.
        """
        self.write("*CLS")

    def check_error(self) -> str:
        """
        Read one item from the VNA error queue.
        """
        try:
            error = self.query("SYST:ERR?", timeout_s=5.0).strip()
        except Exception as exc:
            error = repr(exc)

        self.last_error = error
        return error

    def prepare_for_remote_measurement(self) -> None:
        """
        Prepare the VNA for a remote readout.

        This intentionally does not reset or preset the instrument,
        because that would destroy the user’s current setup.
        """
        self.clear_status()

    # ------------------------------------------------------------------
    # Triggering
    # ------------------------------------------------------------------

    def configure_immediate_trigger(self, channel: int = 1) -> None:
        """
        Try to configure software/immediate triggering.

        Different R&S firmware versions accept slightly different trigger
        source commands. We send the common variants and do not fail if one
        of them is not understood.
        """
        channel = int(channel)

        commands = [
            "TRIG:SEQ:SOUR IMM",
            "TRIG:SOUR IMM",
            f"INIT{channel}:CONT OFF",
        ]

        for command in commands:
            try:
                self.write(command)
                time.sleep(0.05)
            except Exception:
                pass

        # Do not raise here. One unsupported alias is not fatal.
        try:
            self.last_error = self.query("SYST:ERR?", timeout_s=2.0).strip()
        except Exception:
            pass

    def trigger_single_sweep(
        self,
        channel: int = 1,
        wait_s: float = 2.0,
        use_opc: bool = False,
    ) -> None:
        """
        Trigger one single sweep.

        Default behavior avoids *OPC? because that was causing timeout
        problems on your setup. It uses a fixed wait instead.

        Once the trigger behavior is confirmed, you can try use_opc=True.
        """
        channel = int(channel)

        self.configure_immediate_trigger(channel=channel)
        self.write(f"INIT{channel}:IMM")

        if use_opc:
            answer = self.query("*OPC?", timeout_s=self.timeout_s).strip()
            if answer not in {"1", "+1"}:
                raise RuntimeError(f"Unexpected *OPC? response: {answer!r}")
        else:
            time.sleep(float(wait_s))

        try:
            self.last_error = self.query("SYST:ERR?", timeout_s=5.0).strip()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Trace discovery and readout
    # ------------------------------------------------------------------

    def get_trace_catalog(self, channel: int = 1) -> list[tuple[str, str]]:
        """
        Return the active trace catalog.

        Returns:
            [(trace_name, measurement_name), ...]

        Example:
            [("Trc1", "S11"), ("Trc2", "S21")]
        """
        channel = int(channel)

        raw = self.query(f"CALC{channel}:PAR:CAT?", timeout_s=10.0).strip()

        # Some instruments return the whole catalog quoted.
        if len(raw) >= 2 and raw[0] in {"'", '"'} and raw[-1] == raw[0]:
            raw = raw[1:-1]

        try:
            items = next(csv.reader([raw]))
        except Exception as exc:
            raise RuntimeError(f"Could not parse trace catalog: {raw!r}") from exc

        items = [item.strip().strip("'").strip('"') for item in items]

        if len(items) < 2:
            raise RuntimeError(f"Could not parse trace catalog: {raw!r}")

        if len(items) % 2 != 0:
            raise RuntimeError(f"Unexpected trace catalog format: {items!r}")

        pairs: list[tuple[str, str]] = []

        for i in range(0, len(items), 2):
            trace_name = items[i]
            measurement_name = items[i + 1]
            pairs.append((trace_name, measurement_name))

        return pairs

    def select_trace(
        self,
        trace_name: str,
        channel: int = 1,
        trace_index: int | None = None,
    ) -> None:
        """
        Select an active trace.

        Prefer numeric trace index when available because it is less sensitive
        to trace-name quoting differences.
        """
        channel = int(channel)

        if trace_index is not None:
            self.write(f"CALC{channel}:PAR:MNUM {int(trace_index)}")
            time.sleep(0.05)
            return

        self.write(f"CALC{channel}:PAR:SEL '{trace_name}'")
        time.sleep(0.05)

    def read_complex_trace(
        self,
        trace_name: str,
        channel: int = 1,
        trace_index: int | None = None,
    ) -> list[complex]:
        """
        Read one trace as complex SDATA.

        Returns:
            [complex(real0, imag0), complex(real1, imag1), ...]
        """
        channel = int(channel)

        self.select_trace(
            trace_name=trace_name,
            channel=channel,
            trace_index=trace_index,
        )

        # ASCII makes parsing simple and avoids binary-block handling.
        self.write("FORM:DATA ASC")

        raw = self.query(f"CALC{channel}:DATA? SDATA", timeout_s=self.timeout_s)
        values = self._parse_float_list(raw)

        if len(values) % 2 != 0:
            raise RuntimeError(
                f"Expected even number of SDATA values, got {len(values)}."
            )

        return [
            complex(values[i], values[i + 1])
            for i in range(0, len(values), 2)
        ]

    def read_frequency_axis(self, channel: int = 1) -> list[float]:
        """
        Read the frequency/stimulus axis.

        First tries:
            CALCulate:DATA:STIMulus?

        If unavailable, falls back to:
            SENS:FREQ:STAR?
            SENS:FREQ:STOP?
            SENS:SWE:POIN?
        """
        channel = int(channel)

        try:
            raw = self.query(f"CALC{channel}:DATA:STIM?", timeout_s=30.0)
            values = self._parse_float_list(raw)

            if values:
                return values
        except Exception:
            pass

        start = float(self.query(f"SENS{channel}:FREQ:STAR?", timeout_s=10.0).strip())
        stop = float(self.query(f"SENS{channel}:FREQ:STOP?", timeout_s=10.0).strip())
        points = int(float(self.query(f"SENS{channel}:SWE:POIN?", timeout_s=10.0).strip()))

        if points <= 1:
            return [start]

        step = (stop - start) / (points - 1)
        return [start + i * step for i in range(points)]

    def acquire_trace_data(
        self,
        channel: int = 1,
        trigger: bool = False,
        trigger_wait_s: float = 2.0,
        trigger_use_opc: bool = False,
        use_measurement_names: bool = True,
    ) -> VNATraceData:
        """
        Acquire all active traces on one channel.

        This is the Python/socket equivalent of the MATLAB idea:

            [sname, smat, trace_num, ff] = rs_vna_getdata(...)

        Default trigger=False is intentional. First verify that reading the
        current displayed data works. Then enable trigger=True.
        """
        channel = int(channel)

        if trigger:
            self.trigger_single_sweep(
                channel=channel,
                wait_s=trigger_wait_s,
                use_opc=trigger_use_opc,
            )

        trace_catalog = self.get_trace_catalog(channel=channel)

        if not trace_catalog:
            raise RuntimeError("No traces found on the VNA.")

        frequency_hz = self.read_frequency_axis(channel=channel)

        trace_names: list[str] = []
        complex_traces: list[list[complex]] = []

        for index, (trace_name, measurement_name) in enumerate(trace_catalog, start=1):
            label = measurement_name if use_measurement_names else trace_name

            trace_data = self.read_complex_trace(
                trace_name=trace_name,
                channel=channel,
                trace_index=index,
            )

            if frequency_hz and len(trace_data) != len(frequency_hz):
                raise RuntimeError(
                    f"Trace {trace_name!r} has {len(trace_data)} points, "
                    f"but frequency axis has {len(frequency_hz)} points."
                )

            trace_names.append(label)
            complex_traces.append(trace_data)

        return VNATraceData(
            trace_names=trace_names,
            frequency_hz=frequency_hz,
            complex_traces=complex_traces,
        )

    # ------------------------------------------------------------------
    # CSV writing
    # ------------------------------------------------------------------

    def save_matlab_style_csv(
        self,
        filename: str | Path,
        channel: int = 1,
        trigger: bool = False,
        trigger_wait_s: float = 2.0,
        trigger_use_opc: bool = False,
        use_measurement_names: bool = True,
        header_source: str = "#Read via matlab",
        release_after: bool | None = None,
    ) -> Path:
        """
        Acquire data and save a MATLAB-style semicolon CSV locally.

        Output format:

            #Read via matlab
            #
            trigger;Re(S11);Im(S11);Re(S21);Im(S21)
            frequency;real;imag;real;imag;...

        release_after:
            If True, sends *CLS and SYST:LOC at the end.
            If the acquisition fails, sends ABOR, *CLS, SYST:LOC.
        """
        if release_after is None:
            release_after = self.release_after_measurement

        try:
            self.prepare_for_remote_measurement()

            data = self.acquire_trace_data(
                channel=channel,
                trigger=trigger,
                trigger_wait_s=trigger_wait_s,
                trigger_use_opc=trigger_use_opc,
                use_measurement_names=use_measurement_names,
            )

            saved_path = self.write_matlab_style_csv(
                filename=filename,
                data=data,
                header_source=header_source,
            )

            if release_after:
                self.release_control(
                    abort=False,
                    clear_status=True,
                    go_local=True,
                )

            return saved_path

        except Exception:
            if release_after:
                self.release_control(
                    abort=True,
                    clear_status=True,
                    go_local=True,
                )
            raise

    def write_matlab_style_csv(
        self,
        filename: str | Path,
        data: VNATraceData,
        header_source: str = "#Read via matlab",
    ) -> Path:
        """
        Write already-acquired VNA data to a MATLAB-style semicolon CSV.

        This method does not communicate with the VNA.
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        if not data.trace_names or not data.complex_traces:
            raise RuntimeError("No VNA trace data to write.")

        n_points = len(data.complex_traces[0])

        for name, trace in zip(data.trace_names, data.complex_traces):
            if len(trace) != n_points:
                raise RuntimeError(
                    f"Trace {name!r} length mismatch: "
                    f"{len(trace)} != {n_points}"
                )

        if data.frequency_hz and len(data.frequency_hz) != n_points:
            raise RuntimeError(
                f"Frequency length mismatch: "
                f"{len(data.frequency_hz)} != {n_points}"
            )

        with filename.open("w", newline="") as file:
            file.write(f"{header_source}\r\n")
            file.write("#  \r\n")

            header = "trigger"
            for name in data.trace_names:
                header += f";Re({name});Im({name})"
            header += "\r\n"
            file.write(header)

            for point_index in range(n_points):
                frequency = (
                    data.frequency_hz[point_index]
                    if data.frequency_hz
                    else float(point_index)
                )

                row = f"{frequency:19.15e};"

                for trace in data.complex_traces:
                    value = trace[point_index]
                    row += f"{value.real: 19.15e};{value.imag: 19.15e};"

                row += "\r\n"
                file.write(row)

        return filename

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_ascii_response(self) -> str:
        """
        Read an ASCII SCPI response.

        Reads until newline. If the VNA sends a large ASCII response, this
        keeps reading chunks until the final newline arrives.

        If a timeout occurs after some data has already arrived, the partial
        response is returned. This helps with instruments that omit the final LF.
        """
        self._require_connected()

        chunks: list[bytes] = []

        while True:
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                if chunks:
                    break
                raise

            if not chunk:
                break

            chunks.append(chunk)

            if chunk.endswith(b"\n"):
                break

        return b"".join(chunks).decode("ascii", errors="replace")

    @staticmethod
    def _parse_float_list(answer: str) -> list[float]:
        """
        Parse comma-separated SCPI float data.
        """
        answer = answer.strip()

        if not answer:
            return []

        return [
            float(item)
            for item in answer.replace("\n", "").split(",")
            if item.strip()
        ]

    def _require_connected(self) -> None:
        if not self.connected or self.sock is None:
            raise RuntimeError("VNA is not connected.")

    def __enter__(self) -> RSVNASocketController:
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.close(release_to_local=True)
        return False