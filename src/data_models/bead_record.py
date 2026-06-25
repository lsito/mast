from dataclasses import dataclass, field
from typing import Any, Optional

from pathlib import Path

import numpy as np
import pandas as pd
import re
from scipy.constants import c as c0
from scipy.interpolate import CubicSpline

from src.data_models.bead_config import BeadpullConfig
from src.data_models.meas_config import MeasurementConfig
from src.data_models.rf_structure import RFStructureParams

@dataclass(slots=True)
class BeadpullRecord:
    """
    Data container for one bead-pull measurement.

    The stored attribute names follow `Calculations_test.ipynb`.

    This class stores data and results only. The calculation logic should live
    in `BeadPullAnalyzer`.
    """

    # Quantities I need from configuration for computations
    RF_params: RFStructureParams
    Meas_params: MeasurementConfig
    BP_options: BeadpullConfig

    # Filename of the BP measurement in the format BP_xxxx.xx_yy.ydeg.csv
    filename: Optional[str] = None

    # Parsing the 
    @property
    def BP_filename_data(self) -> dict[str, float]:
        """
        Parse bead-pull frequency and temperature from `filename`.

        Returns
        -------
        dict[str, float]
            Dictionary containing `f0_MHz`, `f0`, and `temperature_degC`.
        """
        if self.filename is None:
            raise ValueError("Cannot parse bead-pull data because filename is None.")


        pattern = (
            r"BP_"
            r"(?P<f0_MHz>[0-9]+(?:\.[0-9]+)?)_"
            r"(?P<temperature_degC>[-+]?[0-9]+(?:\.[0-9]+)?)"
            r"deg\.csv$"
        )

        match = re.search(pattern, self.filename)

        if match is None:
            raise ValueError(
                f"Could not parse bead-pull frequency and temperature from "
                f"filename {self.filename!r}. Expected format like "
                f"'BP_11989.24_19.1deg.csv'."
            )

        f0_MHz = float(match.group("f0_MHz"))
        temperature_degC = float(match.group("temperature_degC"))

        return {
            "f0_MHz": f0_MHz,
            "f0": f0_MHz * 1e6,
            "temperature_degC": temperature_degC,
        }
    
    @property
    def f0(self) -> float:
        """
        Return the bead-pull frequency parsed from the filename in Hz.
        """
        return self.BP_filename_data["f0"]

    @property
    def temperature_degC(self) -> float:
        """
        Return the bead-pull measurement temperature parsed from the filename.
        """
        return self.BP_filename_data["temperature_degC"]

    # Maybe I should update the temperature of the measurement in the Meas_params
    @property
    def f1(self) -> float:
        """
        Return the target/source frequency in Hz.
        """
        return self.Meas_params.target_frequency_mhz * 1e6

    @property
    def DeltaF(self) -> float:
        """
        Return the normalized frequency detuning. ([1], Eq. (3))
        """
        return -2 * (self.f1 - self.f0) / self.f0

    # Group velocity interpolation
    # This is a list of group velocities for each cell, given in the json config
    # file, apparently at the beginning of the cell (0, 1, 2, ..., noc-1).
    # We need vg at the middle of the cell (0.5, 1.5, ..., noc-0.5) for the 
    # calculation of Qe and beta, so we need to interpolate with a cubic spline
    # 0.5   1   1.5   2   2.5   3   ...
    #  |    *    |    *    |    *
    #      vg1       vg2       vg3 

    @property
    def vg(self) -> np.ndarray:
        """
        Return the group velocity array from `RF_params` given in the JSON
        structure configuration file. Notice, this is interpolated once.
        """
        return np.asarray(self.RF_params.vg_, dtype=float)

    @property
    def noc(self) -> int:
        return int(self.RF_params.noc)

    @property
    def x(self) -> np.ndarray:
        """
        Return the original cell-position vector used for interpolation.
        """
        return np.arange(1, self.noc + 1)

    @property
    def x_interp(self) -> np.ndarray:
        """
        Return the interpolated cell-position vector.
        """
        return np.arange(0.5, self.noc + 1 + 0.5)

    @property
    def vg_(self) -> np.ndarray:
        """
        Return the group velocity interpolated for a second time. Needed for Qe.
        """
        spline = CubicSpline(
            self.x,
            self.vg,
            bc_type="not-a-knot",
            extrapolate=True,
            axis=0,
        )

        return spline(self.x_interp)

    # External Q approximated from [1], Eq. (5)
    # This is the phase advance (NOT per cell), given in the json config file
    @property
    def phi0(self) -> float:
        """
        Return the reference phase advance.
        """
        return float(self.RF_params.phi0)

    @property
    def Q0(self) -> np.ndarray:
        """
        Return the unloaded quality factor array.
        """
        return np.asarray(self.RF_params.Q0_, dtype=float)

    # Coupling coefficient beta of each cell from Wangler Chapt.5.5
    # Qe:      Qe1      Qe2      Qe3      Qe4      Qe5
    #           |        |        |        |        |
    # 
    # beta:   beta1    beta2    beta3    beta4
    #         uses     uses     uses     uses
    #         Qe1,Qe2  Qe2,Qe3  Qe3,Qe4  Qe4,Qe5
    # Notice that here the cell is treated as a single input system; hence the power flowing out of the second 
    # port is accounted for as power lost
    # Hence Beta(i) = P_ext(1)/(P0(i)+P_ext(i+1)) = 1/Qe(i) / (1/Q0(i) + 1/Qe(i+1))

    @property
    def Qe(self) -> np.ndarray:
        """
        Return the approximated external quality factor array.
        The returned array has length `noc + 1`, because `vg_` is evaluated on the
        half-cell shifted grid.
        """
        return c0 * self.phi0 / self.vg_

    @property
    def Qe_i(self) -> np.ndarray:
        """
        Return the left-side external-Q array.
        It has length `noc`.
        """
        return np.asarray(self.Qe[:-1], dtype=float)

    @property
    def Qe_ip1(self) -> np.ndarray:
        """
        Return the right-side external-Q array.
        """
        return np.asarray(self.Qe[1:], dtype=float)

    @property
    def beta(self) -> np.ndarray:
        """
        Return the coupling coefficient of each cell.
        Each cell is treated as a single-input system where the power flowing out
        of the second port is accounted for as power loss.
        """
        return (1 / self.Qe_i) / (1 / self.Q0 + 1 / self.Qe_ip1)

    @property
    def gamma_num(self) -> np.ndarray:
        return (
            (self.beta - 1) * (self.beta + 1)
            - self.Qe_i**2 * self.DeltaF**2
            - 1j * 2 * self.beta * self.Qe_i * self.DeltaF
        )

    @property
    def gamma_den(self) -> np.ndarray:
        return (self.beta + 1) ** 2 + self.Qe_i**2 * self.DeltaF**2

    @property
    def gamma(self) -> np.ndarray:
        """
        Return the input-port reflection coefficient.
        It corresponds to the reflection from the input port from [1], Eq. (1), 
        with the same assumptions used in the notebook.
        """
        return self.gamma_num / self.gamma_den

    @property
    def v_particles(self) -> float:
        return float(self.RF_params.v_particles)

    @property
    def alpha(self) -> np.ndarray:
        """
        This is [1], Eq. (19) before converting it into a cumulative exponential
        attenuation factor.
        """
        return (self.v_particles * self.phi0) / (self.Q0 * self.vg)

    @property
    def att(self) -> np.ndarray:
        """
        Return the cumulative attenuation factor.
        The returned array has length `noc + 1`, matching the forward/backward
        wave arrays `A` and `B`.
        """
        att = np.ones(self.noc + 1)

        att[0] = 1.0
        att[1:self.noc] = np.exp(-np.cumsum(self.alpha[:-1]))
        att[-1] = att[-2] * np.exp(-self.alpha[-1])

        return att

    # Import and Computation from BP file will be done in the main using methods
    # from src/core/beadpull_analyzer.py. The dataclass here acts just as
    # container

    f: Optional[np.ndarray] = None
    scc11: Optional[np.ndarray] = None
    scc21: Optional[np.ndarray] = None
    scc12: Optional[np.ndarray] = None
    scc22: Optional[np.ndarray] = None

    @property
    def file_extension(self) -> str:
        """
        Return the lowercase file extension of `filename`.
        """
        if self.filename is None:
            raise ValueError("Cannot determine file extension because filename is None.")

        return Path(self.filename).suffix.lower()

    ## If the structure was measured with Ports (1, 3) as input, then we take scc11
    ## If the structure was measured with Ports (2, 4) as input, then we take scc22
    @property
    def use_S_output_for_BP(self) -> bool:
        """
        Return the bead-pull signal-selection option.
        """
        return bool(self.BP_options.use_S_output_for_BP)


    aorg: Optional[np.ndarray] = None
    sorg: Optional[np.ndarray] = None

    a_zero_l: Optional[complex] = None
    gamma0: Optional[complex] = None
    a_zero_lr: Optional[np.ndarray] = None
    a_zero: Optional[np.ndarray] = None
    a: Optional[np.ndarray] = None

    zero_region: Optional[np.ndarray] = None
    abs_signal_sorted: Optional[np.ndarray] = None
    reference_amplitude: Optional[float] = None
    relative_zero_residual: Optional[np.ndarray] = None
    relative_zero_sorted: Optional[np.ndarray] = None
    third_largest_residual: Optional[float] = None
    zero_line_passed: Optional[bool] = None

    atp: Optional[np.ndarray] = None
    threshold: Optional[float] = None
    abs_atp_smooth: Optional[np.ndarray] = None
    locpk: Optional[np.ndarray] = None
    pks: Optional[np.ndarray] = None
    dref: Optional[np.ndarray] = None

    phase_tolerance: Optional[float] = None
    phase_diff: Optional[np.ndarray] = None
    idx_double: Optional[np.ndarray] = None

    locpk_raw: Optional[np.ndarray] = None
    dref_raw: Optional[np.ndarray] = None
    pks_raw: Optional[np.ndarray] = None

    nop: Optional[int] = None
    bad_peaks_idx: Optional[np.ndarray] = None
    bad_peaks_number: Optional[np.ndarray] = None

    @property
    def phi(self) -> np.ndarray:
        """
        Return the per-cell phase advance array.
        """
        return np.asarray(self.RF_params.phi, dtype=float)

    phase: Optional[np.ndarray] = None
    phase_peaks: Optional[np.ndarray] = None
    dphase_peaks: Optional[np.ndarray] = None
    Dpp: Optional[np.ndarray] = None
    ffx: Optional[np.ndarray] = None
    dphi_c: Optional[np.ndarray] = None
    Dpp_c: Optional[np.ndarray] = None
    ff: Optional[np.ndarray] = None

    Ebp: Optional[np.ndarray] = None
    phiadv: Optional[np.ndarray] = None
    phimean: Optional[float] = None
    phisig: Optional[float] = None

    @property
    def fref(self) -> float:
        """
        Return the reference frequency.
        """
        return float(self.RF_params.fref)

    d: Optional[float] = None
    rovq: Optional[np.ndarray] = None
    squ: Optional[np.ndarray] = None
    I: Optional[np.ndarray] = None
    A: Optional[np.ndarray] = None
    B: Optional[np.ndarray] = None

    mean_phi_inner: Optional[float] = None
    s11local: Optional[np.ndarray] = None
    ds11local: Optional[np.ndarray] = None
    ds11global: Optional[np.ndarray] = None
    s11local_org: Optional[np.ndarray] = None
    ds11local_dtemp: Optional[np.ndarray] = None
    ds11: Optional[np.ndarray] = None
    df2tune: Optional[np.ndarray] = None

    idx112: Optional[np.ndarray] = None
    ref_end_comp: Optional[np.ndarray] = None
    ref_mean: Optional[complex] = None
    ds11_comp_local_0: Optional[float] = None
    ds11_comp_local_1: Optional[float] = None
    ds11_0: Optional[float] = None
    ds11_1: Optional[float] = None

    info: dict[str, Any] = field(default_factory=dict)

    @property
    def rovq_(self) -> np.ndarray:
        """
        Return the base R/Q array from `RF_params`.
        """
        return np.asarray(self.RF_params.rovq_, dtype=float)

    @property
    def has_tuning_results(self) -> bool:
        """
        Return True if the main tuning quantities have been computed.
        """
        return self.ds11 is not None and self.df2tune is not None

    def summary(self) -> dict[str, Any]:
        """
        Return a compact summary of the bead-pull result.
        """
        return {
            "filename": self.filename,
            "f0": self.f0,
            "f1": self.f1,
            "DeltaF": self.DeltaF,
            "temperature_degC": self.temperature_degC,
            "noc": self.noc,
            "nop": self.nop,
            "phimean": self.phimean,
            "phisig": self.phisig,
            "ds11_0": self.ds11_0,
            "ds11_1": self.ds11_1,
            "has_tuning_results": self.has_tuning_results,
        }

    def tuning_dataframe(self) -> pd.DataFrame:
        """
        Return the main per-cell tuning quantities as a DataFrame.
        """
        if self.ds11 is not None:
            n = len(self.ds11)
        elif self.df2tune is not None:
            n = len(self.df2tune)
        elif self.dref is not None:
            n = len(self.dref)
        else:
            return pd.DataFrame()

        data: dict[str, Any] = {
            "cell": np.arange(1, n + 1),
        }

        def add_array_column(name: str, values: Optional[np.ndarray]) -> None:
            """
            Add an array column when the data exists and is long enough.
            """
            if values is None:
                return

            values_arr = np.asarray(values)

            if len(values_arr) >= n:
                data[name] = values_arr[:n]

        add_array_column("locpk", self.locpk)
        add_array_column("pks", self.pks)
        add_array_column("dref_real", None if self.dref is None else np.real(self.dref))
        add_array_column("dref_imag", None if self.dref is None else np.imag(self.dref))
        add_array_column("Ebp_abs", None if self.Ebp is None else np.abs(self.Ebp))
        add_array_column("Ebp_real", None if self.Ebp is None else np.real(self.Ebp))
        add_array_column("Ebp_imag", None if self.Ebp is None else np.imag(self.Ebp))
        add_array_column("s11local_real", None if self.s11local is None else np.real(self.s11local))
        add_array_column("s11local_imag", None if self.s11local is None else np.imag(self.s11local))
        add_array_column("ds11local_real", None if self.ds11local is None else np.real(self.ds11local))
        add_array_column("ds11local_imag", None if self.ds11local is None else np.imag(self.ds11local))
        add_array_column("ds11global_real", None if self.ds11global is None else np.real(self.ds11global))
        add_array_column("ds11global_imag", None if self.ds11global is None else np.imag(self.ds11global))
        add_array_column("ds11", self.ds11)
        add_array_column("df2tune", self.df2tune)

        return pd.DataFrame(data)