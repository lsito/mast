from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class TuningSimulationResult:
    """
    Result of one tuning simulation refresh.
    """

    dS11_mU: np.ndarray
    wfn_c: np.ndarray
    wbn_c: np.ndarray
    wbn_original: np.ndarray
    ebp_c: np.ndarray
    ebp_original: np.ndarray
    phiadv_c: np.ndarray
    phiadv_original: np.ndarray
    phimean_c: float
    ddphi: float


class TuningSimulatorModel:
    """
    Computation-only tuning simulator.

    This class contains the numerical equivalent of the MATLAB `mw_refresh`
    tuning computation. It has no PySide or Matplotlib dependencies.

    The applied tuning vector `dS11_mU` has one value per cell, in mU.
    The GUI may expose sliders for all cells or only selected cells.
    """

    def __init__(
        self,
        bdata,
        limits_mU: float = 100.0,
        dS11_start=None,
        sliders_for_these_cells=None,
    ) -> None:
        """
        Initialize the tuning simulator model.
        """
        self.bdata = bdata

        self.wfn = self._get_required_array(bdata, "A", "wfn").astype(np.complex128)
        self.wbn = self._get_required_array(bdata, "B", "wbn").astype(np.complex128)
        self.ebp = self._get_required_array(bdata, "Ebp", "ebp").astype(np.complex128)
        self.phiadv = self._get_required_array(bdata, "phiadv").astype(float)

        self.number_of_cells = int(len(self.ebp))
        self.number_of_waves = int(len(self.wfn))

        self.wfn = self._match_complex_length(
            self.wfn,
            self.number_of_cells + 1,
            fill_value=0.0,
        )
        self.wbn = self._match_complex_length(
            self.wbn,
            self.number_of_cells + 1,
            fill_value=0.0,
        )

        self.number_of_waves = int(len(self.wfn))

        self.limit_mU = float(limits_mU)
        self.slider_cell_numbers = self._validated_slider_cell_numbers(
            sliders_for_these_cells
        )

        self.s11local_without_temperature = self._extract_s11local_without_temperature()
        self.phi_design = self._extract_phi_design()
        self.design_phi_advance = self._extract_design_phi_advance()
        self.rovq = self._extract_rovq()
        self.attenuation = self._extract_attenuation()

        self.dS11_mU = self._initial_dS11_values(dS11_start)
        self.dS11_previous_mU = np.full_like(self.dS11_mU, np.inf)

    def _initial_dS11_values(self, dS11_start) -> np.ndarray:
        """
        Return the initialized full dS11 vector in mU.
        """
        if dS11_start is None:
            return np.zeros(self.number_of_cells, dtype=float)

        values = np.asarray(dS11_start, dtype=float).reshape(-1)

        if len(values) != self.number_of_cells:
            return np.zeros(self.number_of_cells, dtype=float)

        return values.copy()

    def _validated_slider_cell_numbers(self, sliders_for_these_cells) -> np.ndarray:
        """
        Return valid one-based cell numbers for slider creation.

        MATLAB accepts something like [0:20] and then removes invalid cell zero.
        This method does the same filtering.
        """
        if sliders_for_these_cells is None:
            return np.arange(1, self.number_of_cells + 1, dtype=int)

        values = np.asarray(sliders_for_these_cells, dtype=int).reshape(-1)

        if len(values) == 0:
            return np.arange(1, self.number_of_cells + 1, dtype=int)

        values = values[
            (values > 0)
            & (values <= self.number_of_cells)
        ]

        if len(values) == 0:
            return np.arange(1, self.number_of_cells + 1, dtype=int)

        return values

    def set_cell_value(self, cell_number: int, value_mU: float) -> None:
        """
        Set one cell's applied dS11 value in mU.

        `cell_number` is one-based, matching the MATLAB slider labels.
        """
        if cell_number < 1 or cell_number > self.number_of_cells:
            return

        self.dS11_mU[cell_number - 1] = float(value_mU)

    def set_dS11_values(self, values_mU) -> None:
        """
        Replace the full dS11 vector in mU.
        """
        values = np.asarray(values_mU, dtype=float).reshape(-1)

        if len(values) != self.number_of_cells:
            values = self._match_real_length(
                values,
                self.number_of_cells,
                fill_value=0.0,
            )

        self.dS11_mU = values.copy()

    def zero_all(self) -> None:
        """
        Reset all applied tuning values to zero.
        """
        self.dS11_mU[:] = 0.0

    def use_record_ds11(self) -> None:
        """
        Initialize the applied dS11 from the bead-pull record's `ds11`.
        """
        ds11 = self._get_optional_array(self.bdata, "ds11")

        if ds11 is None:
            raise ValueError("The selected record does not contain `ds11`.")

        values_mU = np.asarray(ds11, dtype=float).reshape(-1) * 1e3
        self.set_dS11_values(values_mU)

    def use_output_correction(self) -> None:
        """
        Initialize the last two cells from `ds11_0` and `ds11_1`.
        """
        ds11_0 = getattr(self.bdata, "ds11_0", None)
        ds11_1 = getattr(self.bdata, "ds11_1", None)

        if ds11_0 is None or ds11_1 is None:
            raise ValueError("The selected record does not contain `ds11_0` and `ds11_1`.")

        values = np.zeros(self.number_of_cells, dtype=float)

        if self.number_of_cells >= 2:
            values[-2] = float(ds11_0) * 1e3
            values[-1] = float(ds11_1) * 1e3

        self.set_dS11_values(values)

    def simulate(self) -> TuningSimulationResult:
        """
        Compute one tuning simulation.

        This follows the MATLAB logic:

        - start from measured forward wave `wfn`
        - remove temperature contribution from local S11
        - convert applied global dS11 in mU into local imaginary reflection
        - recompute backward wave `wbn_c` backwards from the output
        - recompute E peaks and phase advance
        """
        wfn_c = self.wfn.copy()
        wbn_c = np.zeros_like(self.wbn, dtype=np.complex128)

        s11local_c = self.s11local_without_temperature.copy()

        ds11local_c = 1j * (
            self.dS11_mU
            * 1e-3
            / self.attenuation[:self.number_of_cells]
        )

        s11local_c = s11local_c + ds11local_c

        for matlab_n in range(self.number_of_waves - 1, 1, -1):
            python_i = matlab_n - 1
            phi_i = matlab_n - 2

            if phi_i < len(self.phi_design):
                phase_value = self.phi_design[phi_i]
            else:
                phase_value = self.design_phi_advance

            if python_i < len(s11local_c):
                local_reflection = s11local_c[python_i]
            else:
                local_reflection = s11local_c[-1]

            wbn_c[python_i] = (
                wbn_c[python_i + 1] * np.exp(-1j * phase_value)
                + local_reflection * wfn_c[python_i]
            )

        wbn_c[0] = (
            wbn_c[1] * np.exp(-1j * self.design_phi_advance)
            + s11local_c[0] * wfn_c[0]
        )

        squ_c = wbn_c + wfn_c
        ebp_c = squ_c[:self.number_of_cells] * np.sqrt(
            self.rovq[:self.number_of_cells]
        )

        phiadv_c = -(
            np.angle(ebp_c[1:self.number_of_cells])
            - np.angle(ebp_c[0:self.number_of_cells - 1])
        ) * 180.0 / np.pi

        phiadv_c = np.where(phiadv_c < 0.0, phiadv_c + 360.0, phiadv_c)

        if len(phiadv_c) > 1:
            phimean_c = float(np.mean(phiadv_c[1:]))
        elif len(phiadv_c) > 0:
            phimean_c = float(np.mean(phiadv_c))
        else:
            phimean_c = float("nan")

        if len(self.phiadv) > 0 and len(phiadv_c) > 0:
            ddphi = 360.0 * round(
                (float(np.mean(self.phiadv)) - float(np.mean(phiadv_c))) / 360.0
            )
        else:
            ddphi = 0.0

        self.dS11_previous_mU = self.dS11_mU.copy()

        return TuningSimulationResult(
            dS11_mU=self.dS11_mU.copy(),
            wfn_c=wfn_c,
            wbn_c=wbn_c,
            wbn_original=self.wbn.copy(),
            ebp_c=ebp_c,
            ebp_original=self.ebp.copy(),
            phiadv_c=phiadv_c,
            phiadv_original=self.phiadv.copy(),
            phimean_c=phimean_c,
            ddphi=ddphi,
        )

    def _extract_s11local_without_temperature(self) -> np.ndarray:
        """
        Return local S11 with the temperature detuning removed.
        """
        s11local_org = self._get_optional_array(self.bdata, "s11local_org")

        if s11local_org is not None:
            return self._match_complex_length(
                s11local_org,
                self.number_of_cells,
                fill_value=0.0,
            )

        s11local = self._get_required_array(self.bdata, "s11local").astype(np.complex128)
        s11local = self._match_complex_length(
            s11local,
            self.number_of_cells,
            fill_value=0.0,
        )

        ds11local_dtemp = self._get_optional_array(self.bdata, "ds11local_dtemp")

        if ds11local_dtemp is None:
            return s11local

        ds11local_dtemp = self._match_complex_length(
            ds11local_dtemp,
            self.number_of_cells,
            fill_value=0.0,
        )

        return s11local - ds11local_dtemp

    def _extract_phi_design(self) -> np.ndarray:
        """
        Return the design phase advance array.
        """
        phi = self._get_required_array(self.bdata, "phi").astype(float).reshape(-1)

        target_length = max(self.number_of_cells - 1, 1)

        if len(phi) == 1:
            return np.full(target_length, float(phi[0]), dtype=float)

        return self._match_real_length(
            phi,
            target_length,
            fill_value=float(np.mean(phi)),
        )

    def _extract_design_phi_advance(self) -> float:
        """
        Return the mean design phase advance.
        """
        if len(self.phi_design) > 2:
            return float(np.mean(self.phi_design[1:-1]))

        return float(np.mean(self.phi_design))

    def _extract_rovq(self) -> np.ndarray:
        """
        Return R/Q times cell length.
        """
        rovq = self._get_optional_array(self.bdata, "rovq")

        if rovq is not None:
            return self._match_real_length(
                np.asarray(rovq, dtype=float),
                self.number_of_cells,
                fill_value=1.0,
            )

        rovq_base = self._get_required_array(self.bdata, "rovq_").astype(float)

        phi0 = float(getattr(self.bdata, "phi0"))
        fref = float(getattr(self.bdata, "fref"))
        v_particles = float(getattr(self.bdata, "v_particles"))

        d_cell = phi0 * v_particles / fref / (2.0 * np.pi)
        rovq = rovq_base * d_cell

        return self._match_real_length(
            rovq,
            self.number_of_cells,
            fill_value=1.0,
        )

    def _extract_attenuation(self) -> np.ndarray:
        """
        Return the cumulative attenuation used to convert global dS11 to local dS11.

        In the MATLAB code this variable is named `alpha`, but in the Python
        bead-pull record the cumulative attenuation factor is `att`.
        """
        attenuation = self._get_optional_array(self.bdata, "att")

        if attenuation is not None:
            attenuation = np.asarray(attenuation, dtype=float).reshape(-1)
            attenuation = self._match_real_length(
                attenuation,
                self.number_of_cells,
                fill_value=1.0,
            )
            return np.where(np.abs(attenuation) < 1e-15, 1.0, attenuation)

        alpha = self._get_optional_array(self.bdata, "alpha")

        if alpha is None:
            return np.ones(self.number_of_cells, dtype=float)

        alpha = np.asarray(alpha, dtype=float).reshape(-1)

        if len(alpha) == 0:
            return np.ones(self.number_of_cells, dtype=float)

        if float(np.nanmedian(np.abs(alpha))) < 0.5:
            alpha = self._match_real_length(
                alpha,
                self.number_of_cells,
                fill_value=0.0,
            )

            attenuation = np.ones(self.number_of_cells, dtype=float)

            if self.number_of_cells > 1:
                attenuation[1:] = np.exp(-np.cumsum(alpha[:-1]))

        else:
            attenuation = self._match_real_length(
                alpha,
                self.number_of_cells,
                fill_value=1.0,
            )

        return np.where(np.abs(attenuation) < 1e-15, 1.0, attenuation)

    def _get_required_array(self, bdata, *names: str) -> np.ndarray:
        """
        Return a required array from the bead-pull record.
        """
        value = self._get_optional_array(bdata, *names)

        if value is None:
            names_text = ", ".join(names)
            raise ValueError(f"The selected bead-pull record does not contain: {names_text}")

        return np.asarray(value)

    def _get_optional_array(self, bdata, *names: str):
        """
        Return the first available array from record attributes or `info`.
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

    def _match_real_length(
        self,
        values,
        target_length: int,
        fill_value: float,
    ) -> np.ndarray:
        """
        Resize a real-valued vector to a target length.
        """
        values = np.asarray(values, dtype=float).reshape(-1)

        if len(values) == target_length:
            return values

        if len(values) > target_length:
            return values[:target_length]

        output = np.full(target_length, fill_value, dtype=float)

        if len(values) > 0:
            output[:len(values)] = values
            output[len(values):] = values[-1]

        return output

    def _match_complex_length(
        self,
        values,
        target_length: int,
        fill_value: complex,
    ) -> np.ndarray:
        """
        Resize a complex-valued vector to a target length.
        """
        values = np.asarray(values, dtype=np.complex128).reshape(-1)

        if len(values) == target_length:
            return values

        if len(values) > target_length:
            return values[:target_length]

        output = np.full(target_length, fill_value, dtype=np.complex128)

        if len(values) > 0:
            output[:len(values)] = values
            output[len(values):] = values[-1]

        return output