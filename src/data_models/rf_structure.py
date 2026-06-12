from dataclasses import dataclass, field
from typing import Any, Optional
import copy
import numpy as np

from scipy.constants import c as c0
from scipy.interpolate import CubicSpline
from src.core.config_parser import DataMixin

@dataclass(slots=True)
class RFStructureParams(DataMixin): # We use inheritance here
    """
    Container describing the RF structure parameters used throughout the
    bead-pull and tuning analysis workflow.

    This dataclass stores both full-structure properties and per-cell RF
    quantities loaded from a structure definition file (typically JSON).
    The data loading methods are inherited from the DataMixin class.
    The parameters are used by the core evaluation routines.

    Notes
    -----
    - Frequencies are expressed in Hz unless otherwise stated.
    - Angles/phases are expressed in degrees.
    - ``slots=True`` is enabled to reduce memory usage and prevent accidental
      dynamic attribute creation.
    - The class inherits from ``DataMixin`` to get utility methods for JSON data
      loading.

    Attributes
    ----------
    filename : str | None
        Path to the JSON structure-definition file from which the parameters
        were loaded.

    noc : int
        Total number of cells in the RF structure (including coupling cells).

    NIn : int
        Number of the input-side cells.

    NOut : int
        Number of the output cells. Typically NOut = noc

    fref : float | None
        Reference operating frequency in Hz.

    phi0 : float | None
        Reference phase advance in degrees.

    coupling : float
        Input/output coupling factor.

    Q0 : np.ndarray | None
        Unloaded quality factor per cell. Array length noc-2.

    phi : float | None
        Phase advance per cell in degrees.

    rovq : np.ndarray | None
        R/Q values per cell in Ohms. Array length noc-2.

    c0 : float
        Speed of light in vacuum or equivalent scaling constant.

    v_particles : float
        Particle velocity used in the model.

    vg : np.ndarray | None
        Group velocity per cell. Array length noc-2.

    desc : str | None
        Human-readable description of the RF structure.

    ans : object | None
        Generic container for derived quantities, intermediate results,
        or analysis outputs.
    """

    filename: Optional[str] = None

    # Overall cells
    noc: Optional[int] = None
    NIn: int = 1
    NOut: Optional[int] = None

    # Global RF parameters
    fref: Optional[float] = None
    phi0: Optional[float] = None
    coupling: float = 1.0

    # Boundary phase/cell scaling
    phi_in: Optional[float] = None
    phi_out: Optional[float] = None
    rfac_in: float = 1.0
    rfac_out: float = 1.0

    # Analysis options
    option_inverse: bool = False

    # Per-cell RF parameters
    Q0: Optional[np.ndarray] = None
    rovq: Optional[np.ndarray] = None
    vg: Optional[np.ndarray] = None

    # Particle constants
    v_particles: Optional[float] = None

    # Metadata / results
    desc: Optional[str] = None
    ans: Optional[Any] = None


    @property
    def phi(self) -> np.ndarray:

        phi_in = self.phi_out if self.option_inverse else self.phi_in
        phi_out = self.phi_in if self.option_inverse else self.phi_out

        phi = self.phi0 * np.ones(self.noc - 1)

        phi[0] = phi_in
        phi[-1] = phi_out

        return phi
        
    @property
    def vg_(self) -> np.ndarray:
        # spline interpolation
        x_old = np.arange(1, self.noc - 1)
        x_new = np.arange(0, self.noc)

        y = self._maybe_invert_array(self.vg)
        y = CubicSpline(x_old, y)(x_new)

        return y
    
    @property
    def Q0_(self) -> np.ndarray:
        # spline interpolation
        x_old = np.arange(1, self.noc - 1)
        x_new = np.arange(0, self.noc)

        y = self._maybe_invert_array(self.Q0)
        y = CubicSpline(x_old, y)(x_new)

        return y

    @property
    def rovq_(self) -> np.ndarray:
    # boundary cells

        y = self._maybe_invert_array(self.rovq)

        rfac_in = self.rfac_out if self.option_inverse else self.rfac_in
        rfac_out = self.rfac_in if self.option_inverse else self.rfac_out

        return np.concatenate([
            [y[0] * rfac_in],
            y,
            [y[-1] * rfac_out],
        ])

    def _maybe_invert_array(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)

        if self.option_inverse:
            return arr[::-1]

        return arr