from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from scipy.constants import c
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
    filename: Optional[str] = None      # JSON filename/filepath

    # Overall cells
    noc: int = None           # Number of cells
    NIn: int = 1           # Number of input cells
    NOut: int = noc          # Number of output cells
    
    fref: Optional[float] = None        # Reference frequency in Hz
    phi0: Optional[float] = None        # Reference phase in degrees
    coupling: Optional[float] = 1   # Input/output coupling factor

    # RF Params per cell (np.ndarrays are Q0, rovq, and vg; length noc-2)
    Q0: Optional[np.ndarray] = None          # Unloaded Q factor
    phi: Optional[float] = None    # Phase per cell
    rovq: Optional[np.ndarray] = None        # R/Q in Ohms

    c0: Optional[float] = c          # Speed of light (or scaling)
    v_particles: Optional[float] = 1 * c0 # Particle velocity
    vg: Optional[np.ndarray] = None          # Group velocity

    desc: Optional[str] = None          # Structure description
    ans: Optional[object] = None        # Generic answer/result field