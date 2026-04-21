from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class RFStructureParams:
    """
    Parameters loaded from the .astr file describing the RF structure.
    Corresponds to MATLAB's bdata.astr nested struct.
    """
    NIn: Optional[int] = None           # Number of input cells
    NOut: Optional[int] = None          # Number of output cells
    Q0: Optional[float] = None          # Unloaded Q factor
    ans: Optional[object] = None        # Generic answer/result field
    c0: Optional[float] = None          # Speed of light (or scaling)
    v_particles: Optional[float] = None # Particle velocity
    coupling: Optional[float] = None    # Input/output coupling factor
    desc: Optional[str] = None          # Structure description
    filename: Optional[str] = None      # Source filename
    fref: Optional[float] = None        # Reference frequency
    noc: Optional[int] = None           # Number of cells
    phi: Optional[np.ndarray] = None    # Phase per cell
    phi0: Optional[float] = None        # Reference phase
    rovq: Optional[float] = None        # R/Q (shunt impedance / Q)
    vg: Optional[float] = None          # Group velocity


