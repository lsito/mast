
from dataclasses import dataclass


@dataclass
class BeadpullOptions:
    """
    Flags controlling how the beadpull measurement is interpreted.
    Corresponds to MATLAB's bdata.options nested struct.
    """
    use_S_output_for_BP: Optional[bool] = None
    invert_RF_structure_parameters: Optional[bool] = None
    invert_measurement_direction: Optional[bool] = None


@dataclass
class BeadpullRecord:
    """
    Complete data record for a single beadpull measurement.
    Corresponds to one element of MATLAB's bdata struct array.
    """
    # --- File references ---
    filename: Optional[str] = None          # Raw data file
    file_astr: Optional[str] = None         # .astr structure file

    # --- RF structure parameters (from .astr file) ---
    astr: RFStructureParams = field(default_factory=RFStructureParams)

    # --- Frequency references ---
    freq: Optional[float] = None            # Beadpull measurement frequency
    ftarget: Optional[float] = None         # Target frequency at measurement condition

    # --- Metadata ---
    info: Optional[object] = None

    # --- Raw outputs ---
    poweroutput: Optional[np.ndarray] = None
    bpoutput: Optional[np.ndarray] = None

    # --- Raw S-parameter measurement ---
    sorg: Optional[np.ndarray] = None       # S11, full measurement
    a_zero: Optional[np.ndarray] = None     # Zero-line values (bead retracted)
    gamma0: Optional[np.ndarray] = None     # Zero-line at reference bead position

    # --- Perturbation signal ---
    ssub: Optional[np.ndarray] = None       # S11 perturbation due to bead

    # --- Beadpull results ---
    locpk: Optional[np.ndarray] = None      # Peak positions (cell locations)
    ebp: Optional[np.ndarray] = None        # E-field max amplitude per cell
    phiadv: Optional[np.ndarray] = None     # Phase advance between cells
    phimean: Optional[float] = None         # Mean phase advance (excl. matching cell)
    phisig: Optional[float] = None          # Std dev of phase advance

    # --- Wave decomposition ---
    wfn: Optional[np.ndarray] = None        # Forward wave (normalized)
    wbn: Optional[np.ndarray] = None        # Backward wave (normalized)
    dsn: Optional[np.ndarray] = None
    dkn: Optional[np.ndarray] = None
    dun: Optional[np.ndarray] = None

    # --- Local S11 and temperature sensitivity ---
    s11local: Optional[np.ndarray] = None           # Local S11 per cell
    ds11local_dtemp: Optional[np.ndarray] = None    # dS11_local / dT

    # --- Global tuning quantities ---
    ds11: Optional[np.ndarray] = None       # Delta S11 for tuning (>0 push, <0 pull)
    alpha: Optional[np.ndarray] = None      # Damping: local -> global S11
    df2tune: Optional[np.ndarray] = None    # Delta freq for tuning (>0 push, <0 pull)

    # --- Frequency targets ---
    f0: Optional[float] = None              # Beadpull frequency
    f1: Optional[float] = None              # Target lab frequency

    # --- Tuning state ---
    est: Optional[object] = None
    s11local_org: Optional[np.ndarray] = None
    ref_end_comp: Optional[object] = None
    ds11_0: Optional[np.ndarray] = None
    ds11_1: Optional[np.ndarray] = None

    # --- Measurement options ---
    options: BeadpullOptions = field(default_factory=BeadpullOptions)
