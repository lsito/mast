from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.constants import c as c0
from scipy.interpolate import CubicSpline

from src.data_models.bead_config import BeadpullConfig
from src.data_models.meas_config import MeasurementConfig
from src.data_models.rf_structure import RFStructureParams
from src.io_utils.csv import read_csv


def read_beadpull_file(self, bdata: BeadpullRecord) -> None:
    """
    Read the bead-pull CSV file and store the corrected S-parameter signals.

    This follows the notebook block:

    `f, scc11, scc21, scc12, scc22 = read_csv(filename)`

    Signal selection is handled by the `BeadpullRecord.aorg` property.
    """
    if bdata.filename is None:
        raise ValueError("Cannot read bead-pull file because filename is None.")

    if bdata.file_extension != ".csv":
        raise NotImplementedError(
            f"Only .csv is supported here, got {bdata.file_extension}"
        )

    f, scc11, scc21, scc12, scc22 = read_csv(bdata.filename)

    bdata.f = f
    bdata.scc11 = scc11
    bdata.scc21 = scc21
    bdata.scc12 = scc12
    bdata.scc22 = scc22


    
#    extract_zero_line()
#    find_beadpull_peaks()
#    extract_phase()
#    compute_forward_backward_waves()
#    compute_local_reflection_and_tuning()