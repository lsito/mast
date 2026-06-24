from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from src.core.config_parser import DataMixin

from dataclasses import dataclass
from typing import Optional

from src.core.config_parser import DataMixin


from dataclasses import dataclass, field

import numpy as np

from src.core.config_parser import DataMixin


from dataclasses import dataclass, field

import numpy as np

from src.core.config_parser import DataMixin


@dataclass(slots=True)
class BeadpullConfig(DataMixin):
    """
    Options used by the bead-pull analysis.

    This class follows the variable names and numerical constants used in
    `Calculations_test.ipynb`.

    Attributes
    ----------
    use_S_output_for_BP:
        Notebook variable `use_S_output_for_BP`.

        If False, the analyzer uses the input-side corrected signal, usually
        `scc11`.

        If True, the analyzer uses the output-side corrected signal, usually
        `scc22`.

    n_zero:
        Notebook variable `n_zero`.

        Number of samples used at the beginning and end of the bead-pull trace
        for zero-line/background estimation and checking.

    max_zero_line_deviation:
        Notebook variable `max_zero_line_deviation`.

        Maximum accepted deviation in the zero-line quality check.

    threshold_fraction:
        Fraction used to define the bead-pull peak extraction threshold.

        The notebook logic is equivalent to:

        `threshold = threshold_fraction * max(abs(atp))`

    smooth_size:
        Smoothing window used before peak extraction.

        The notebook uses `uniform_filter1d(..., size=5)`.

    phase_tolerance:
        Phase tolerance used when checking double peaks.

        The notebook uses `5 * np.pi / 180`.

    remove_peaks:
        Manual list of peaks to remove.

        The notebook initializes this as an empty list.

    verbose:
        Diagnostic verbosity level used by helper functions such as
        `zero_line_check`.
    """

    use_S_output_for_BP: bool = False
    n_zero: int = 30
    max_zero_line_deviation: float = 1e-3
    threshold_fraction: float = 0.15
    smooth_size: int = 5
    phase_tolerance: float = 5 * np.pi / 180
    remove_peaks: list[int] = field(default_factory=list)
    verbose: int = 0