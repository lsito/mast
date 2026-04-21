import pandas as pd
import numpy as np
import skrf as rf

from pathlib import Path

def read_csv(filename: str | Path, sep: str | None = None):
    
    # Read CSV into a DataFrame
    if sep is None:
        df = pd.read_csv(filename, sep=None, engine="python", comment="#", index_col=False)
    else:
        df = pd.read_csv(filename, sep=sep, comment="#", index_col=False)

    # Keep only numeric columns
    # df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna(axis=0, how="any")

    
    f = df.iloc[:, 0].to_numpy(dtype=float)
    x = df.iloc[:, 1:].to_numpy(dtype=float)

    if x.shape[1] % 2 != 0:
        raise ValueError("After the frequency column, columns must come in Re/Im pairs")

    n_complex = x.shape[1] // 2
    nports = int(round(np.sqrt(n_complex)))

    if nports * nports != n_complex:
        raise ValueError(
            f"Expected a square number of complex S-parameters, got {n_complex}"
        )

    # Vectorized Re/Im -> complex
    s_flat = x[:, 0::2] + 1j * x[:, 1::2]

    # Reshape to scikit-rf format: (n_freq, n_ports, n_ports)
    s = s_flat.reshape(len(f), nports, nports)

    freq = rf.Frequency.from_f(f, unit="Hz")
    ntwk = rf.Network(frequency=freq, s=s)

    f = ntwk.f # Should be already in [Hz], but check just in case
            # Also note that in zero-span measurements, this is time [s] or samples

    # Check the renumbering; might be a good idea to pass is 
    ntwk.renumber([0, 1, 2, 3], [0, 2, 1, 3]) # Reorder ports to match mixed-mode convention
    ntwk.se2gmm(p=2)

    scc11 = ntwk.s[:, 2, 2]
    scc21 = ntwk.s[:, 3, 2]
    scc12 = ntwk.s[:, 2, 3]
    scc22 = ntwk.s[:, 3, 3]

    return f, scc11, scc21, scc12, scc22
