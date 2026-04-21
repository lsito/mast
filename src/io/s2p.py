import skrf as rf
from pathlib import Path

def read_s2p(filename: str | Path):

    ntwk = rf.Network(filename)

    if ntwk.nports != 2:
        raise ValueError(f"Expected a 2-port network, got {ntwk.nports}-port")

    f = ntwk.f # Should be already in Hz, but check just in case
               # Also note that in zero-span measurements, this is time [s]

    s11 = ntwk.s[:, 0, 0]
    s21 = ntwk.s[:, 1, 0]
    s12 = ntwk.s[:, 0, 1]
    s22 = ntwk.s[:, 1, 1]

    return f, s11, s21, s12, s22