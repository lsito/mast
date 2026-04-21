# Load the 4-port S-parameter data and convert to mixed-mode S-parameters. The
# goal is to have a 2x2 matix in output.
# Note: Port mapping is critical. The typical configuration used is:
# Ports 1, 3: Input pair (combined)
# Ports 2, 4: Output pair (combined)
# More can be found at: https://scikit-rf.readthedocs.io/en/latest/examples/mixedmodeanalysis/Mixed%20Mode%20Basics.html

import skrf as rf
from pathlib import Path

def read_s4p(filename: str | Path, renumber_ports_list: list[int] = [0, 2, 1, 3]):

    ntwk = rf.Network(filename)

    if ntwk.nports != 4:
        raise ValueError(f"Expected a 4-port network, got {ntwk.nports}-port")

    f = ntwk.f # Should be already in [Hz], but check just in case
               # Also note that in zero-span measurements, this is time [s]

    # Check the renumbering; might be a good idea to pass is 
    ntwk.renumber(renumber_ports_list) # Reorder ports to match mixed-mode convention
    ntwk.se2gmm(p=2)

    scc11 = ntwk.s[:, 2, 2]
    scc21 = ntwk.s[:, 3, 2]
    scc12 = ntwk.s[:, 2, 3]
    scc22 = ntwk.s[:, 3, 3]

    return f, scc11, scc21, scc12, scc22
