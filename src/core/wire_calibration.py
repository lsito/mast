from __future__ import annotations
from pathlib import Path
import numpy as np

from src.io_utils.s4p import read_s4p


def calculate_wire_calibration(
    file_without_wire: str,
    file_with_wire: str,
    center_mhz: float,
    span_mhz: float,
    sflag: str = "S21",
    formatflag: str = "phase",
):
    """
    Read two S4P files with the existing project reader and run wire calibration.
    """

    f_no, s_no = read_s4p_as_dict(file_without_wire)
    f_wire, s_wire = read_s4p_as_dict(file_with_wire)

    fop = float(center_mhz) * 1e6
    fsp = float(span_mhz) * 1e6

    idx = (f_no > fop - fsp / 2) & (f_no < fop + fsp / 2)

    if not np.any(idx):
        raise ValueError("No frequency points found inside the selected span.")

    sflag = sflag.upper().strip()

    if sflag not in s_no:
        raise ValueError(f"{sflag} is not available. Use S11, S21, S12, or S22.")

    df_removewire, ff, ss, pf, ps = cal_wire(
        f_no[idx],
        s_no[sflag][idx],
        f_wire[idx],
        s_wire[sflag][idx],
        formatflag,
    )

    px = ff / 1e9
    py = np.unwrap(np.angle(ss), axis=0) * 180.0 / np.pi

    pxx = pf / 1e9
    pyy = np.unwrap(np.angle(ps), axis=0) * 180.0 / np.pi

    if len(pxx) > 0:
        ua_ps1 = np.zeros_like(pyy)

        for nn in range(2):
            ua_ps1[:, nn] = np.interp(pxx[:, nn], px[:, nn], py[:, nn])

        dphi = 360.0 * np.ones((pyy.shape[0], 1)) * np.round(
            np.mean(pyy - ua_ps1, axis=0) / 360.0
        )

        pyy_aligned = pyy - dphi
    else:
        pyy_aligned = pyy

    return {
        "df_removewire": df_removewire,
        "ff": ff,
        "ss": ss,
        "pf": pf,
        "ps": ps,
        "px": px,
        "py": py,
        "pxx": pxx,
        "pyy": pyy,
        "pyy_aligned": pyy_aligned,
        "fop": fop,
        "fsp": fsp,
        "sflag": sflag,
        "formatflag": formatflag,
    }


def read_s4p_as_dict(filename: str):
    """
    Use the existing project read_s4p function and expose the returned arrays by name.
    """
    f, scc11, scc21, scc12, scc22 = read_s4p(filename)

    return np.asarray(f, dtype=float), {
        "S11": np.asarray(scc11, dtype=np.complex128),
        "S21": np.asarray(scc21, dtype=np.complex128),
        "S12": np.asarray(scc12, dtype=np.complex128),
        "S22": np.asarray(scc22, dtype=np.complex128),
    }


def cal_wire(f_no, s_no, f_wire, s_wire, formatflag="phase"):
    """
    Python version of the MATLAB cal_wire function.

    Inputs are already the selected S-parameter, for example S21.
    """
    f_no = np.asarray(f_no, dtype=float).reshape(-1)
    f_wire = np.asarray(f_wire, dtype=float).reshape(-1)

    snw = np.asarray(s_no, dtype=np.complex128).reshape(-1)
    sww = np.asarray(s_wire, dtype=np.complex128).reshape(-1)

    if len(f_no) != len(f_wire) or np.any(f_wire != f_no):
        raise ValueError("not same frequency, do not support yet")

    ff = np.column_stack((f_no, f_wire))
    ss = np.column_stack((snw, sww))

    if formatflag.lower() == "phase":
        val1 = np.unwrap(np.angle(sww))
        val2 = np.unwrap(np.angle(snw))
        ther_d = 0.008
    elif formatflag.lower() == "mag":
        val1 = np.abs(sww)
        val2 = np.abs(snw)
        ther_d = 0.00005
    else:
        raise ValueError("formatflag must be 'phase' or 'mag'.")

    if len(f_wire) < 5:
        return np.array([]), ff, ss, ff[:0], ss[:0]

    df_step = (f_wire[-1] - f_wire[0]) / (len(f_wire) - 1)

    if df_step == 0:
        return np.array([]), ff, ss, ff[:0], ss[:0]

    dag1 = (val1[1:] - val1[:-1]) / df_step * 1e6
    dag2 = (val2[1:] - val2[:-1]) / df_step * 1e6

    daguse = 0.25 * (
        dag1[1:]
        + dag1[:-1]
        + dag2[1:]
        + dag2[:-1]
    )

    ag1use = val1[2:-2]
    ag2use = val2[2:-2]

    idx_good = (
        np.abs(daguse[2:] - daguse[1:-1]) < ther_d
    ) & (
        np.abs(daguse[1:-1] - daguse[:-2]) < ther_d
    )

    ff_use = ff[2:2 + len(idx_good), :]
    ss_use = ss[2:2 + len(idx_good), :]
    daguse_for_df = daguse[:len(idx_good)]

    pf = ff_use[idx_good, :]
    ps = ss_use[idx_good, :]

    df_removewire = (
        (ag1use[idx_good] - ag2use[idx_good])
        / daguse_for_df[idx_good]
        * 1e6
    )

    return df_removewire, ff, ss, pf, ps