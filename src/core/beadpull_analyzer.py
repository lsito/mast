from pathlib import Path
from typing import Any, Optional
import warnings

import numpy as np
import pandas as pd
from scipy.constants import c as c0
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from src.data_models.bead_record import BeadpullRecord
from src.io_utils.csv import read_csv

class BeadPullAnalyzer:
    """
    Analysis engine for filling a `BeadpullRecord`.

    This class does not own `RF_params`, `Meas_params`, or `BP_options`.
    Those objects are stored in the `BeadpullRecord`.

    The GUI should create a `BeadpullRecord` when a bead-pull file is loaded,
    then pass it to `evaluate`.
    """

    def evaluate(self, bdata: BeadpullRecord) -> BeadpullRecord:
        """
        Run the full bead-pull workflow and store all results in `bdata`.
        """
        self.read_beadpull_file(bdata)
        self.select_beadpull_signal(bdata)
        self.extract_zero_line(bdata)
        self.check_zero_line(bdata)
        self.find_beadpull_peaks(bdata)
        self.check_number_of_peaks(bdata)
        self.extract_phase(bdata)
        self.compute_forward_backward_waves(bdata)
        self.compute_local_reflection_and_tuning(bdata)
        self.compute_output_matching_correction(bdata)

        return bdata

    ## Signals are already balanced by src.io_utils.csv read_csv method
    def read_beadpull_file(self, bdata: BeadpullRecord) -> None:
        """
        Read the bead-pull CSV file and store the corrected S-parameter signals.
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

    ## If the structure was measured with Ports (1, 3) as input, then we take scc11
    ## If the structure was measured with Ports (2, 4) as input, then we take scc22
    def select_beadpull_signal(self, bdata: BeadpullRecord) -> None:
        """
        Generate aorg and sorg, the beadpull files after selection
        """
        if bdata.scc11 is None or bdata.scc22 is None:
            raise ValueError("Cannot select bead-pull signal before reading the CSV.")

        if not bdata.BP_options.use_S_output_for_BP:
            aorg = bdata.scc11
        else:
            aorg = bdata.scc22

        aorg = np.asarray(aorg, dtype=np.complex128)

        ## Check if the units are not in mU
        if np.max(np.abs(aorg)) > 1:
            aorg = 1e-3 * aorg

        ## Did we select inverse measurement option?
        # Input  -----------------> Output
        # Cell 1 Cell 2 ... Cell 33

        # Output -----------------> Input
        # Cell 33 Cell 32 ... Cell 1
        if bdata.RF_params.option_inverse:
            aorg = aorg[::-1]

        bdata.aorg = aorg
        bdata.sorg = aorg.copy()

    def extract_zero_line(self, bdata: BeadpullRecord) -> None:
        """
        Compute and store `a_zero_l`, `gamma0`, `a_zero_lr`, `a_zero`, and `a`.
        """
        if bdata.aorg is None:
            raise ValueError("Cannot extract zero-line before `aorg` exists.")

        aorg = bdata.aorg
        n_zero = bdata.BP_options.n_zero
        n = len(aorg)

        if n < 2 * n_zero:
            raise ValueError(
                f"`aorg` has only {n} points, but zero-line extraction needs "
                f"at least {2 * n_zero} points."
            )

        # First and last 30 samples
        x_fit = np.concatenate([
                np.arange(n_zero),
                np.arange(n - n_zero, n),
            ])

        y_fit = np.concatenate([
                aorg[:n_zero],
                aorg[-n_zero:],
            ])

        # Global zero-line estimate (needed later)
        a_zero_l = np.mean(aorg[:n_zero])
        gamma0 = a_zero_l

        a_zero_lr = np.mean(y_fit) * np.ones_like(aorg)

        # Linear zero-line fit, 
        # Fit real and imaginary parts separately
        coef_re = np.polyfit(x_fit, y_fit.real, deg=1)
        coef_im = np.polyfit(x_fit, y_fit.imag, deg=1)

        x = np.arange(n)

        a_zero = np.polyval(coef_re, x) + 1j * np.polyval(coef_im, x)

        # Baseline-subtracted bead-pull signal
        a = aorg - a_zero

        # Storing in the dataclass
        bdata.a_zero_l = a_zero_l
        bdata.gamma0 = gamma0
        bdata.a_zero_lr = a_zero_lr
        bdata.a_zero = a_zero
        bdata.a = a

        # Just for some checks
        bdata.info["x_fit"] = x_fit
        bdata.info["y_fit"] = y_fit
        bdata.info["coef_re"] = coef_re
        bdata.info["coef_im"] = coef_im


    def check_zero_line(self, bdata: BeadpullRecord) -> None:
        """
        Check the quality of the zero-line subtraction.
        """
        if bdata.a is None:
            raise ValueError("Cannot check zero-line before `a` has been computed.")

        a = bdata.a
        n_zero = bdata.BP_options.n_zero
        max_zero_line_deviation = bdata.BP_options.max_zero_line_deviation

        zero_region = np.concatenate([
                a[:n_zero],
                a[-n_zero:],
            ])

        # Robust estimate of bead-pull signal amplitude:
        # use the 10%-largest value instead of the absolute maximum,
        # to avoid being dominated by one noisy spike.
        abs_signal_sorted = np.sort(np.abs(a))[::-1]

        idx_10_percent = int(np.floor(len(abs_signal_sorted) * 0.1))
        reference_amplitude = float(abs_signal_sorted[idx_10_percent])

        if reference_amplitude == 0:
            relative_zero_residual = np.full(
                zero_region.shape,
                np.inf,
                dtype=float
            )

            relative_zero_sorted = np.sort(np.abs(relative_zero_residual))[::-1]
            third_largest_residual = float("inf")

        else:
            relative_zero_residual = zero_region / reference_amplitude
            relative_zero_sorted = np.sort(np.abs(relative_zero_residual))[::-1]
            third_largest_residual = float(relative_zero_sorted[2])

        zero_line_passed = bool(third_largest_residual <= max_zero_line_deviation)

        bdata.zero_region = zero_region
        bdata.abs_signal_sorted = abs_signal_sorted
        bdata.reference_amplitude = reference_amplitude
        bdata.relative_zero_residual = relative_zero_residual
        bdata.relative_zero_sorted = relative_zero_sorted
        bdata.third_largest_residual = third_largest_residual
        bdata.zero_line_passed = zero_line_passed

        if not zero_line_passed:
            warnings.warn(
                f'Problem in 0-line fitting, filename="{bdata.filename}". '
                f"Third largest relative residual is "
                f"{third_largest_residual:.6f}, larger than "
                f"{max_zero_line_deviation:.6f}.",
                RuntimeWarning,
                stacklevel=2,
            )

    def find_beadpull_peaks(self, bdata: BeadpullRecord) -> None:
        """
        Compute and store `atp`, `threshold`, `abs_atp_smooth`, `locpk`, `pks`, and `dref`.
        """
        if bdata.a is None:
            raise ValueError("Cannot find peaks before `a` has been computed.")

        # Working on a copy of the signal a (so after zero line)
        a = bdata.a
        atp = a.copy()

        # All values below 15% of the signal maximum amplitude are water-levelled
        threshold = bdata.BP_options.threshold_fraction * np.max(np.abs(atp))
        atp[np.abs(atp) < threshold] = 0

        abs_atp_smooth = uniform_filter1d(
            np.abs(atp),
            size=bdata.BP_options.smooth_size,
            mode="nearest",
        )

        # Find peaks and peaks' locations
        locpk, _ = find_peaks(abs_atp_smooth)
        pks = abs_atp_smooth[locpk]

        # Reference complex values at the detected peaks
        dref = a[locpk]

        # Detect double peaks allowing a tollerance of at most 5 deg
        phase_tolerance = bdata.BP_options.phase_tolerance

        # Phase difference between two following peaks computed like:
        # dref(0)   dref(1)    dref(2)     ...    dref(end-1)    dref(end)
        #              /          /                                  /
        #           dref(0)    dref(1)    dref(2)    ...         dref(end-1)    dref(end)    

        phase_diff = np.abs(np.angle(dref[1:] / dref[:-1]))
        idx_double = np.where(phase_diff < phase_tolerance)[0]

        # Keep copies before removing double peaks, useful for diagnostics

        locpk_raw = locpk.copy()
        dref_raw = dref.copy()
        pks_raw = pks.copy()

        # Remove double peaks
        #
        # For each pair of neighboring peaks with almost the same phase,
        # keep the one with the larger amplitude.
        n_removed = 0

        for idx in idx_double:
            # If I have removed something all the indexes after have to be shifted
            i = idx - n_removed

            if i < 0 or i + 1 >= len(dref):
                continue

            if np.abs(dref[i]) >= np.abs(dref[i + 1]):
                # Keep first peak, remove second
                locpk = np.delete(locpk, i + 1)
                dref = np.delete(dref, i + 1)
            else:
                # Remove first peak, keep second
                locpk = np.delete(locpk, i)
                dref = np.delete(dref, i)

            n_removed += 1

        """TODO
        if len(bdata.BP_options.remove_peaks) > 0:
            remove_idx = np.asarray(bdata.BP_options.remove_peaks, dtype=int) - 1
            remove_idx = remove_idx[
                (remove_idx >= 0) & (remove_idx < len(locpk))
            ]
            remove_idx = np.sort(remove_idx)[::-1]

            for idx in remove_idx:
                locpk = np.delete(locpk, idx)
                dref = np.delete(dref, idx)
        """

        pks = abs_atp_smooth[locpk]

        bdata.atp = atp
        bdata.threshold = float(threshold)
        bdata.abs_atp_smooth = abs_atp_smooth
        bdata.locpk = locpk
        bdata.pks = pks
        bdata.dref = dref
        bdata.phase_tolerance = phase_tolerance
        bdata.phase_diff = phase_diff
        bdata.idx_double = idx_double
        bdata.locpk_raw = locpk_raw
        bdata.dref_raw = dref_raw
        bdata.pks_raw = pks_raw

    def check_number_of_peaks(self, bdata: BeadpullRecord) -> None:
        """
        Check whether the number of detected peaks matches the number of cells.
        """
        if bdata.dref is None or bdata.locpk is None:
            raise ValueError("Cannot check peak count before peaks have been found.")

        noc = bdata.noc
        dref = bdata.dref
        locpk = bdata.locpk
        nop = len(dref)

        bdata.nop = nop

        if nop == noc:
            bdata.bad_peaks_idx = np.array([], dtype=int)
            bdata.bad_peaks_number = np.array([], dtype=int)
            return

        peak_magnitudes = np.abs(dref)

        meanpeaks = np.mean(peak_magnitudes)
        meanpeakdistance = np.mean(np.diff(locpk))
        deviation = np.abs(peak_magnitudes - meanpeaks)
        ns = np.argsort(deviation)[::-1]

        bdata.info["meanpeaks"] = meanpeaks
        bdata.info["meanpeakdistance"] = meanpeakdistance

        if nop > noc:
            n_extra = nop - noc
            bad_peaks_idx = np.sort(ns[:n_extra])
            bad_peaks_number = bad_peaks_idx + 1

            bdata.bad_peaks_idx = bad_peaks_idx
            bdata.bad_peaks_number = bad_peaks_number

            warnings.warn(
                f"Detected {nop} peaks but expected {noc}. "
                f"Suggested false peaks to remove: {bad_peaks_number}.",
                RuntimeWarning,
                stacklevel=2,
            )

            return

        bdata.bad_peaks_idx = np.array([], dtype=int)
        bdata.bad_peaks_number = np.array([], dtype=int)

        warnings.warn(
            f"Detected {nop} peaks but expected {noc}. Too few peaks were detected.",
            RuntimeWarning,
            stacklevel=2,
        )

    def extract_phase(self, bdata: BeadpullRecord) -> None:
        """
        Compute and store phase, field, and phase advance quantities.
        """
        if bdata.a is None or bdata.locpk is None or bdata.dref is None:
            raise ValueError("Cannot extract phase before `a`, `locpk`, and `dref` exist.")

        a = bdata.a
        locpk = bdata.locpk
        dref = bdata.dref
        phi = bdata.phi
        noc = bdata.noc
        nop = len(dref)

        if nop != noc:
            raise ValueError(
                f"Cannot extract phase with nop={nop} and noc={noc}. "
                "Adjust peak detection or `remove_peaks` first."
            )

        phase = np.unwrap(np.angle(a))
        phase_peaks = phase[locpk].copy()

        # Correct phase branch issues
        dphase_peaks = -np.diff(phase_peaks)

        Dpp = (dphase_peaks - 2 * phi) / (2 * np.pi)

        ffx = np.where(np.abs(Dpp) >= 0.5)[0]

        if len(ffx) > 0:
            for idx in ffx:
                phase_shift = np.round(Dpp[idx]) * 2 * np.pi
                # Correct this phase jump
                dphase_peaks[idx] = dphase_peaks[idx] - phase_shift
                # Shfit all following peak phases
                phase_peaks[idx + 1:] += phase_shift

        dphi_c = -np.diff(phase_peaks)
        Dpp_c = (dphi_c - 2 * phi) / (2 * np.pi)

        ff = np.where(np.abs(Dpp_c) >= 0.5)[0]

        phase_peaks = (
            phase_peaks
            - np.round(phase_peaks[0] / (2 * np.pi)) * 2 * np.pi
        )

        Ebp = np.sqrt(np.abs(dref)) * np.exp(1j * phase_peaks / 2)

        phiadv = -np.diff(phase_peaks / 2) / np.pi * 180

        if len(phiadv) > 2:
            phimean = float(np.mean(phiadv[1:-1]))
        elif len(phiadv) > 0:
            phimean = float(np.mean(phiadv))
        else:
            phimean = float("nan")

        phisig = 0.0

        bdata.phase = phase
        bdata.phase_peaks = phase_peaks
        bdata.dphase_peaks = dphase_peaks
        bdata.Dpp = Dpp
        bdata.ffx = ffx
        bdata.dphi_c = dphi_c
        bdata.Dpp_c = Dpp_c
        bdata.ff = ff
        bdata.Ebp = Ebp
        bdata.phiadv = phiadv
        bdata.phimean = phimean
        bdata.phisig = phisig

    def compute_forward_backward_waves(self, bdata: BeadpullRecord) -> None:
        """
        Compute and store `d`, `rovq`, `squ`, `I`, `A`, and `B`.
        """
        if bdata.Ebp is None or bdata.gamma0 is None or bdata.dref is None:
            raise ValueError(
                "Cannot compute forward/backward waves before `Ebp`, "
                "`gamma0`, and `dref` exist."
            )

        phi = bdata.phi
        phi0 = bdata.phi0
        fref = bdata.fref
        v_particles = bdata.v_particles
        noc = bdata.noc
        Ebp = bdata.Ebp
        gamma0 = bdata.gamma0
        dref = bdata.dref

        if len(Ebp) != noc:
            raise ValueError(f"`Ebp` has length {len(Ebp)}, but `noc` is {noc}.")

        d = phi0 * v_particles / fref / (2 * np.pi)

        rovq = bdata.rovq_ * d

        squ = Ebp / np.sqrt(rovq)

        # Determining forward and backward wave (from [1], Eq. 8 to 15)

        # # Field in the structure superposition of forward and backward waves
        I = squ

        A = np.zeros(noc + 1, dtype=np.complex128) # Forward wave
        B = np.zeros(noc + 1, dtype=np.complex128) # Backward wave

        # input                                           output
        #   |                                               |
        #   v                                               v

        #   A[0] -> cell 1 -> A[1] -> cell 2 -> ... -> A[noc]
        #   B[0] <- cell 1 <- B[1] <- cell 2 <- ... <- B[noc]

        # Implementing Eq. 10
        # A[1:-1]=             x        x             x
        # A      =    A[0]    A[1]    A[2] ... ... A[N-2]    A[N-1]
        # B      =    B[0]    B[1]    B[2] ... ... B[N-2]    B[N-1]
        # I      =    I[0]    I[1]    I[2] ... ... I[N-2]    I[N-1]
        # I[:-1] =      x      x        x             x
        # I[1:]  =             x        x             x        x
        
        A[1:-1] = (I[:-1] - I[1:] * np.exp(-1j * phi)) / (2j * np.sin(phi))

        B[1:-1] = (I[:-1] - I[1:] * np.exp(1j * phi)) / (-2j * np.sin(phi))

        # Output cell
        A[-1] = (A[-2] * (1 - abs(B[-2] / A[-2])) * np.exp(1j * phi[-1]))
        B[-1] = 0

        # Input cell ([1], Eq. 13)
        A[0] = A[1] * np.exp(1j * phi[0])
        # From Eq. 13 B[0] = A[0]*S11*exp(-2jphi0), with phi0 the phase offset 
        # of the input waveguide.
        # This can be computed from Eq. 14: exp(-2jphi0)=-j*|dref[0]|/dref[0]
        B[0] = A[0] * gamma0 * (-1j * np.abs(dref[0])) / dref[0]

        bdata.d = float(d)
        bdata.rovq = rovq
        bdata.squ = squ
        bdata.I = I
        bdata.A = A
        bdata.B = B

    def compute_local_reflection_and_tuning(self, bdata: BeadpullRecord) -> None:
        """
        Compute and store local reflection and tuning quantities.
        """
        if bdata.A is None or bdata.B is None:
            raise ValueError("Cannot compute local reflection before `A` and `B` exist.")

        vg = bdata.vg # Notice this is vg_ including the in and out cells but not the further interpolation done in the calculations
        phi = bdata.phi
        phi0 = bdata.phi0
        fref = bdata.fref
        v_particles = bdata.v_particles
        f0 = bdata.f0
        f1 = bdata.f1
        A = bdata.A
        B = bdata.B
        att = bdata.att

        # Using a mean phase value for filling the array in a first pass 
        # (it will be overwritten)
        mean_phi_inner = float(np.mean(phi[1:-1]))

        # From [1], Eq. 11
        s11local = (
            B[:-1] - B[1:] * np.exp(-1j * mean_phi_inner)
        ) / A[:-1]

        ds11local = -(
            B[:-1] - B[1:] * np.exp(-1j * mean_phi_inner)
        ) / A[:-1]

        # Now truly Eq. 11 with 2<=n<=N-1; Notice, we are still missing the 
        # correct values in input and output
        s11local[1:] = (
            B[1:-1] - B[2:] * np.exp(-1j * phi)
        ) / A[1:-1]

        ds11local[1:] = -(
            B[1:-1] - B[2:] * np.exp(-1j * phi)
        ) / A[1:-1]

        # Including losses
        ds11global = ds11local * att[:len(ds11local)]

        # Local correction for frequency variation due to temperature (bpparse.m line 273)
            # Store a copy
        s11local_org = np.array(s11local, copy=True)

        # Compute temperature correction based on frequency shifts
        ds11local_dtemp = (
            1j
            * (f0 - f1)
            / fref
            * v_particles
            * phi0
            / vg
        )

        # Compute again s11 local adding this correction
        s11local = s11local + ds11local_dtemp

        # Global correction for s11
        ds11 = (
            np.imag(ds11global)
            + (f1 - f0)
            / fref
            * v_particles
            * phi0
            / vg
            * att[:len(ds11local)]
        )

        # Frequency shift to apply to each cell
        df2tune = (
            np.imag(ds11local)
            / (v_particles * phi0 / vg)
            * fref
            + (f1 - f0)
        )

        bdata.mean_phi_inner = mean_phi_inner
        bdata.s11local = s11local
        bdata.ds11local = ds11local
        bdata.ds11global = ds11global
        bdata.s11local_org = s11local_org
        bdata.ds11local_dtemp = ds11local_dtemp
        bdata.ds11 = ds11
        bdata.df2tune = df2tune

    def compute_output_matching_correction(self, bdata: BeadpullRecord) -> None:
        """
        Compute and store the final two-cell output matching correction.
        """
        if bdata.A is None or bdata.B is None:
            raise ValueError(
                "Cannot compute output matching correction before `A` and `B` exist."
            )

        phi = bdata.phi
        phi0 = bdata.phi0
        A = bdata.A
        B = bdata.B
        att = bdata.att

        if len(A) < 8 or len(B) < 8 or len(att) < 8:
            raise ValueError(
                "Output matching correction requires at least eight wave samples."
            )

        idx112 = np.arange(5, -1, -1)

        ref_end_comp = (
            B[-8:-2]
            / A[-8:-2]
            / (att[-2] / att[-7:-1])
            * np.exp(2j * phi0 * idx112)
        )

        ref_mean = np.mean(ref_end_comp)

        ds11_comp_local_0 = -np.real(ref_mean) / np.sin(2 * phi[-1])

        ds11_comp_local_1 = (
            -np.imag(ref_mean)
            - ds11_comp_local_0 * np.cos(2 * phi[-1])
        )

        ds11_0 = ds11_comp_local_0 * att[-1]
        ds11_1 = ds11_comp_local_1 * att[-2]

        bdata.idx112 = idx112
        bdata.ref_end_comp = ref_end_comp
        bdata.ref_mean = ref_mean
        bdata.ds11_comp_local_0 = float(ds11_comp_local_0)
        bdata.ds11_comp_local_1 = float(ds11_comp_local_1)
        bdata.ds11_0 = float(ds11_0)
        bdata.ds11_1 = float(ds11_1)