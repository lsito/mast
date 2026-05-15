from dataclasses import dataclass
from src.core.config_parser import DataMixin

@dataclass(slots=True)
class MeasurementConfig(DataMixin):
    """
    Container describing the RF measurement conditions used during bead-pull 
    analysis and frequency tuning calculations.

    This dataclass stores the environmental and operating parameters needed
    to compute frequency corrections between the nominal RF design conditions
    and the actual measurement conditions.

    The correction model includes:
    - temperature-dependent frequency shifts,
    - atmosphere-dependent frequency shifts,
    - additional frequency offsets introduced by the wire measurement setup.

    Notes
    -----
    - Frequencies are expressed in MHz.
    - Temperatures are expressed in degrees Celsius.
    - The target tuning frequency is computed dynamically from the correction
      terms and is therefore exposed as a read-only property.
    - ``slots=True`` is enabled to reduce memory usage and prevent accidental
      dynamic attribute creation.
    - The class inherits from ``DataMixin`` to provide serialization and/or
      utility methods shared across the application. Loading from JSON is not
      supporte yet.

    Attributes
    ----------
    designed_frequency_mhz : float, default=11994.0
        Nominal RF design frequency of the structure in MHz.

    designed_temperature_c : float, default=30.0
        Reference design temperature of the RF structure in degrees Celsius.

    rf_measurement_temperature_c : float, default=20.0
        Actual RF measurement temperature in degrees Celsius.

    frequency_shift_mhz : float, default=0.5
        Frequency correction introduced by the bead-pull wire measurement
        setup, expressed in MHz. The default value is chosen from experience,
        might need to be adjusted.

    atmosphere : str, default="Nitrogen"
        Atmosphere inside the RF structure during the measurement.
        Typical values are:
        - ``"Nitrogen"``
        - ``"Air"``

    relative_humidity_percent : float, default=0.0
        Relative humidity inside the structure during measurement, expressed
        as a percentage. Typically, 0% for dry nitrogen.

    Properties
    ----------
    temperature_correction_mhz : float
        Frequency correction due to the temperature difference between the
        RF design condition and the actual measurement condition.

    atmosphere_correction_mhz : float
        Frequency correction due to the atmosphere inside the structure.
        A correction is currently applied only when the atmosphere is air.

    total_correction_mhz : float
        Total frequency correction combining:
        - atmosphere correction,
        - wire-induced frequency shift,
        - temperature correction.

    target_frequency_mhz : float
        Corrected target frequency used for tuning calculations, computed as:

        ``designed_frequency_mhz - total_correction_mhz``
    """

    designed_frequency_mhz: float = 11994.0
    designed_temperature_c: float = 30.0
    rf_measurement_temperature_c: float = 20.0
    frequency_shift_mhz: float = 0.5 # Why this already defined?
    atmosphere: str = "Nitrogen"
    relative_humidity_percent: float = 0.0

    @property
    def temperature_correction_mhz(self):
        return -self.designed_frequency_mhz * 1.66e-5 * (
            self.designed_temperature_c - self.rf_measurement_temperature_c
        )

    @property
    def atmosphere_correction_mhz(self):
        if self.atmosphere.lower() == "air":
            return self.designed_frequency_mhz * 0.00029
        return 0.0

    @property
    def total_correction_mhz(self):
        return (
            self.atmosphere_correction_mhz
            + self.frequency_shift_mhz
            + self.temperature_correction_mhz
        )

    @property
    def target_frequency_mhz(self):
        return self.designed_frequency_mhz - self.total_correction_mhz