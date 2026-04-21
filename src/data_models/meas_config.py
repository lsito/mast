from dataclasses import dataclass

@dataclass
class MeasurementConfig:
    designed_frequency_mhz: float = 11994.0
    designed_temperature_c: float = 30.0
    rf_measurement_temperature_c: float = 20.0
    frequency_shift_mhz: float = 0.5
    atmosphere: str = "Nitrogen"
    relative_humidity_percent: float = 0.0
    target_frequency_mhz: float = 0.0