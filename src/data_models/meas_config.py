from dataclasses import dataclass

@dataclass
class MeasurementConfig:
    designed_frequency_mhz: float = 11994.0
    designed_temperature_c: float = 30.0
    rf_measurement_temperature_c: float = 20.0
    frequency_shift_mhz: float = 0.5
    atmosphere: str = "Nitrogen"
    relative_humidity_percent: float = 0.0
    target_frequency_mhz: float = 11994.0

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

    """
    @classmethod
    def from_file(cls, filename):
        values = {}

        mapping = {
            "trf": "rf_measurement_temperature_c",
            "tmeas": "rf_measurement_temperature_c",
            "thighpower": "designed_temperature_c",
            "fop": "designed_frequency_mhz",
            "dfwire": "frequency_shift_mhz",
        }

        with open(filename, "r") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                parts = line.split()

                if len(parts) < 2:
                    continue

                key = parts[0].lower()

                if key not in mapping:
                    continue

                attr = mapping[key]
                values[attr] = float(parts[1])

        return cls(**values)
    """