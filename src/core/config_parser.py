from dataclasses import fields
import json
import numpy as np
from numpy.typing import NDArray
from typing import get_origin, get_args, Union

# Get the JSON file into a dictionary
# Unpack dictionary into arg values for dataclass constructor

class DataMixin:

    @classmethod
    def from_dict(cls, data: dict):

        kwargs = {}

        for f in fields(cls):

            if f.name not in data:
                continue

            value = data[f.name]
            field_type = f.type

            if value is None:
                kwargs[f.name] = None
                continue

            # Convert JSON lists -> numpy arrays
            if "NDArray" in str(field_type):
                kwargs[f.name] = np.asarray(value, dtype=float)

            else:
                kwargs[f.name] = value

        # Notice ** unpack kwarg (keyword arguments) dictionary into class args
        return cls(**kwargs)

    @classmethod
    def from_json(cls, filename: str):

        with open(filename, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)