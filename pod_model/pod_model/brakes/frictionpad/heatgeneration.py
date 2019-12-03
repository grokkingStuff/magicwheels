"""Module to measure heat generation due to braking force"""

import numpy as np  # pylint: disable=import-error
from openmdao.api import ExplicitComponent  # pylint: disable=import-error


class HeatGeneration(ExplicitComponent):
    """Class to find the heat generated due to the braking force"""

    def initialize(self):
        """Declare options"""
        self.options.declare('HeatRatePadRatio',
                             default=0.5,
                             types=np.ScalarType,
                             desc="Ratio of heat absorbed by the pad vs track")

    def setup(self):
        """Declare inputs and outputs"""
        self.add_input('BrakingForce',
                       1.,
                       desc="Braking Force of the friction pad.")
        self.add_input('SurfaceVelocity',
                       1.,
                       desc="Velocity of the surface relative to the pad.")
        self.add_output('HeatRatePad',
                        0.45,
                        desc="Heat absorbed by the friction pad")
        self.add_output('HeatRateTrack',
                        0.45,
                        desc="Heat absorbed by the track")

    def compute(self, inputs, outputs):
        """Compute outputs"""
        braking_force = inputs["BrakingForce"]
        surface_velocity = inputs["SurfaceVelocity"]

        ratio = self.options["HeatRatePadRatio"]
        assert ratio > 0 and ratio < 1
        total_heat_rate = braking_force * surface_velocity
        heat_rate_pad = ratio * total_heat_rate
        heat_rate_track = (1-ratio) * total_heat_rate
        outputs["HeatRatePad"] = heat_rate_pad
        outputs["HeatRateTrack"] = heat_rate_track
