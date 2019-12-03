"""Module to measure heat loss due to conduction"""

import numpy as np  # pylint: disable=import-error
from openmdao.api import ExplicitComponent  # pylint: disable=import-error


class HeatConduction(ExplicitComponent):
    """Class to find the heat generated due to the braking force"""

    def initialize(self):
        """Declare options"""
        self.options.declare('ThermalContactConductance',
                             default=0.5,
                             types=np.ScalarType,
                             desc="Thermal Resistance between pad and pod")

    def setup(self):
        """Declare inputs and outputs"""
        self.add_input('TemperatureBrakePad',
                       1.,
                       desc="Temperature of the brake pad")
        self.add_input('TemperatureContact',
                       1.,
                       desc="Temperature of the contact area")
        self.add_input('AreaContact',
                       1.,
                       desc="Area subject to conductive heat loss")
        self.add_output('HeatRate',
                        0.45,
                        desc="Rate of heat lost through conduction")

    def compute(self, inputs, outputs):
        """Compute outputs"""
        k_cond = self.options["ThermalContactConductance"]
        assert k_cond > 0, "Conductance must be positive"
        area = inputs["AreaContact"]
        assert area > 0, "Area must be a positive non-zero quantity"
        sur_temp = inputs["TemperatureContact"]
        pad_temp = inputs["TemperatureBrakePad"]
        assert pad_temp > sur_temp
        temp_diff = pad_temp - sur_temp

        heat_loss = -(k_cond * temp_diff * area)

        assert heat_loss < 0, "Heat Loss is always a negative quantity"
        outputs["HeatRate"] = heat_loss
