"""Module to calculate the friction of coefficient"""

# pylint: disable=R0914,E0611
import numpy as np  # pylint: disable=import-error
from openmdao.api import ExplicitComponent  # pylint: disable=import-error


class FrictionCoefficient(ExplicitComponent):
    """Class to find the friction coefficient of the friction pad"""

    def initialize(self):
        """Declare options"""

        self.options.declare('SteadyStateFrictionCoefficient',
                             default=1.0,
                             types=np.ScalarType,
                             desc="Coefficient of Friction at SteadyState")

        self.options.declare('MultiplicationFactorSpeed',
                             default=1.0,
                             types=np.ScalarType,
                             desc="Multiplication Factor - friction speed")

        self.options.declare('MultiplicationFactorTempertature',
                             default=1.0,
                             types=np.ScalarType,
                             desc="Multiplication Factor - rise in temperature")

        self.options.declare('ParametricFactorSpeed',
                             default=1.0,
                             types=np.ScalarType,
                             desc="Parametric Factor - friction speed")

        self.options.declare('ParametricFactorTempertature',
                             default=1.0,
                             types=np.ScalarType,
                             desc="Parametric Factor - rise in temperature")

        self.options.declare('OriginalTemperature',
                             default=1.0,
                             types=np.ScalarType,
                             desc="Original Temperature")
    def setup(self):
        """Declare inputs and outputs"""

        self.add_input('SurfaceVelocity',
                       1.,
                       desc="Velocity of the friction pad.")

        self.add_input('Temperature',
                       1.,
                       desc="Temperature of the friction pad.")

        self.add_output('FrictionCoefficient',
                        0.45,
                        desc="Friction Coefficient of the friction pad")

    def compute(self, inputs, outputs):
        """Compute outputs"""

        mu_d0 = self.options['SteadyStateFrictionCoefficient']
        n_v = self.options['MultiplicationFactorSpeed']
        n_t = self.options['MultiplicationFactorFrictionTempertature']
        m_v = self.options['ParametricFactorSpeed']
        m_t = self.options['ParametricFactorFrictionTempertature']
        t_o = self.options['OriginalTemperature']

        v_curr = inputs["SurfaceVelocity"]
        t_curr = inputs["Temperature"]

        import math

        velocity_factor = 1 + n_v*math.exp(-(m_v*v_curr))
        temperature_factor = 1 + n_t*math.exp(-(m_t*(t_curr-t_o)))
        mu_curr = mu_d0 * velocity_factor * temperature_factor

        outputs["FrictionCoefficient"] = mu_curr
