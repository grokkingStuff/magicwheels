import unittest

import pytest
from openmdao.api import Group, IndepVarComp, Problem
from parameterized import parameterized
from pod_model.brakes.frictionpad.frictioncoefficient import \
    FrictionCoefficient

testdata = [
    (1, 0, 0, 0, 0, 5, [5], [0], [1]),
    (1, 0, 0, 0, 0, 5, [5], [0], [1]),
]


class TestFrictionCoefficient(unittest.TestCase):
    @parameterized.expand(testdata)
    def test_init(self,
                  sFC,  # steadyStateFrictionCoefficient,
                  mFS,  # multiplicationFactorSpeed,
                  mFT,  # multiplicationFactorTemperature,
                  pFS,  # parameticFactorSpeed,
                  pFT,  # parametricFactorTemperature,
                  oT,  # originalTemperature,
                  list_temp,  # list of temperatures to test at
                  list_vel,  # list of velocities to test at
                  expected):            # expected frictionCoefficients


        my_comp=FrictionCoefficient(SteadyStateFrictionCoefficient=sFC,
                                    MultiplicationFactorSpeed=mFS,
                                    MultiplicationFactorTemperature=mFT,
                                    ParametricFactorSpeed=pFS,
                                    ParametricFactorTemperature=pFT,
                                    OriginalTemperature=oT,
                                   )
        import numpy as np
        for temp,vel,expected_val in zip(list_temp,list_vel,expected):
            comp = IndepVarComp()
            comp.add_output('SurfaceVelocity', val=3.0, lower=0, upper=10)
            comp.add_output('Temperature', val=2.0, lower=1, upper=20)

            prob=Problem()
            prob.model.add_subsystem('indep_var', comp)
            prob.model.add_subsystem('my_comp', my_comp)

            prob.model.connect('indep_var.SurfaceVelocity', 'my_comp.SurfaceVelocity')
            prob.model.connect('indep_var.Temperature',    'my_comp.Temperature')
            prob.setup()
            prob.run_model()
            assert prob['my_comp.FrictionCoefficient'] == np.array([expected_val])
