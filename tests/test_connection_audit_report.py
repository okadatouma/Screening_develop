import unittest
import numpy as np
from screening_analysis.connection_audit_report import standardized_selection,sensitivity_summary


class ConnectionReportChecks(unittest.TestCase):
    def test_standardization_removes_distance_composition_confounding(self):
        rows=[]
        for distance in (0,1):
            for same in (False,True):
                exposure=100 if distance==int(same) else 10
                rate=.1 if distance==0 else .9
                for angle in range(12):
                    rows.append(dict(input_origin='A',head='0',k='90',distance_bin=str(distance),
                        same_spin=str(same),angle_bin=str(angle),promotion_opportunities=str(exposure),
                        promoted=str(int(exposure*rate))))
        result=standardized_selection(rows,'model',0,90,'promotion')
        self.assertGreater(result['raw_same_rate']-result['raw_opposite_rate'],.5)
        self.assertAlmostEqual(result['standardized_same_rate'],.5)
        self.assertAlmostEqual(result['standardized_opposite_rate'],.5)
        np.testing.assert_allclose(result['standardized_angular_rates'],.5)
        self.assertAlmostEqual(result['spin_standardization_exposure_fraction'],1.)

    def test_global_normalization_and_phase_are_removed_from_centered_response(self):
        rows=[dict(head='0',k='0',group='all_offdiagonal',input_origin='A',dlogabs='.2',dphase='.3',
                   linear_dlogabs='.2',linear_dphase='.3',edges='6',screening_mass='1',
                   mean_abs_amplitude_derivative='.1',mean_abs_phase_derivative='.1') for _ in range(4)]
        result=sensitivity_summary(rows,'model')[0]
        self.assertAlmostEqual(result['raw_amplitude_response_rms'],.2)
        self.assertAlmostEqual(result['raw_phase_response_rms'],.3)
        self.assertAlmostEqual(result['centered_amplitude_response_rms'],0.)
        self.assertAlmostEqual(result['centered_phase_response_rms'],0.)


if __name__=='__main__':unittest.main()
