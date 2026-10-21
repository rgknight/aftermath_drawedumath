import numpy as np
import pytest
from error_overlap import expected_wrong_count_distribution


def test_expected_distribution_for_identical_independent_models_is_binomial():
    distribution = expected_wrong_count_distribution([0.5, 0.5])

    np.testing.assert_allclose(distribution, [0.25, 0.5, 0.25])


def test_expected_distribution_handles_unequal_error_rates():
    # P(0 wrong) = 0.9 * 0.8, P(both wrong) = 0.1 * 0.2
    distribution = expected_wrong_count_distribution([0.1, 0.2])

    np.testing.assert_allclose(distribution, [0.72, 0.26, 0.02])
    assert distribution.sum() == pytest.approx(1.0)
