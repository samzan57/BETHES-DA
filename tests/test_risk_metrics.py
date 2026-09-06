"""
Unit tests for the VaR/ES estimators (src/volatility_risk.py) and the
backtesting statistics (src/backtesting.py).

Run with:
    pytest tests/
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from volatility_risk import historical_var_es, parametric_normal_var_es
from backtesting import count_exceptions, kupiec_test


def test_historical_var_es_matches_empirical_quantile():
    rng = np.random.default_rng(42)
    returns = rng.standard_normal(5000)

    var, es = historical_var_es(returns, alpha=0.01)

    expected_var = -np.percentile(returns, 1)
    assert var == pytest_approx(expected_var)
    # Expected shortfall is the average loss beyond VaR, so it must be
    # at least as large as VaR itself.
    assert es >= var


def test_parametric_normal_var_matches_gaussian_quantile():
    # Standard normal returns: VaR_1% should be close to the theoretical
    # z-score of the 1st percentile of N(0, 1), i.e. ~2.326.
    rng = np.random.default_rng(0)
    returns = rng.standard_normal(200_000)

    var, es = parametric_normal_var_es(returns, alpha=0.01)

    assert var == pytest_approx(2.326, abs=0.05)
    assert es > var


def test_count_exceptions_flags_losses_beyond_var():
    returns = pd.Series([-0.05, -0.01, 0.02, -0.10, 0.00])
    var = pd.Series([0.03, 0.03, 0.03, 0.03, 0.03])

    exceptions = count_exceptions(returns, var)

    assert list(exceptions) == [1, 0, 0, 1, 0]


def test_kupiec_test_accepts_correctly_calibrated_var():
    # Exactly alpha * n exceptions -> observed rate equals the target
    # rate -> the likelihood-ratio statistic is 0 and the model is not
    # rejected (p-value = 1).
    n, alpha = 1000, 0.01
    exceptions = pd.Series([1] * int(alpha * n) + [0] * (n - int(alpha * n)))

    result = kupiec_test(exceptions, alpha)

    assert result["exceptions"] == 10
    assert result["lr_uc"] == pytest_approx(0.0, abs=1e-9)
    assert result["p_value"] == pytest_approx(1.0, abs=1e-9)


def test_kupiec_test_rejects_underestimated_risk():
    # Far more exceptions than the target rate -> the model understates
    # risk -> the test should reject it at the 5% level.
    n, alpha = 1000, 0.01
    exceptions = pd.Series([1] * 60 + [0] * (n - 60))

    result = kupiec_test(exceptions, alpha)

    assert result["p_value"] < 0.05


def pytest_approx(value, rel=1e-6, abs=1e-9):
    # Thin wrapper so this file has no hard dependency on the pytest
    # import name inside helper functions defined above pytest's own
    # collection of the module.
    import pytest

    return pytest.approx(value, rel=rel, abs=abs)
