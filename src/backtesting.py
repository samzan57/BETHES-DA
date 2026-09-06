"""
backtesting.py

Backtesting de VaR sur les rendements SPY.
 
Le script charge le CSV de données SPY, calcule des prévisions de VaR
rolling (historique, gaussien et ARMA-GARCH) puis applique les tests
de Kupiec et de Christoffersen pour évaluer la qualité du modèle de risque.

Usage:
  python backtesting.py
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from volatility_risk import load_spy_returns, fit_arma_mean, fit_garch


def rolling_historical_var(returns, lookback=252, alpha=0.01):
    var = -returns.shift(1).rolling(window=lookback).quantile(alpha)
    return var.dropna()


def rolling_gaussian_var(returns, lookback=252, alpha=0.01):
    mu = returns.shift(1).rolling(window=lookback).mean()
    sigma = returns.shift(1).rolling(window=lookback).std(ddof=1)
    z = norm.ppf(alpha)
    var = -(mu + sigma * z)
    return var.dropna()


def count_exceptions(returns, var):
    return (returns < -var).astype(int)


def kupiec_test(exceptions, alpha):
    n = len(exceptions)
    x = int(exceptions.sum())
    p_hat = x / n
    ll_uncond = np.log((1 - alpha) ** (n - x) * alpha**x)
    ll_cond = np.log((1 - p_hat) ** (n - x) * p_hat**x)
    lr_uc = -2 * (ll_uncond - ll_cond)
    p_value = 1 - chi2.cdf(lr_uc, 1)
    return {
        "exceptions": x,
        "expected": alpha * n,
        "exception_rate": p_hat,
        "lr_uc": float(lr_uc),
        "p_value": float(p_value),
    }


def christoffersen_test(exceptions):
    exc = exceptions.astype(int)
    prev_exc = exc.shift(1).fillna(0).astype(int)
    curr_exc = exc.astype(int)

    n00 = int(((prev_exc == 0) & (curr_exc == 0)).sum())
    n01 = int(((prev_exc == 0) & (curr_exc == 1)).sum())
    n10 = int(((prev_exc == 1) & (curr_exc == 0)).sum())
    n11 = int(((prev_exc == 1) & (curr_exc == 1)).sum())

    total = n00 + n01 + n10 + n11
    p = (n01 + n11) / total
    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0

    def safe_log(x):
        return np.log(max(x, 1e-12))

    ll_indep = safe_log(1 - p) * (n00 + n10) + safe_log(p) * (n01 + n11)
    ll_dep = safe_log(1 - p01) * n00 + safe_log(p01) * n01 + safe_log(1 - p11) * n10 + safe_log(p11) * n11
    lr_ind = -2 * (ll_indep - ll_dep)
    p_value = 1 - chi2.cdf(lr_ind, 1)
    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "lr_ind": float(lr_ind),
        "p_value": float(p_value),
    }


def rolling_arma_garch_var(returns, lookback=252, alpha=0.01, p=1, q=1):
    index = []
    var_values = []
    for end in range(lookback, len(returns)):
        window = returns.iloc[end - lookback : end]
        try:
            arma_res = fit_arma_mean(window, p=p, q=q)
            residuals = arma_res.resid.dropna()
            garch_params, sigma2 = fit_garch(residuals)
            mu_forecast = float(arma_res.forecast(steps=1).iloc[0])
            last_resid = float(residuals.iloc[-1])
            last_sigma2 = float(sigma2[-1])
            forecast_sigma2 = (
                garch_params["omega"]
                + garch_params["alpha"] * last_resid ** 2
                + garch_params["beta"] * last_sigma2
            )
            var_values.append(-(mu_forecast + np.sqrt(forecast_sigma2) * norm.ppf(alpha)))
        except Exception:
            var_values.append(np.nan)
        index.append(returns.index[end])

    return pd.Series(var_values, index=index).dropna()


def run_backtest(returns, lookback=252, alpha=0.01, arma_garch=True):
    hist_var = rolling_historical_var(returns, lookback=lookback, alpha=alpha)
    gaussian_var = rolling_gaussian_var(returns, lookback=lookback, alpha=alpha)

    hist_ex = count_exceptions(returns.loc[hist_var.index], hist_var)
    gaussian_ex = count_exceptions(returns.loc[gaussian_var.index], gaussian_var)

    results = []
    for name, var, exc in [("historical", hist_var, hist_ex), ("gaussian", gaussian_var, gaussian_ex)]:
        exc = exc.dropna()
        kupiec = kupiec_test(exc, alpha)
        independence = christoffersen_test(exc)
        results.append(
            {
                "method": name,
                "lookback": lookback,
                "alpha": alpha,
                "n_obs": len(exc),
                "exceptions": kupiec["exceptions"],
                "expected": kupiec["expected"],
                "exception_rate": kupiec["exception_rate"],
                "lr_uc": kupiec["lr_uc"],
                "p_value_uc": kupiec["p_value"],
                "lr_ind": independence["lr_ind"],
                "p_value_ind": independence["p_value"],
                "n00": independence["n00"],
                "n01": independence["n01"],
                "n10": independence["n10"],
                "n11": independence["n11"],
            }
        )

    if arma_garch:
        arma_garch_var = rolling_arma_garch_var(returns, lookback=lookback, alpha=alpha)
        arma_garch_ex = count_exceptions(returns.loc[arma_garch_var.index], arma_garch_var)
        arma_garch_ex = arma_garch_ex.dropna()
        if len(arma_garch_ex) > 0:
            kupiec = kupiec_test(arma_garch_ex, alpha)
            independence = christoffersen_test(arma_garch_ex)
            results.append(
                {
                    "method": "arma-garch",
                    "lookback": lookback,
                    "alpha": alpha,
                    "n_obs": len(arma_garch_ex),
                    "exceptions": kupiec["exceptions"],
                    "expected": kupiec["expected"],
                    "exception_rate": kupiec["exception_rate"],
                    "lr_uc": kupiec["lr_uc"],
                    "p_value_uc": kupiec["p_value"],
                    "lr_ind": independence["lr_ind"],
                    "p_value_ind": independence["p_value"],
                    "n00": independence["n00"],
                    "n01": independence["n01"],
                    "n10": independence["n10"],
                    "n11": independence["n11"],
                }
            )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Backtesting de VaR sur SPY")
    parser.add_argument(
        "--data",
        default=os.path.join("data", "SPY_2010-01-01_2026-08-03.csv"),
        help="Chemin vers le CSV SPY",
    )
    parser.add_argument("--lookback", type=int, default=252, help="Fenêtre de rolling window")
    parser.add_argument("--alpha", type=float, default=0.01, help="Niveau de VaR")
    parser.add_argument(
        "--skip-arma-garch",
        action="store_true",
        help="Ne pas inclure le backtest ARMA-GARCH (plus lent)",
    )
    args = parser.parse_args()

    returns = load_spy_returns(args.data)
    results = run_backtest(
        returns,
        lookback=args.lookback,
        alpha=args.alpha,
        arma_garch=not args.skip_arma_garch,
    )
    print(results.to_string(index=False))

    out_csv = os.path.join("data", "var_backtest_results.csv")
    results.to_csv(out_csv, index=False)
    print(f"\nRésultats sauvegardés dans {out_csv}")


if __name__ == "__main__":
    main()
