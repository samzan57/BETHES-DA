"""
volatility_risk.py

Calibrer un modèle GARCH(1,1) simple sur les rendements SPY,
calculer des métriques de risque VaR et ES, et comparer plusieurs méthodes.

Usage:
  python volatility_risk.py

Le script charge par défaut le fichier CSV de données collectées dans data/.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, t

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError as e:
    raise ImportError(
        "Le package statsmodels est requis pour l'ARMA-GARCH. "
        "Installez-le avec pip install statsmodels"
    ) from e


def _find_price_column(columns, candidates):
    for col in columns:
        if isinstance(col, tuple):
            if col[0] in candidates:
                return col
        elif col in candidates:
            return col
    return None


def _find_date_column(columns):
    for col in columns:
        if isinstance(col, tuple):
            if any(str(part) == "Date" for part in col):
                return col
        elif col == "Date":
            return col
    return None


def load_spy_returns(csv_path):
    # Tenter d'abord la lecture des CSV yfinance avec en-têtes multi-lignes.
    try:
        df = pd.read_csv(csv_path, header=[0, 1, 2])
        date_col = _find_date_column(df.columns)
        if date_col is not None:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            price_col = _find_price_column(df.columns, ["Adj Close", "Close", "Price"])
            if price_col is not None:
                prices = df[price_col].astype(float)
                returns = np.log(prices / prices.shift(1)).dropna()
                returns.name = "log_return"
                return returns
    except Exception:
        pass

    # Lecture de secours pour CSV plus simples.
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    if "Adj Close" in df.columns:
        price_col = "Adj Close"
    elif "Close" in df.columns:
        price_col = "Close"
    else:
        raise ValueError("Aucune colonne de prix trouvée dans le CSV SPY")
    prices = df[price_col].astype(float)
    returns = np.log(prices / prices.shift(1)).dropna()
    returns.name = "log_return"
    return returns


def summary_stats(returns):
    return {
        "count": len(returns),
        "mean": float(returns.mean()),
        "std": float(returns.std(ddof=1)),
        "skew": float(returns.skew()),
        "kurtosis": float(returns.kurtosis()),
        "min": float(returns.min()),
        "max": float(returns.max()),
    }


def compute_garch_sigma2(returns, omega, alpha, beta):
    values = returns.values
    n = len(values)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(values, ddof=1)
    for t in range(1, n):
        sigma2[t] = omega + alpha * values[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2


def garch_neg_loglik(params, returns):
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e15
    values = returns.values
    n = len(values)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(values, ddof=1)
    ll = 0.0
    for t in range(1, n):
        sigma2[t] = omega + alpha * values[t - 1] ** 2 + beta * sigma2[t - 1]
        if sigma2[t] <= 0:
            return 1e15
        ll += 0.5 * (
            np.log(2 * np.pi)
            + np.log(sigma2[t])
            + values[t] ** 2 / sigma2[t]
        )
    return ll


def fit_garch(returns):
    # Initial guess and bounds
    init = np.array([1e-6, 0.05, 0.90])
    bounds = [(1e-12, None), (1e-12, 1 - 1e-8), (1e-12, 1 - 1e-8)]
    constraints = ({"type": "ineq", "fun": lambda x: 0.999 - x[1] - x[2]})

    # Ensure initial guess satisfies the stationarity constraint alpha+beta < 0.999
    if init[1] + init[2] >= 0.999:
        init[2] = max(1e-6, 0.998 - init[1])

    # First attempt: SLSQP with explicit constraint
    result = minimize(
        garch_neg_loglik,
        init,
        args=(returns,),
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if result.success:
        omega, alpha, beta = result.x
        sigma2 = compute_garch_sigma2(returns, omega, alpha, beta)
        return {"omega": omega, "alpha": alpha, "beta": beta}, sigma2

    # Fallback: try a bounded optimizer (L-BFGS-B) with a penalty on alpha+beta
    print(f"Warning: SLSQP failed ({result.message}). Tentative de secours avec L-BFGS-B...")

    def penalized_obj(x, returns):
        # Use original neg-loglik but add a smooth penalty if alpha+beta >= 0.999
        val = garch_neg_loglik(x, returns)
        alpha = x[1]
        beta = x[2]
        excess = alpha + beta - 0.999
        if excess > 0:
            val += 1e12 * excess * excess
        return val

    # Make sure the starting point is within bounds for L-BFGS-B
    init_b = np.minimum(np.maximum(init, [b[0] for b in bounds]), [b[1] if b[1] is not None else init[i] for i, b in enumerate(bounds)])

    result2 = minimize(
        penalized_obj,
        init_b,
        args=(returns,),
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 1000},
    )

    if not result2.success:
        raise RuntimeError(
            f"Calibration GARCH échouée: SLSQP: {result.message}; L-BFGS-B: {result2.message}"
        )

    omega, alpha, beta = result2.x
    sigma2 = compute_garch_sigma2(returns, omega, alpha, beta)
    return {"omega": omega, "alpha": alpha, "beta": beta}, sigma2


def historical_var_es(returns, alpha=0.01):
    quantile = np.percentile(returns, 100 * alpha)
    var = -quantile
    es = -returns[returns <= quantile].mean()
    return var, es


def parametric_normal_var_es(returns, alpha=0.01):
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = norm.ppf(alpha)
    var = -(mu + sigma * z)
    es = -mu + sigma * norm.pdf(z) / alpha
    return var, es


def student_t_var_es(returns, alpha=0.01):
    mu = returns.mean()
    df, loc, scale = t.fit(returns, floc=mu)
    z = t.ppf(alpha, df)
    var = -(mu + scale * z)
    es = -mu + scale * t.pdf(z, df) * (df + z ** 2) / (alpha * (df - 1))
    return var, es, df, loc, scale


def fit_arma_mean(returns, p=1, q=1):
    model = ARIMA(returns, order=(p, 0, q), trend="c")
    result = model.fit()
    return result


def fit_arma_garch(returns, p=1, q=1):
    arma_res = fit_arma_mean(returns, p=p, q=q)
    residuals = arma_res.resid.dropna()
    garch_params, sigma2 = fit_garch(residuals)
    return arma_res, garch_params, sigma2


def arma_garch_var_es(returns, arma_res, garch_params, alpha=0.01, n_sim=20000, seed=12345):
    np.random.seed(seed)
    omega = garch_params["omega"]
    alpha_g = garch_params["alpha"]
    beta = garch_params["beta"]
    residuals = arma_res.resid.dropna()
    last_resid = float(residuals.iloc[-1])
    last_sigma2 = float(compute_garch_sigma2(residuals, omega, alpha_g, beta)[-1])
    forecast_sigma2 = omega + alpha_g * last_resid ** 2 + beta * last_sigma2
    # statsmodels may return a Series with a non-zero-based index when no supported
    # frequency is present — use iloc to extract the first scalar forecast robustly.
    mu_forecast = float(arma_res.forecast(steps=1).iloc[0])
    draws = np.random.normal(loc=mu_forecast, scale=np.sqrt(forecast_sigma2), size=n_sim)
    quantile = np.percentile(draws, 100 * alpha)
    var = -quantile
    es = -draws[draws <= quantile].mean()
    return var, es, forecast_sigma2, mu_forecast


def garch_monte_carlo_var_es(returns, garch_params, alpha=0.01, n_sim=20000, seed=12345):
    np.random.seed(seed)
    omega = garch_params["omega"]
    alpha_g = garch_params["alpha"]
    beta = garch_params["beta"]
    last_return = float(returns.iloc[-1])
    last_sigma2 = float(compute_garch_sigma2(returns, omega, alpha_g, beta)[-1])
    forecast_sigma2 = omega + alpha_g * last_return ** 2 + beta * last_sigma2
    draws = np.random.normal(loc=0.0, scale=np.sqrt(forecast_sigma2), size=n_sim)
    quantile = np.percentile(draws, 100 * alpha)
    var = -quantile
    es = -draws[draws <= quantile].mean()
    return var, es, forecast_sigma2


def format_currency(x):
    return f"{100 * x:.4f}%"


def main():
    parser = argparse.ArgumentParser(description="Calibrer un GARCH(1,1) et calculer VaR/ES pour SPY")
    parser.add_argument(
        "--data",
        default=os.path.join("data", "SPY_2010-01-01_2026-08-03.csv"),
        help="Chemin vers le fichier CSV SPY",
    )
    parser.add_argument("--alpha", type=float, default=0.01, help="Niveau de confiance pour VaR/ES")
    parser.add_argument(
        "--arma-garch",
        action="store_true",
        help="Calibrer un ARMA(1,1)-GARCH(1,1) sur la moyenne et la variance",
    )
    args = parser.parse_args()

    returns = load_spy_returns(args.data)
    stats = summary_stats(returns)
    print("=== Statistiques des rendements log SPY ===")
    for name, value in stats.items():
        print(f"{name:>10}: {value}")
    print()

    if args.arma_garch:
        print("Calibration du modèle ARMA(1,1)-GARCH(1,1) en cours...")
        arma_res, garch_params, sigma2 = fit_arma_garch(returns, p=1, q=1)
        print("Paramètres ARMA(1,1)-GARCH(1,1):")
        print(f"  ARMA coef: {arma_res.arparams if hasattr(arma_res, 'arparams') else 'N/A'}")
        print(f"  MA coef: {arma_res.maparams if hasattr(arma_res, 'maparams') else 'N/A'}")
        print(f"  const: {arma_res.params.get('const', np.nan):.6e}")
        print("  GARCH params:")
        for key, value in garch_params.items():
            print(f"    {key}: {value:.6e}")
        print(f"Dernière volatilité conditionnelle (sigma): {float(np.sqrt(sigma2[-1])):.6f}")
        print()
    else:
        print("Calibration du modèle GARCH(1,1) en cours...")
        garch_params, sigma2 = fit_garch(returns)
        print("Paramètres GARCH:")
        for key, value in garch_params.items():
            print(f"  {key}: {value:.6e}")
        print(f"Dernière volatilité conditionnelle (sigma): {float(np.sqrt(sigma2[-1])):.6f}")
        print()

    print("=== VaR / ES à {:.1%} ===".format(args.alpha))
    hist_var, hist_es = historical_var_es(returns, args.alpha)
    print(f"Historical VaR : {format_currency(hist_var)}")
    print(f"Historical ES  : {format_currency(hist_es)}")

    norm_var, norm_es = parametric_normal_var_es(returns, args.alpha)
    print(f"Parametric Normal VaR : {format_currency(norm_var)}")
    print(f"Parametric Normal ES  : {format_currency(norm_es)}")

    t_var, t_es, df, loc, scale = student_t_var_es(returns, args.alpha)
    print(f"Student-t VaR : {format_currency(t_var)} (df={df:.2f})")
    print(f"Student-t ES  : {format_currency(t_es)}")

    if args.arma_garch:
        mc_var, mc_es, h_forecast, mu_forecast = arma_garch_var_es(returns, arma_res, garch_params, args.alpha)
        print(f"ARMA-GARCH VaR : {format_currency(mc_var)}")
        print(f"ARMA-GARCH ES  : {format_currency(mc_es)}")
        print(f"ARMA forecast mean (1 jour): {mu_forecast:.8e}")
        print(f"GARCH forecast variance (1 jour): {h_forecast:.8f}")
    else:
        mc_var, mc_es, h_forecast = garch_monte_carlo_var_es(returns, garch_params, args.alpha)
        print(f"Monte Carlo GARCH VaR : {format_currency(mc_var)}")
        print(f"Monte Carlo GARCH ES  : {format_currency(mc_es)}")
        print(f"GARCH forecast variance (1 jour): {h_forecast:.8f}")


if __name__ == "__main__":
    main()
