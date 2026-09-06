"""
data_fetch.py

Script simple pour télécharger les données historiques de SPY (S&P 500 ETF)
- Utilise yfinance
- Sauvegarde en CSV et Parquet dans le dossier 'data'

Dépendances recommandées:
 pip install yfinance pandas pyarrow

Usage:
 python data_fetch.py
"""

import os
from datetime import datetime

import pandas as pd
import yfinance as yf


def download_spy(start='2000-01-01', end=None, out_dir='data'):
    """Télécharge SPY entre start et end (YYYY-MM-DD) et sauvegarde CSV + Parquet.
    Retourne le DataFrame pandas.
    """
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    os.makedirs(out_dir, exist_ok=True)

    ticker = 'SPY'
    # auto_adjust=True applique ajustements pour dividendes/splits => 'Adj Close' devient 'Close' ajusté
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)

    if df.empty:
        print('Aucune donnée téléchargée — vérifier la connexion ou les dates.')
        return df

    # Sauvegardes
    safe_start = start.replace(':', '-')
    safe_end = end.replace(':', '-')
    csv_path = os.path.join(out_dir, f"{ticker}_{safe_start}_{safe_end}.csv")
    parquet_path = os.path.join(out_dir, f"{ticker}_{safe_start}_{safe_end}.parquet")

    # Enregistrer
    df.to_csv(csv_path)
    try:
        df.to_parquet(parquet_path)
    except Exception as e:
        print('Erreur lors de l''écriture Parquet (pyarrow manquant?). Parquet non créé. Erreur:', e)

    print(f"Données sauvegardées: {csv_path}")
    if os.path.exists(parquet_path):
        print(f"Données sauvegardées: {parquet_path}")

    return df


if __name__ == '__main__':
    # Paramètres par défaut — modifier si besoin
    download_spy(start='2010-01-01')
