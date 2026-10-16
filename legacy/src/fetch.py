import io

import pandas as pd
import requests


def get_combined_universe():
    """
    Get a list of all US stock tickers from NASDAQ.

    The NASDAQ provides a file with all traded stocks.
    US tickers don't need a suffix (like .TO for Toronto).

    Returns:
        list: List of US stock symbols like ['AAPL', 'MSFT', 'GOOGL', ...]
    """
    print("--- STEP 1: Fetching North American Universe ---")
    tickers = []

    # Fetch from NASDAQ's official list
    try:
        url_us = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqtraded.txt"

        # Download the file
        s = requests.get(url_us).content

        # Read it as a CSV with pipe (|) as the separator
        # The file looks like: Symbol|Security Name|ETF|Test Issue|...
        df_us = pd.read_csv(io.StringIO(s.decode("utf-8")), sep="|")

        # Filter out test issues and ETFs (we only want real stocks)
        df_us = df_us[(df_us["Test Issue"] == "N") & (df_us["ETF"] == "N")]

        # Get symbols, clean them, and keep only short ones (< 5 characters)
        # Long symbols are usually special securities we don't want
        us_list = [
            x.replace("$", "-") for x in df_us["Symbol"].astype(str) if len(x) < 5
        ]

        tickers.extend(us_list)  # Add to our list
        print(f"   -> Found {len(us_list)} US stocks.")

    except:
        print("   -> Error fetching USA list.")

    return tickers
