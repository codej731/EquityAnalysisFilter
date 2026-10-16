import json
import os

import pandas as pd
import yfinance as yf

# =============================================================================
# CELL 2: CONFIGURATION AND SETUP
# =============================================================================


# Saves all configuration settings and helper functions
DATA_FOLDER = "YfinanceDataDump"

# Check if the folder already exists.
if not os.path.exists(DATA_FOLDER):
    try:
        # os.makedirs() creates the folder
        os.makedirs(DATA_FOLDER)
        print(f"Created data folder: {DATA_FOLDER}")
    except Exception as e:
        # If something goes wrong (like no permission), save to current folder
        print(
            f"Warning: Could not create folder '{DATA_FOLDER}'. Saving to current directory. Error: {e}"
        )
        DATA_FOLDER = "."  # The dot means "current folder"

# 2. FILE PATHS
# -------------
# These are the names of the files where we'll save our results
# os.path.join() combines the folder name with the file name properly
# (handles / vs \ on different operating systems automatically)
CACHE_FILE = os.path.join(
    DATA_FOLDER, "financial_cache.json"
)  # Stores data we've already fetched
FORTRESS_CSV = os.path.join(DATA_FOLDER, "fortress_stocks.csv")  # Best quality stocks
STRONG_CSV = os.path.join(DATA_FOLDER, "strong_stocks.csv")  # Good quality stocks
RISKY_CSV = os.path.join(DATA_FOLDER, "risky_stocks.csv")  # Lower quality stocks
ANALYST_CSV = os.path.join(
    DATA_FOLDER, "Analyst_Fortress_Picks.csv"
)  # Analyst favorites
BUFFETT_CSV = os.path.join(DATA_FOLDER, "Buffett_Value_Picks.csv")  # Value picks
DEEPVAL_CSV = os.path.join(
    DATA_FOLDER, "Deep_Value_Gems.csv"
)  # Intersection of Buffett + Burry
# 3. UNIVERSE FILTERS - Minimum requirements for a stock to be considered


# ==========================================
# HELPER FUNCTIONS - Reusable code blocks
# ==========================================


def load_cache():
    """
    Load previously saved financial data from our cache file.
    Returns:
        dict: A dictionary of saved data, or empty dict {} if no cache exists
    """
    # Check if the cache file exists
    if os.path.exists(CACHE_FILE):
        try:
            # 'r' means "read mode" - we're reading the file, not writing to it
            with open(CACHE_FILE, "r") as f:
                # json.load() reads the JSON file and converts it to a Python dictionary
                return json.load(f)
        except:
            # If something goes wrong reading the file, return empty dictionary
            return {}
    # If file doesn't exist, return empty dictionary
    return {}


def save_cache(cache_data):
    """
    Save our financial data to a file so we can reuse it later.

    Args:
        cache_data (dict): The data we want to save
    """
    try:
        # 'w' means "write mode" - creates the file or overwrites existing
        with open(CACHE_FILE, "w") as f:
            # json.dump() converts the Python dictionary to JSON format and saves it
            json.dump(cache_data, f)
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")


def calculate_altman_z_yfinance(bs, fin, market_cap):
    """
    Calculate the Altman Z-Score - a formula that predicts bankruptcy risk.

    Created by Professor Edward Altman in 1968.
    - Z > 2.99: "Safe Zone" - company is financially healthy
    - Z between 1.81 and 2.99: "Grey Zone" - uncertain
    - Z < 1.81: "Distress Zone" - high risk of bankruptcy

    The Formula: Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E

    Where:
    A = Working Capital / Total Assets (liquidity)
    B = Retained Earnings / Total Assets (profitability over time)
    C = EBIT / Total Assets (operating efficiency)
    D = Market Value of Equity / Total Liabilities (market confidence)
    E = Sales / Total Assets (asset efficiency)

    Args:
        bs: Balance Sheet data (DataFrame from yfinance)
        fin: Financial data (DataFrame from yfinance)
        market_cap: Market capitalization in dollars

    Returns:
        float: The Z-Score, or 0 if calculation fails
    """
    try:
        # Helper function to safely get values from DataFrames
        # Sometimes the data has different names, so we try multiple
        def get_val(df, keys):
            for k in keys:
                if k in df.index:
                    # .iloc[0] gets the first value (most recent year)
                    return df.loc[k].iloc[0]
            return 0

        # Get values from Balance Sheet (bs) and Financials (fin)
        total_assets = get_val(bs, ["Total Assets"])
        total_liab = get_val(
            bs, ["Total Liabilities Net Minority Interest", "Total Liabilities"]
        )
        current_assets = get_val(bs, ["Current Assets", "Total Current Assets"])
        current_liab = get_val(bs, ["Current Liabilities", "Total Current Liabilities"])
        retained_earnings = get_val(bs, ["Retained Earnings"])

        ebit = get_val(
            fin, ["EBIT", "Operating Income"]
        )  # Earnings Before Interest & Taxes
        total_revenue = get_val(fin, ["Total Revenue"])

        # Can't divide by zero, so return 0 if missing key data
        if total_assets == 0 or total_liab == 0:
            return 0

        # Calculate each component of the Z-Score formula
        # A: Working Capital (can company pay short-term bills?)
        A = (current_assets - current_liab) / total_assets

        # B: Retained Earnings (accumulated profits over the years)
        B = retained_earnings / total_assets

        # C: EBIT (operating profitability)
        C = ebit / total_assets

        # D: Market Cap vs Debt (how much does market trust vs owe?)
        D = market_cap / total_liab

        # E: Revenue (is the company using its assets efficiently?)
        E = total_revenue / total_assets

        # The final formula with Altman's coefficients
        return (1.2 * A) + (1.4 * B) + (3.3 * C) + (0.6 * D) + (1.0 * E)

    except Exception as e:
        return 0  # Return 0 if any calculation fails


# =============================================================================
def apply_trend_alignment(df_input):
    """
    Calculates the 200-day SMA and filters for stocks trading ABOVE it.

    Args:
        df_input: DataFrame containing a 'Ticker' column.

    Returns:
        DataFrame: Filtered list of stocks in an Uptrend.
    """
    if df_input is None or df_input.empty:
        print("   [Trend Alignment] Input DataFrame is empty. Skipping.")
        return df_input

    tickers = df_input["Ticker"].tolist()
    print(f"\n--- TREND ALIGNMENT: Checking 200-Day SMA for {len(tickers)} stocks ---")

    # 1. Fetch History (Need at least 200 trading days, so 1y is perfect)
    #    auto_adjust=True fixes prices for splits/dividends
    try:
        data = yf.download(
            tickers,
            period="1y",
            interval="1d",
            progress=False,
            group_by="ticker",
            auto_adjust=True,
        )
    except Exception as e:
        print(f"   [Error] Failed to fetch history: {e}")
        return df_input

    trend_data = []

    for ticker in tickers:
        try:
            # Handle MultiIndex if multiple tickers, or single Index if one ticker
            if len(tickers) > 1:
                if ticker not in data.columns.levels[0]:
                    continue
                df_hist = data[ticker].copy()
            else:
                df_hist = data.copy()

            # Need at least 200 days of data
            if len(df_hist) < 200:
                print(f"   [Skipping] {ticker}: Insufficient history (<200 days).")
                continue

            # 2. Calculate 200-Day SMA
            # .rolling(window=200).mean() averages the last 200 closing prices
            sma_200 = df_hist["Close"].rolling(window=200).mean().iloc[-1]
            current_price = df_hist["Close"].iloc[-1]

            # 3. The Filter Condition: Price > SMA 200
            # We want stocks that have "reclaimed" their trend
            is_uptrend = current_price > sma_200

            # Calculate how far above/below the SMA it is (as a %)
            distance_pct = round(((current_price - sma_200) / sma_200) * 100, 2)

            if is_uptrend:
                # Get the original row data
                base_row = df_input[df_input["Ticker"] == ticker].iloc[0].to_dict()

                # Add technical metrics
                base_row["SMA_200"] = round(sma_200, 2)
                base_row["Trend_Dist_%"] = distance_pct

                trend_data.append(base_row)

        except Exception as e:
            continue

    # Create the new filtered DataFrame
    df_uptrend = pd.DataFrame(trend_data)

    if not df_uptrend.empty:
        print(
            f"   [Result] {len(df_uptrend)}/{len(tickers)} stocks are in a Long-Term Uptrend."
        )
        # Sort by strength of trend (distance above SMA)
        df_uptrend = df_uptrend.sort_values(by="Trend_Dist_%", ascending=True)
    else:
        print("   [Result] No stocks passed the Trend Alignment filter.")

    return df_uptrend
