import pandas as pd
from yahooquery import Ticker

# =============================================================================
# CELL 4: STEP 2 - LIGHTWEIGHT FILTER (QUICK SCREENING)
# =============================================================================
# This is the first filter - it quickly eliminates stocks that don't meet
# basic requirements. It's "lightweight" because it uses fast bulk data fetching.


def get_initial_survivors(
    tickers,
    MIN_PRICE,
    MIN_VOLUME,
    MIN_CAP,
    MIN_CURRENT_RATIO,
    EXCLUDED_SECTORS,
    MAX_PE_RATIO,
):
    """
    Filter stocks using basic criteria.
    Args:
        tickers (list): List of all stock symbols to check

    Returns:
        DataFrame: Table of stocks that passed all filters with their metrics
    """
    print(f"\n--- STEP 2: Running 'Lightweight' Filter on {len(tickers)} stocks ---")

    chunk_size = 500  # Process stocks in batches of 500 at a time
    survivors = []  # This list will hold stocks that pass all filters

    # Split our list into chunks (batches) divides a large request into smaller parts
    chunks = [tickers[i : i + chunk_size] for i in range(0, len(tickers), chunk_size)]

    # Process each chunk
    for i, chunk in enumerate(chunks):
        # Print progress every 5 batches
        if i % 5 == 0:
            print(f" -> Processing Batch {i+1}/{len(chunks)}...")

        try:
            # Create a Ticker object for all stocks in this chunk
            # asynchronous=True means it fetches data for multiple stocks simultaneously
            yq = Ticker(chunk, asynchronous=True)

            # Get multiple types of data at once (this is the "bulk fetch")
            # These are different Yahoo Finance data modules
            df_modules = yq.get_modules(
                "summaryProfile summaryDetail financialData price defaultKeyStatistics"
            )

            # Loop through each stock's data
            for symbol, data in df_modules.items():
                # If data is just a string, it means there was an error
                if isinstance(data, str):
                    continue

                try:
                    # Extract the stock price
                    # The .get() method returns the value if it exists, or 0 if it doesn't
                    price = data.get("price", {}).get("regularMarketPrice", 0)
                    if price is None:
                        price = 0

                    # Extract average daily trading volume
                    vol = data.get("summaryDetail", {}).get("averageVolume", 0)
                    if vol is None or vol == 0:
                        # Try alternative location for volume data
                        vol = data.get("price", {}).get("averageDailyVolume10Day", 0)

                    # Extract market capitalization (total company value)
                    cap = data.get("price", {}).get("marketCap", 0)
                    if cap is None:
                        cap = 0

                    # Extract sector (Technology, Healthcare, etc.)
                    sector = data.get("summaryProfile", {}).get("sector", "Unknown")

                    # Get financial data
                    fin_data = data.get("financialData", {})
                    curr_ratio = fin_data.get("currentRatio", 0)  # Current Ratio
                    op_margins = fin_data.get(
                        "operatingMargins", 0
                    )  # Operating Margin (as decimal)
                    if curr_ratio is None:
                        curr_ratio = 0
                    if op_margins is None:
                        op_margins = 0

                    # Get P/E ratio
                    pe = data.get("summaryDetail", {}).get("trailingPE")

                    # ====== APPLY FILTERS ======
                    # Each 'continue' statement skips to the next stock (odd yes but continue skipping to the next stock essentially means it failed the filter and we move on)

                    # Skip if P/E is too high (overvalued)
                    if pe is not None and pe > MAX_PE_RATIO:
                        continue

                    # Skip if price is too low (penny stock)
                    if price < MIN_PRICE:
                        continue

                    # Skip if company is too small
                    if cap < MIN_CAP:
                        continue

                    # Skip if not enough trading volume (hard to buy/sell)
                    if vol < MIN_VOLUME:
                        continue

                    # Skip if in excluded sectors
                    # any() returns True if ANY item in the list is True
                    if any(x in sector for x in EXCLUDED_SECTORS):
                        continue

                    # Skip if current ratio is too low (can't pay bills)
                    if curr_ratio < MIN_CURRENT_RATIO:
                        continue

                    # Skip if operating margin is zero or negative (not profitable)
                    if op_margins <= 0:
                        continue

                    # If we get here, the stock passed ALL filters!
                    # Add it to our survivors list
                    survivors.append(
                        {
                            "Ticker": symbol,
                            "Sector": sector,
                            "Price": price,
                            "Op Margin %": round(
                                op_margins * 100, 2
                            ),  # Convert to percentage
                            "P/E": round(pe, 2) if pe else 0,
                            "Curr Ratio": curr_ratio,
                            "Mkt Cap (B)": round(
                                cap / 1_000_000_000, 2
                            ),  # Convert to billions
                        }
                    )

                except:
                    continue  # Skip this stock if any error occurs

        except:
            continue  # Skip this batch if any error occurs

    # Convert our list of dictionaries to a pandas DataFrame (table)
    return pd.DataFrame(survivors)
