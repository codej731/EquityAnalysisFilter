import time

import yfinance as yf
import pandas as pd

from src.setup import load_cache

# =============================================================================
# CELL 5: STEP 3 - DEEP FINANCIAL ANALYSIS
# =============================================================================
# This is the detailed analysis phase. For each stock that passed Step 2,
# we download detailed financial statements and calculate advanced metrics.


def get_advanced_metrics(survivor_df, CACHE_EXPIRY_DAYS, FORTRESS_MARGIN_THRESHOLD, MIN_INTEREST_COVERAGE, MIN_ROIC, calculate_altman_z_yfinance, save_cache):
    """
    Perform deep financial analysis on stocks that passed initial screening.


    Args:
        survivor_df: DataFrame of stocks from Step 2

    Returns:
        DataFrame: Stocks with full analysis and tier classification
    """
    tickers = survivor_df["Ticker"].tolist()
    print(f"\n--- STEP 3: Fetching Deep Financials for {len(tickers)} Survivors ---")

    # Load our cached data (data we've already fetched before)
    cache = load_cache()
    current_time = time.time()  # Current time in seconds since 1970 (Unix timestamp)
    expiry_seconds = (
        CACHE_EXPIRY_DAYS * 86400
    )  # Convert days to seconds (86400 sec/day)

    final_data = []  # Will store our results

    # Loop through each stock
    for i, ticker in enumerate(tickers):
        # Print progress every 20 stocks
        if i % 20 == 0:
            print(f" -> Analyzing {i+1}/{len(tickers)}: {ticker}...")

        # Uncomment the line below if you're getting throttled (too many requests)
        # time.sleep(0.75)  # Wait 0.75 seconds between requests

        def determine_tier_history(metrics, is_fortress_margin, is_pos_margin):
            """
            Determine tier based on Margins AND Financial Health (Z-Score).

            STRICT RULES:
            1. Fortress = High Margin (>5%) AND Safe Z-Score (>2.99)
            2. Strong   = Positive Margin (>0%) AND Acceptable Z-Score (>1.81)
            3. Risky    = Fails either margins or safety
            """

            # Extract Z-Score from the metrics dictionary
            z_val = metrics.get("z_score", 0)

            # 1. IMMEDIATE FAILURES (Hard Safety Stops)
            # If a company can't pay interest or has low return on capital, it's Risky.
            if metrics["int_cov"] < MIN_INTEREST_COVERAGE:
                return "Risky"
            if metrics["roic"] < MIN_ROIC:
                return "Risky"

            # 2. FORTRESS CRITERIA (The "Perfect" Stock)
            # Must have BOTH strong margins AND a safe Z-Score
            if is_fortress_margin and z_val >= 2.99:
                return "Fortress"

            # 3. STRONG CRITERIA (The "Good" Stock)
            # Must have at least positive margins AND be out of the "Distress Zone"
            # We use 1.81 because Z < 1.81 implies high bankruptcy risk
            elif is_pos_margin and z_val >= 1.81:
                return "Strong"

            # 4. FALLTHROUGH
            # If it failed the above, it's Risky (either unprofitable or unsafe balance sheet)
            else:
                return "Risky"

        # Check if we have cached data for this stock that's not expired
        cached_data = cache.get(ticker)
        if cached_data and (current_time - cached_data["timestamp"] < expiry_seconds):
            if cached_data.get("roic") == -999:
                continue  # Skip if previous fetch failed

        # FETCH NEW DATA using yfinance
        try:
            stock = yf.Ticker(ticker)
            fin = stock.financials  # Income statement data
            bs = stock.balance_sheet  # Balance sheet data

            # Check if Yahoo actually gave us data
            if fin.empty or bs.empty:
                print(f"   No data for {ticker} (skipping)")
                continue

            # --- CALCULATE 4-YEAR AVERAGE OPERATING MARGIN ---
            # We look at multiple years to see if profitability is consistent
            try:
                # Try to get Operating Income (might be called different things)
                if "Operating Income" in fin.index:
                    op_income_history = fin.loc["Operating Income"]
                elif "EBIT" in fin.index:
                    op_income_history = fin.loc["EBIT"]
                else:
                    op_income_history = pd.Series([0])

                # Get Revenue for same periods
                revenue_history = fin.loc["Total Revenue"]

                # Calculate margin for each year: Operating Income / Revenue
                yearly_margins = (op_income_history / revenue_history).dropna()

                if len(yearly_margins) > 0:
                    avg_margin = yearly_margins.mean()  # Average of all years

                    # Check if average meets our thresholds
                    is_fortress_margin = avg_margin > FORTRESS_MARGIN_THRESHOLD
                    is_positive_margin = avg_margin > 0
                else:
                    is_fortress_margin = False
                    is_positive_margin = False

            except Exception as e:
                is_fortress_margin = False
                is_positive_margin = False

            # --- STANDARD CALCULATIONS ---
            def get_item(df, keys):
                """Helper to get values that might have different names."""
                for k in keys:
                    if k in df.index:
                        return df.loc[k].iloc[0]
                return 0

            # Get needed values
            ebit = get_item(fin, ["EBIT", "Operating Income", "Pretax Income"])
            int_exp = get_item(
                fin, ["Interest Expense", "Interest Expense Non Operating"]
            )
            total_assets = get_item(bs, ["Total Assets"])
            curr_liab = get_item(
                bs, ["Current Liabilities", "Total Current Liabilities"]
            )

            # Calculate Interest Coverage Ratio
            # This tells us if the company can pay its interest bills
            int_exp = abs(int_exp)  # Make positive (expenses are often negative)
            if int_exp == 0:
                int_cov = 100  # No debt = infinite coverage, use 100 as placeholder
            else:
                int_cov = ebit / int_exp

            # Calculate ROIC (Return on Invested Capital)
            # Shows how efficiently management uses capital
            invested_cap = total_assets - curr_liab  # Simplified invested capital
            if invested_cap <= 0:
                roic = 0
            else:
                roic = ebit / invested_cap

            # Calculate Altman Z-Score
            base_row = survivor_df[survivor_df["Ticker"] == ticker].iloc[0]
            mkt_cap_raw = (
                base_row["Mkt Cap (B)"] * 1_000_000_000
            )  # Convert back to dollars
            z = calculate_altman_z_yfinance(bs, fin, mkt_cap_raw)

            # Store metrics in cache for future use
            metrics = {
                "timestamp": current_time,
                "z_score": round(z, 2),
                "roic": roic,
                "int_cov": round(int_cov, 2),
            }
            cache[ticker] = metrics

            # Determine final tier based on all metrics
            tier = determine_tier_history(
                metrics, is_fortress_margin, is_positive_margin
            )

            # Add all data to our final results
            final_data.append(
                {
                    "Ticker": ticker,
                    "Tier": tier,
                    "Price": base_row["Price"],
                    "P/E": base_row["P/E"],
                    "Sector": base_row["Sector"],
                    "Z-Score": round(z, 2),
                    "ROIC %": round(roic * 100, 2),
                    "Op Margin %": base_row["Op Margin %"],
                    "Avg Margin (4Y)": (
                        round(avg_margin * 100, 2) if "avg_margin" in locals() else 0
                    ),
                    "Curr Ratio": base_row["Curr Ratio"],
                    "Int Cov": round(int_cov, 2),
                    "Mkt Cap (B)": base_row["Mkt Cap (B)"],
                }
            )

        except Exception as e:
            continue  # Skip if any error

    # Save updated cache to file
    save_cache(cache)

    return pd.DataFrame(final_data)
