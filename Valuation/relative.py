"""
Relative Valuation Module — Phase 1
====================================
Computes multiples-based valuation for a single ticker, with comparison to:
  (a) Sector peer median (across a user-provided peer list)
  (b) Company's own ~5-year historical median

Multiples covered:
  - Current:    P/E (TTM), Fwd P/E, PEG, EV/EBITDA, EV/EBIT, EV/Sales, P/B, P/S, P/FCF
  - Historical: P/E, EV/EBITDA, EV/EBIT, P/B, P/S, P/FCF
                (Fwd P/E and PEG omitted — yfinance has no consensus-estimate history)

Output: per-multiple table with current value, peer median, own historical median,
        implied price under each benchmark, and margin of safety vs current price.

Limitations
-----------
- yfinance annual statements typically go back 4 years, so "5Y history" is really 4 data points.
- Historical multiples use fiscal year-end prices vs annual fiscal metrics (no intra-year detail).
- Doesn't handle banks/insurers/REITs differently — these need DDM / FFO / residual income
  (Phase 5). Flag sector in the survivor df and skip or branch accordingly.
- .info from yfinance is occasionally rate-limited or stale; cache layer mitigates this.

Usage
-----
    from relative import RelativeValuation, peers_from_survivors

    # Standalone:
    rv = RelativeValuation("AAPL")
    summary = rv.valuation_summary(peer_tickers=["MSFT", "GOOGL", "META", "AMZN"])
    print(summary)

    # Integrated with existing screener output:
    survivors = pd.read_csv("YfinanceDataDump/fortress_stocks.csv")
    peers = peers_from_survivors("AAPL", survivors)
    summary = RelativeValuation("AAPL").valuation_summary(peer_tickers=peers)
"""

import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Cache layer (self-contained for prototyping; lift into common/ later)
# ---------------------------------------------------------------------------

CACHE_FILE = "valuation_cache.json"
CACHE_EXPIRY_HOURS = 24


def _load_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, default=str)
    except Exception as e:
        print(f"[cache] Could not save: {e}")


# ---------------------------------------------------------------------------
# Safe-extract helper for yfinance financial statements
# ---------------------------------------------------------------------------

def _get_line(df: Optional[pd.DataFrame], keys: list, col_index: int = 0):
    """Pull a value from a yfinance statement DataFrame, trying multiple key names.
    
    yfinance uses inconsistent labels (e.g. 'EBIT' vs 'Operating Income'); the keys
    list gives a fallback chain. col_index=0 = most recent fiscal period.
    """
    if df is None or df.empty:
        return None
    for key in keys:
        if key in df.index:
            try:
                val = df.loc[key].iloc[col_index]
                if pd.notna(val):
                    return float(val)
            except Exception:
                continue
    return None


# Standardized key fallbacks — centralized so adding new metrics is one place
KEYS = {
    "revenue":      ["Total Revenue", "Revenue"],
    "ebit":         ["EBIT", "Operating Income"],
    "net_income":   ["Net Income", "Net Income Common Stockholders"],
    "total_equity": ["Stockholders Equity", "Total Equity Gross Minority Interest"],
    "total_debt":   ["Total Debt", "Long Term Debt"],
    "cash":         ["Cash And Cash Equivalents",
                     "Cash Cash Equivalents And Short Term Investments"],
    "shares":       ["Diluted Average Shares", "Basic Average Shares", "Share Issued"],
    "ocf":          ["Operating Cash Flow",
                     "Cash Flow From Continuing Operating Activities"],
    "capex":        ["Capital Expenditure", "Capital Expenditures"],
    "dep_amort":    ["Depreciation And Amortization", "Reconciled Depreciation"],
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RelativeValuation:
    """Compute relative valuation multiples for a single ticker."""

    MULTIPLES_FOR_IMPLIED_PRICE = ["P/E (TTM)", "EV/EBITDA", "EV/EBIT", "P/B", "P/S", "P/FCF"]

    def __init__(self, ticker: str, use_cache: bool = True):
        self.ticker = ticker.upper()
        self.use_cache = use_cache
        self._stock = yf.Ticker(self.ticker)

        # Lazy-loaded — avoids unnecessary API calls
        self._info = None
        self._financials = None
        self._balance = None
        self._cashflow = None

    # -- Lazy property loaders ----------------------------------------------

    @property
    def info(self) -> dict:
        if self._info is None:
            try:
                self._info = self._stock.info or {}
            except Exception as e:
                print(f"[{self.ticker}] .info fetch failed: {e}")
                self._info = {}
        return self._info

    @property
    def financials(self) -> pd.DataFrame:
        if self._financials is None:
            self._financials = self._stock.financials
        return self._financials

    @property
    def balance_sheet(self) -> pd.DataFrame:
        if self._balance is None:
            self._balance = self._stock.balance_sheet
        return self._balance

    @property
    def cashflow(self) -> pd.DataFrame:
        if self._cashflow is None:
            self._cashflow = self._stock.cashflow
        return self._cashflow

    # -- Current multiples (TTM / forward) ----------------------------------

    def current_multiples(self) -> dict:
        """Pull current TTM/forward multiples — most from .info, EV/EBIT and P/FCF computed."""
        info = self.info

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        market_cap = info.get("marketCap")
        enterprise_value = info.get("enterpriseValue")

        # Direct from .info
        pe_ttm    = info.get("trailingPE")
        pe_fwd    = info.get("forwardPE")
        peg       = info.get("pegRatio") or info.get("trailingPegRatio")
        ev_ebitda = info.get("enterpriseToEbitda")
        ev_sales  = info.get("enterpriseToRevenue")
        pb        = info.get("priceToBook")
        ps        = info.get("priceToSalesTrailing12Months")

        # EV/EBIT — not in .info, compute manually
        ebit = _get_line(self.financials, KEYS["ebit"])
        ev_ebit = (enterprise_value / ebit) if (enterprise_value and ebit and ebit > 0) else None

        # P/FCF — compute from cashflow statement
        # yfinance returns capex as a negative number; add it to OCF to get FCF
        ocf = _get_line(self.cashflow, KEYS["ocf"])
        capex = _get_line(self.cashflow, KEYS["capex"])
        if ocf is not None and capex is not None:
            fcf = ocf + capex if capex < 0 else ocf - capex
            p_fcf = (market_cap / fcf) if (market_cap and fcf and fcf > 0) else None
        else:
            p_fcf = None

        return {
            "Price":       price,
            "Market Cap":  market_cap,
            "P/E (TTM)":   pe_ttm,
            "Fwd P/E":     pe_fwd,
            "PEG":         peg,
            "EV/EBITDA":   ev_ebitda,
            "EV/EBIT":     ev_ebit,
            "EV/Sales":    ev_sales,
            "P/B":         pb,
            "P/S":         ps,
            "P/FCF":       p_fcf,
        }

    # -- Historical multiples (annual snapshots) ----------------------------

    def historical_multiples(self) -> pd.DataFrame:
        """Fiscal year-end multiples for the past ~4 years using YE price × annual fiscals."""
        fin = self.financials
        bs = self.balance_sheet
        cf = self.cashflow

        if fin is None or fin.empty or bs is None or bs.empty:
            return pd.DataFrame()

        fiscal_dates = fin.columns

        # Pull prices covering the full window with buffer for closest-trading-day lookup
        try:
            prices = self._stock.history(
                start=fiscal_dates.min() - pd.Timedelta(days=30),
                end=fiscal_dates.max() + pd.Timedelta(days=30),
                auto_adjust=True,
            )
        except Exception as e:
            print(f"[{self.ticker}] Historical price fetch failed: {e}")
            return pd.DataFrame()

        if prices.empty:
            return pd.DataFrame()

        # Normalize timezones so date math doesn't blow up
        prices_idx = prices.index.tz_localize(None) if prices.index.tz else prices.index

        # Backup share count source if annual `shares` line is missing
        try:
            shares_hist = self._stock.get_shares_full(start=fiscal_dates.min())
        except Exception:
            shares_hist = None

        records = []
        for i, fy_end in enumerate(fiscal_dates):
            try:
                fy_naive = pd.Timestamp(fy_end).tz_localize(None) \
                    if pd.Timestamp(fy_end).tz else pd.Timestamp(fy_end)

                # Closest trading day's close
                closest_idx = (abs(prices_idx - fy_naive)).argmin()
                price = float(prices["Close"].iloc[closest_idx])

                # Statement values at fiscal year i (col 0 = most recent)
                revenue    = _get_line(fin, KEYS["revenue"],      col_index=i)
                ebit       = _get_line(fin, KEYS["ebit"],         col_index=i)
                net_income = _get_line(fin, KEYS["net_income"],   col_index=i)
                equity     = _get_line(bs,  KEYS["total_equity"], col_index=i)
                debt       = _get_line(bs,  KEYS["total_debt"],   col_index=i) or 0
                cash       = _get_line(bs,  KEYS["cash"],         col_index=i) or 0
                shares     = _get_line(fin, KEYS["shares"],       col_index=i)
                dep        = _get_line(cf,  KEYS["dep_amort"],    col_index=i) or 0
                ocf        = _get_line(cf,  KEYS["ocf"],          col_index=i)
                capex      = _get_line(cf,  KEYS["capex"],        col_index=i)

                # Share count fallback
                if not shares or shares <= 0:
                    if shares_hist is not None and len(shares_hist) > 0:
                        try:
                            shares = float(shares_hist.asof(fy_naive))
                        except Exception:
                            continue
                    else:
                        continue

                market_cap = price * shares
                enterprise_value = market_cap + debt - cash
                ebitda = (ebit + dep) if ebit is not None else None

                pe        = (market_cap / net_income) if (net_income and net_income > 0) else None
                pb        = (market_cap / equity)     if (equity and equity > 0)         else None
                ps        = (market_cap / revenue)    if (revenue and revenue > 0)       else None
                ev_ebitda = (enterprise_value / ebitda) if (ebitda and ebitda > 0)       else None
                ev_ebit   = (enterprise_value / ebit)   if (ebit and ebit > 0)           else None

                if ocf is not None and capex is not None:
                    fcf = ocf + capex if capex < 0 else ocf - capex
                    p_fcf = (market_cap / fcf) if fcf > 0 else None
                else:
                    p_fcf = None

                records.append({
                    "Fiscal Year End": fy_naive.date(),
                    "Price":     round(price, 2),
                    "P/E (TTM)": round(pe, 2)        if pe        else None,
                    "EV/EBITDA": round(ev_ebitda, 2) if ev_ebitda else None,
                    "EV/EBIT":   round(ev_ebit, 2)   if ev_ebit   else None,
                    "P/B":       round(pb, 2)        if pb        else None,
                    "P/S":       round(ps, 2)        if ps        else None,
                    "P/FCF":     round(p_fcf, 2)     if p_fcf     else None,
                })
            except Exception:
                continue

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records).sort_values("Fiscal Year End").reset_index(drop=True)

    # -- Peer multiples (batch fetch with caching) --------------------------

    @staticmethod
    def peer_multiples(peer_tickers: list, use_cache: bool = True) -> pd.DataFrame:
        """Fetch current multiples for each peer ticker; returns DF indexed by ticker."""
        cache = _load_cache() if use_cache else {}
        now = time.time()
        rows = []

        for tk in peer_tickers:
            cached = cache.get(tk)
            if cached and (now - cached.get("_ts", 0)) < CACHE_EXPIRY_HOURS * 3600:
                row = {k: v for k, v in cached.items() if k != "_ts"}
                row["Ticker"] = tk
                rows.append(row)
                continue

            try:
                m = RelativeValuation(tk, use_cache=False).current_multiples()
                m["Ticker"] = tk
                rows.append(m)
                cache[tk] = {**m, "_ts": now}
            except Exception as e:
                print(f"[peer fetch] {tk} failed: {e}")
                continue

        if use_cache:
            _save_cache(cache)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("Ticker")
        return df

    # -- Implied price back-solver ------------------------------------------

    def _implied_price(self, multiple_name: str, mult_value: Optional[float],
                       fundamentals: dict) -> Optional[float]:
        """Back-solve the implied per-share price from a target multiple."""
        if not mult_value or mult_value <= 0:
            return None

        shares = fundamentals["shares"]
        if not shares or shares <= 0:
            return None

        ni     = fundamentals["net_income"]
        rev    = fundamentals["revenue"]
        ebit   = fundamentals["ebit"]
        ebitda = fundamentals["ebitda"]
        eq     = fundamentals["equity"]
        debt   = fundamentals["debt"]
        cash   = fundamentals["cash"]
        fcf    = fundamentals["fcf"]

        try:
            if multiple_name == "P/E (TTM)" and ni and ni > 0:
                return mult_value * ni / shares
            if multiple_name == "P/B" and eq and eq > 0:
                return mult_value * eq / shares
            if multiple_name == "P/S" and rev and rev > 0:
                return mult_value * rev / shares
            if multiple_name == "P/FCF" and fcf and fcf > 0:
                return mult_value * fcf / shares
            if multiple_name == "EV/EBITDA" and ebitda and ebitda > 0:
                implied_ev = mult_value * ebitda
                return (implied_ev - debt + cash) / shares
            if multiple_name == "EV/EBIT" and ebit and ebit > 0:
                implied_ev = mult_value * ebit
                return (implied_ev - debt + cash) / shares
        except Exception:
            return None
        return None

    # -- Triangulation summary (the main output) ----------------------------

    def valuation_summary(self, peer_tickers: Optional[list] = None) -> pd.DataFrame:
        """Per-multiple table: current vs peer median vs own median, with implied prices."""
        current = self.current_multiples()
        hist = self.historical_multiples()
        price = current["Price"]

        # Bundle fundamentals once for implied-price calcs
        ocf   = _get_line(self.cashflow, KEYS["ocf"])
        capex = _get_line(self.cashflow, KEYS["capex"])
        if ocf is not None and capex is not None:
            fcf = ocf + capex if capex < 0 else ocf - capex
        else:
            fcf = None
        ebit = _get_line(self.financials, KEYS["ebit"])
        dep  = _get_line(self.cashflow, KEYS["dep_amort"]) or 0

        fundamentals = {
            "shares":     self.info.get("sharesOutstanding"),
            "net_income": _get_line(self.financials, KEYS["net_income"]),
            "revenue":    _get_line(self.financials, KEYS["revenue"]),
            "ebit":       ebit,
            "ebitda":     (ebit + dep) if ebit is not None else None,
            "equity":     _get_line(self.balance_sheet, KEYS["total_equity"]),
            "debt":       _get_line(self.balance_sheet, KEYS["total_debt"]) or 0,
            "cash":       _get_line(self.balance_sheet, KEYS["cash"]) or 0,
            "fcf":        fcf,
        }

        # Peer medians
        if peer_tickers:
            print(f"Fetching peer multiples for {len(peer_tickers)} peers...")
            peers = self.peer_multiples(peer_tickers)
            if not peers.empty:
                peer_medians = peers[self.MULTIPLES_FOR_IMPLIED_PRICE].median(numeric_only=True)
                peer_n = peers[self.MULTIPLES_FOR_IMPLIED_PRICE].notna().sum()
            else:
                peer_medians = pd.Series(dtype=float)
                peer_n = pd.Series(dtype=int)
        else:
            peer_medians = pd.Series(dtype=float)
            peer_n = pd.Series(dtype=int)

        # Own historical medians
        if not hist.empty:
            hist_medians = hist[self.MULTIPLES_FOR_IMPLIED_PRICE].median(numeric_only=True)
            hist_n = hist[self.MULTIPLES_FOR_IMPLIED_PRICE].notna().sum()
        else:
            hist_medians = pd.Series(dtype=float)
            hist_n = pd.Series(dtype=int)

        # Build per-multiple rows
        rows = []
        for m in self.MULTIPLES_FOR_IMPLIED_PRICE:
            cur_val  = current.get(m)
            peer_med = peer_medians.get(m) if m in peer_medians.index else None
            hist_med = hist_medians.get(m) if m in hist_medians.index else None
            if pd.isna(peer_med): peer_med = None
            if pd.isna(hist_med): hist_med = None

            imp_peer = self._implied_price(m, peer_med, fundamentals)
            imp_hist = self._implied_price(m, hist_med, fundamentals)

            mos_peer = ((imp_peer - price) / price * 100) if (imp_peer and price) else None
            mos_hist = ((imp_hist - price) / price * 100) if (imp_hist and price) else None

            rows.append({
                "Multiple":          m,
                "Current":           round(cur_val, 2)  if cur_val  else None,
                "Peer Median":       round(peer_med, 2) if peer_med else None,
                "Peer n":            int(peer_n.get(m, 0)) if m in peer_n.index else 0,
                "Hist Median":       round(hist_med, 2) if hist_med else None,
                "Hist n":            int(hist_n.get(m, 0)) if m in hist_n.index else 0,
                "Implied Px (Peer)": round(imp_peer, 2) if imp_peer else None,
                "Implied Px (Hist)": round(imp_hist, 2) if imp_hist else None,
                "MoS vs Peer (%)":   round(mos_peer, 1) if mos_peer is not None else None,
                "MoS vs Hist (%)":   round(mos_hist, 1) if mos_hist is not None else None,
            })

        summary = pd.DataFrame(rows)
        # Stash context on the DF for downstream display
        summary.attrs["ticker"] = self.ticker
        summary.attrs["price"]  = price
        return summary


# ---------------------------------------------------------------------------
# Helper: pull peers from your existing screener output
# ---------------------------------------------------------------------------

def peers_from_survivors(ticker: str, survivors_df: pd.DataFrame,
                         exclude_self: bool = True) -> list:
    """Return tickers from the same sector as `ticker`, drawn from the screener output."""
    if "Sector" not in survivors_df.columns or "Ticker" not in survivors_df.columns:
        raise ValueError("survivors_df must contain 'Ticker' and 'Sector' columns")

    matches = survivors_df.loc[survivors_df["Ticker"] == ticker, "Sector"]
    if matches.empty:
        print(f"[peers] {ticker} not in survivors_df; cannot infer sector.")
        return []

    sector = matches.iloc[0]
    peers = survivors_df[survivors_df["Sector"] == sector]["Ticker"].tolist()
    if exclude_self:
        peers = [p for p in peers if p != ticker]
    return peers


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    target = "AAPL"
    peers = ["MSFT", "GOOGL", "META", "AMZN", "ORCL", "CRM"]

    rv = RelativeValuation(target)

    print(f"\n=== Current Multiples: {target} ===")
    for k, v in rv.current_multiples().items():
        if isinstance(v, (int, float)) and v is not None:
            print(f"  {k:14s}: {v:,.2f}" if v < 1e6 else f"  {k:14s}: {v:,.0f}")
        else:
            print(f"  {k:14s}: {v}")

    print(f"\n=== Historical Multiples: {target} ===")
    hist = rv.historical_multiples()
    print(hist.to_string(index=False) if not hist.empty else "  (no data)")

    print(f"\n=== Valuation Summary: {target} ===")
    summary = rv.valuation_summary(peer_tickers=peers)
    print(summary.to_string(index=False))
    print(f"\nCurrent price: ${summary.attrs.get('price'):.2f}")
