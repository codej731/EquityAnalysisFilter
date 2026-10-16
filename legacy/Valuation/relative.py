"""
Relative Valuation Module — Phase 1 (yahooquery edition)
=========================================================
Computes multiples-based valuation for a single ticker, with comparison to:
  (a) Sector peer median (bulk-fetched in one async call)
  (b) Company's own ~5-year historical median

Multiples covered:
  - Current:    P/E (TTM), Fwd P/E, PEG, EV/EBITDA, EV/EBIT, EV/Sales, P/B, P/S, P/FCF
  - Historical: P/E, EV/EBITDA, EV/EBIT, P/B, P/S, P/FCF

Why yahooquery:
  - Bulk async peer fetching (~10x faster than per-ticker yfinance loop)
  - Explicit TTM rows in statement data (vs yfinance's annual-only DataFrames)
  - Direct FreeCashFlow line item (no OCF+capex computation needed in many cases)
  - Already used in your screener (filters.py), so one library to maintain

Why hybrid with yfinance:
  - .history() for adjusted prices is more battle-tested
  - Historical multiples need price lookups at fiscal year-ends; no bulk advantage for one ticker

Usage:
    rv = RelativeValuation("AAPL")
    summary = rv.valuation_summary(peer_tickers=["MSFT", "GOOGL", "META", "AMZN"])
    print(summary)
"""

import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from yahooquery import Ticker as YQTicker


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
# yahooquery field-name dictionaries (CamelCase, mirroring Yahoo's raw fields)
# ---------------------------------------------------------------------------

# Income statement
KEYS_IS = {
    "revenue":    ["TotalRevenue", "Revenue"],
    "ebit":       ["EBIT", "OperatingIncome"],
    "ebitda":     ["EBITDA", "NormalizedEBITDA"],
    "net_income": ["NetIncome", "NetIncomeCommonStockholders"],
    "shares":     ["DilutedAverageShares", "BasicAverageShares"],
}

# Balance sheet
KEYS_BS = {
    "equity": ["StockholdersEquity", "TotalEquityGrossMinorityInterest"],
    "debt":   ["TotalDebt", "LongTermDebt"],
    "cash":   ["CashAndCashEquivalents",
               "CashCashEquivalentsAndShortTermInvestments"],
}

# Cash flow
KEYS_CF = {
    "ocf":       ["OperatingCashFlow",
                  "CashFlowFromContinuingOperatingActivities"],
    "capex":     ["CapitalExpenditure"],
    "fcf":       ["FreeCashFlow"],   # yahooquery often provides this directly
    "dep_amort": ["DepreciationAndAmortization", "ReconciledDepreciation"],
}


# ---------------------------------------------------------------------------
# Statement extraction helpers
# ---------------------------------------------------------------------------

def _select_ticker_rows(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Pull rows for a single ticker from a yahooquery statement DataFrame.

    yahooquery returns the symbol as the index (single ticker) or as a level
    of a MultiIndex (multiple tickers). Normalize to a flat DataFrame.
    """
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()  # yahooquery returns strings on error

    if isinstance(df.index, pd.MultiIndex):
        if ticker in df.index.get_level_values(0):
            return df.loc[ticker].copy()
        return pd.DataFrame()
    if df.index.name == "symbol":
        try:
            sub = df.loc[ticker]
            return sub if isinstance(sub, pd.DataFrame) else sub.to_frame().T
        except KeyError:
            return pd.DataFrame()
    return df.copy()


def _yq_value(df: pd.DataFrame, keys: list, period: str = "latest") -> Optional[float]:
    """Extract a value from a yahooquery statement DataFrame.

    period:
      'latest'        -> last row chronologically (TTM if present, else most recent annual)
      'ttm'           -> TTM row only (returns None if not present)
      'annual_latest' -> most recent annual fiscal period (excludes TTM)
      int             -> Nth row in chronological order (0 = oldest)
    """
    if df is None or df.empty:
        return None

    work = df.copy()
    if "asOfDate" in work.columns:
        work = work.sort_values("asOfDate")

    try:
        if period == "ttm":
            if "periodType" in work.columns:
                ttm_rows = work[work["periodType"] == "TTM"]
                if ttm_rows.empty:
                    return None
                row = ttm_rows.iloc[-1]
            else:
                return None
        elif period == "annual_latest":
            if "periodType" in work.columns:
                annual = work[work["periodType"] == "12M"]
                if annual.empty:
                    return None
                row = annual.iloc[-1]
            else:
                row = work.iloc[-1]
        elif period == "latest":
            row = work.iloc[-1]
        elif isinstance(period, int):
            row = work.iloc[period]
        else:
            return None
    except Exception:
        return None

    for k in keys:
        if k in row.index:
            val = row[k]
            if pd.notna(val):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
    return None


def _annual_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return only annual (12M) periods from a yahooquery statement, sorted by date."""
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    if "periodType" in work.columns:
        work = work[work["periodType"] == "12M"]
    if "asOfDate" in work.columns:
        work = work.sort_values("asOfDate")
    return work


# ---------------------------------------------------------------------------
# Pure function: compute multiples from already-fetched raw data
# Used by both single-ticker and bulk-peer paths so logic stays in one place
# ---------------------------------------------------------------------------

def _compute_current_multiples(modules: dict,
                                income_df: pd.DataFrame,
                                balance_df: pd.DataFrame,
                                cashflow_df: pd.DataFrame) -> dict:
    """Compute current multiples from yahooquery modules + statement DataFrames.

    All data must already be filtered to the single ticker of interest.
    Returns dict with None for any multiple that couldn't be computed.
    """
    def _safe(key):
        v = modules.get(key) if isinstance(modules, dict) else None
        return v if isinstance(v, dict) else {}

    price_mod = _safe("price")
    sd = _safe("summaryDetail")
    ks = _safe("defaultKeyStatistics")

    price = price_mod.get("regularMarketPrice")
    market_cap = price_mod.get("marketCap")
    enterprise_value = ks.get("enterpriseValue")

    # Directly available multiples
    pe_ttm    = sd.get("trailingPE")
    pe_fwd    = sd.get("forwardPE") or ks.get("forwardPE")
    peg       = ks.get("pegRatio") or ks.get("trailingPegRatio")
    ev_ebitda = ks.get("enterpriseToEbitda")
    ev_sales  = ks.get("enterpriseToRevenue")
    pb        = ks.get("priceToBook")
    ps        = sd.get("priceToSalesTrailing12Months")

    # EV/EBIT — not in modules, compute (TTM preferred)
    ebit = _yq_value(income_df, KEYS_IS["ebit"], "ttm") \
        or _yq_value(income_df, KEYS_IS["ebit"], "annual_latest")
    ev_ebit = (enterprise_value / ebit) if (enterprise_value and ebit and ebit > 0) else None

    # P/FCF — try yahooquery's direct FreeCashFlow first
    fcf = _yq_value(cashflow_df, KEYS_CF["fcf"], "ttm") \
        or _yq_value(cashflow_df, KEYS_CF["fcf"], "annual_latest")
    if fcf is None:
        ocf = _yq_value(cashflow_df, KEYS_CF["ocf"], "latest")
        capex = _yq_value(cashflow_df, KEYS_CF["capex"], "latest")
        if ocf is not None and capex is not None:
            fcf = ocf + capex if capex < 0 else ocf - capex
    p_fcf = (market_cap / fcf) if (market_cap and fcf and fcf > 0) else None

    return {
        "Price":      price,
        "Market Cap": market_cap,
        "P/E (TTM)":  pe_ttm,
        "Fwd P/E":    pe_fwd,
        "PEG":        peg,
        "EV/EBITDA":  ev_ebitda,
        "EV/EBIT":    ev_ebit,
        "EV/Sales":   ev_sales,
        "P/B":        pb,
        "P/S":        ps,
        "P/FCF":      p_fcf,
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RelativeValuation:
    """Compute relative valuation multiples for a single ticker."""

    MULTIPLES_FOR_IMPLIED_PRICE = ["P/E (TTM)", "EV/EBITDA", "EV/EBIT", "P/B", "P/S", "P/FCF"]
    MODULES_TO_FETCH = "summaryDetail price financialData defaultKeyStatistics summaryProfile"

    def __init__(self, ticker: str, use_cache: bool = True):
        self.ticker = ticker.upper()
        self.use_cache = use_cache
        self._yq = YQTicker(self.ticker)
        self._yf = yf.Ticker(self.ticker)  # used only for historical price lookups

        # Lazy-loaded
        self._modules = None
        self._income = None
        self._balance = None
        self._cashflow = None

    # -- Lazy property loaders ----------------------------------------------

    @property
    def modules(self) -> dict:
        if self._modules is None:
            try:
                data = self._yq.get_modules(self.MODULES_TO_FETCH)
                self._modules = data.get(self.ticker, {}) if isinstance(data, dict) else {}
            except Exception as e:
                print(f"[{self.ticker}] modules fetch failed: {e}")
                self._modules = {}
        return self._modules

    @property
    def income(self) -> pd.DataFrame:
        if self._income is None:
            try:
                df = self._yq.income_statement(frequency="a", trailing=True)
                self._income = _select_ticker_rows(df, self.ticker)
            except Exception as e:
                print(f"[{self.ticker}] income_statement failed: {e}")
                self._income = pd.DataFrame()
        return self._income

    @property
    def balance(self) -> pd.DataFrame:
        if self._balance is None:
            try:
                df = self._yq.balance_sheet(frequency="a")
                self._balance = _select_ticker_rows(df, self.ticker)
            except Exception as e:
                print(f"[{self.ticker}] balance_sheet failed: {e}")
                self._balance = pd.DataFrame()
        return self._balance

    @property
    def cashflow(self) -> pd.DataFrame:
        if self._cashflow is None:
            try:
                df = self._yq.cash_flow(frequency="a", trailing=True)
                self._cashflow = _select_ticker_rows(df, self.ticker)
            except Exception as e:
                print(f"[{self.ticker}] cash_flow failed: {e}")
                self._cashflow = pd.DataFrame()
        return self._cashflow

    # -- Current multiples --------------------------------------------------

    def current_multiples(self) -> dict:
        return _compute_current_multiples(
            self.modules, self.income, self.balance, self.cashflow
        )

    # -- Historical multiples (annual fiscal periods only) ------------------

    def historical_multiples(self) -> pd.DataFrame:
        """Fiscal year-end multiples for ~4 prior years (yahooquery 12M rows only)."""
        income_ann = _annual_rows(self.income)
        balance_ann = _annual_rows(self.balance)
        cashflow_ann = _annual_rows(self.cashflow)

        if income_ann.empty or balance_ann.empty:
            return pd.DataFrame()

        fiscal_dates = pd.to_datetime(income_ann["asOfDate"]).sort_values()

        try:
            buffer = pd.Timedelta(days=30)
            prices = self._yf.history(
                start=fiscal_dates.min() - buffer,
                end=fiscal_dates.max() + buffer,
                auto_adjust=True,
            )
        except Exception as e:
            print(f"[{self.ticker}] historical price fetch failed: {e}")
            return pd.DataFrame()

        if prices.empty:
            return pd.DataFrame()

        prices_idx = prices.index.tz_localize(None) if prices.index.tz else prices.index

        def _row_for_date(df, date):
            if "asOfDate" not in df.columns:
                return None
            matches = df[pd.to_datetime(df["asOfDate"]) == date]
            return matches.iloc[0] if not matches.empty else None

        def _val(row, keys):
            if row is None:
                return None
            for k in keys:
                if k in row.index and pd.notna(row[k]):
                    try:
                        return float(row[k])
                    except (TypeError, ValueError):
                        continue
            return None

        records = []
        for fy_end in fiscal_dates:
            try:
                fy_naive = pd.Timestamp(fy_end).tz_localize(None) \
                    if pd.Timestamp(fy_end).tz else pd.Timestamp(fy_end)

                closest_idx = (abs(prices_idx - fy_naive)).argmin()
                price = float(prices["Close"].iloc[closest_idx])

                is_row = _row_for_date(income_ann, fy_end)
                bs_row = _row_for_date(balance_ann, fy_end)
                cf_row = _row_for_date(cashflow_ann, fy_end)

                revenue    = _val(is_row, KEYS_IS["revenue"])
                ebit       = _val(is_row, KEYS_IS["ebit"])
                net_income = _val(is_row, KEYS_IS["net_income"])
                shares     = _val(is_row, KEYS_IS["shares"])
                equity     = _val(bs_row, KEYS_BS["equity"])
                debt       = _val(bs_row, KEYS_BS["debt"]) or 0
                cash       = _val(bs_row, KEYS_BS["cash"]) or 0
                fcf        = _val(cf_row, KEYS_CF["fcf"])
                if fcf is None:
                    ocf   = _val(cf_row, KEYS_CF["ocf"])
                    capex = _val(cf_row, KEYS_CF["capex"])
                    if ocf is not None and capex is not None:
                        fcf = ocf + capex if capex < 0 else ocf - capex
                dep        = _val(cf_row, KEYS_CF["dep_amort"]) or 0

                if not shares or shares <= 0:
                    continue

                market_cap = price * shares
                enterprise_value = market_cap + debt - cash
                ebitda = (ebit + dep) if ebit is not None else None

                pe        = (market_cap / net_income)     if (net_income and net_income > 0) else None
                pb        = (market_cap / equity)         if (equity and equity > 0)         else None
                ps        = (market_cap / revenue)        if (revenue and revenue > 0)       else None
                ev_ebitda = (enterprise_value / ebitda)   if (ebitda and ebitda > 0)         else None
                ev_ebit   = (enterprise_value / ebit)     if (ebit and ebit > 0)             else None
                p_fcf     = (market_cap / fcf)            if (fcf and fcf > 0)               else None

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

    # -- Peer multiples (THE bulk-async win) --------------------------------

    @staticmethod
    def peer_multiples(peer_tickers: list, use_cache: bool = True) -> pd.DataFrame:
        """Fetch current multiples for all peers in one bulk async call.

        This is the main reason to use yahooquery over yfinance — one async
        request instead of N sequential ones.
        """
        cache = _load_cache() if use_cache else {}
        now = time.time()

        cached_rows = []
        to_fetch = []
        for tk in peer_tickers:
            tk = tk.upper()
            cached = cache.get(tk)
            if cached and (now - cached.get("_ts", 0)) < CACHE_EXPIRY_HOURS * 3600:
                row = {k: v for k, v in cached.items() if k != "_ts"}
                row["Ticker"] = tk
                cached_rows.append(row)
            else:
                to_fetch.append(tk)

        rows = list(cached_rows)

        if to_fetch:
            print(f"Bulk-fetching {len(to_fetch)} peers via yahooquery (async)...")
            try:
                yq = YQTicker(to_fetch, asynchronous=True)
                modules_all = yq.get_modules(RelativeValuation.MODULES_TO_FETCH)
                income_all = yq.income_statement(frequency="a", trailing=True)
                balance_all = yq.balance_sheet(frequency="a")
                cashflow_all = yq.cash_flow(frequency="a", trailing=True)

                if not isinstance(modules_all, dict):
                    modules_all = {}

                for tk in to_fetch:
                    tk_modules = modules_all.get(tk, {}) if isinstance(modules_all.get(tk), dict) else {}
                    if not tk_modules:
                        print(f"  [skip] {tk}: no module data")
                        continue

                    tk_income = _select_ticker_rows(income_all, tk)
                    tk_balance = _select_ticker_rows(balance_all, tk)
                    tk_cashflow = _select_ticker_rows(cashflow_all, tk)

                    m = _compute_current_multiples(
                        tk_modules, tk_income, tk_balance, tk_cashflow
                    )
                    m["Ticker"] = tk
                    rows.append(m)
                    cache[tk] = {**m, "_ts": now}

                if use_cache:
                    _save_cache(cache)

            except Exception as e:
                print(f"[peer bulk fetch] failed: {e}")

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("Ticker")

    # -- Implied price back-solver ------------------------------------------

    def _implied_price(self, multiple_name: str, mult_value: Optional[float],
                       fundamentals: dict) -> Optional[float]:
        """Back-solve the implied per-share price from a target multiple value."""
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
                return (mult_value * ebitda - debt + cash) / shares
            if multiple_name == "EV/EBIT" and ebit and ebit > 0:
                return (mult_value * ebit - debt + cash) / shares
        except Exception:
            return None
        return None

    # -- Triangulation summary (main output) --------------------------------

    def valuation_summary(self, peer_tickers: Optional[list] = None) -> pd.DataFrame:
        """Per-multiple table: current vs peer median vs own median, with implied prices."""
        current = self.current_multiples()
        hist = self.historical_multiples()
        price = current.get("Price")

        # Bundle fundamentals once (TTM-preferred) for implied-price back-solving
        ebit = _yq_value(self.income, KEYS_IS["ebit"], "ttm") \
            or _yq_value(self.income, KEYS_IS["ebit"], "annual_latest")
        dep = _yq_value(self.cashflow, KEYS_CF["dep_amort"], "ttm") \
            or _yq_value(self.cashflow, KEYS_CF["dep_amort"], "annual_latest") or 0

        fcf = _yq_value(self.cashflow, KEYS_CF["fcf"], "ttm") \
            or _yq_value(self.cashflow, KEYS_CF["fcf"], "annual_latest")
        if fcf is None:
            ocf = _yq_value(self.cashflow, KEYS_CF["ocf"], "latest")
            capex = _yq_value(self.cashflow, KEYS_CF["capex"], "latest")
            if ocf is not None and capex is not None:
                fcf = ocf + capex if capex < 0 else ocf - capex

        ks = self.modules.get("defaultKeyStatistics", {}) if isinstance(self.modules, dict) else {}
        shares = ks.get("sharesOutstanding") if isinstance(ks, dict) else None
        if not shares:
            shares = _yq_value(self.income, KEYS_IS["shares"], "annual_latest")

        fundamentals = {
            "shares":     shares,
            "net_income": _yq_value(self.income, KEYS_IS["net_income"], "ttm")
                          or _yq_value(self.income, KEYS_IS["net_income"], "annual_latest"),
            "revenue":    _yq_value(self.income, KEYS_IS["revenue"], "ttm")
                          or _yq_value(self.income, KEYS_IS["revenue"], "annual_latest"),
            "ebit":       ebit,
            "ebitda":     (ebit + dep) if ebit is not None else None,
            "equity":     _yq_value(self.balance, KEYS_BS["equity"], "annual_latest"),
            "debt":       _yq_value(self.balance, KEYS_BS["debt"], "annual_latest") or 0,
            "cash":       _yq_value(self.balance, KEYS_BS["cash"], "annual_latest") or 0,
            "fcf":        fcf,
        }

        peer_medians = pd.Series(dtype=float)
        peer_n = pd.Series(dtype=int)
        if peer_tickers:
            peers = self.peer_multiples(peer_tickers)
            if not peers.empty:
                cols = [c for c in self.MULTIPLES_FOR_IMPLIED_PRICE if c in peers.columns]
                peer_medians = peers[cols].median(numeric_only=True)
                peer_n = peers[cols].notna().sum()

        hist_medians = pd.Series(dtype=float)
        hist_n = pd.Series(dtype=int)
        if not hist.empty:
            cols = [c for c in self.MULTIPLES_FOR_IMPLIED_PRICE if c in hist.columns]
            hist_medians = hist[cols].median(numeric_only=True)
            hist_n = hist[cols].notna().sum()

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
        summary.attrs["ticker"] = self.ticker
        summary.attrs["price"]  = price
        return summary


# ---------------------------------------------------------------------------
# Helper: pull peers from your existing screener output
# ---------------------------------------------------------------------------

def peers_from_survivors(ticker: str, survivors_df: pd.DataFrame,
                         exclude_self: bool = True) -> list:
    """Return tickers from the same sector as `ticker`, drawn from screener output."""
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
