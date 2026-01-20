# Equities Fundamental Analysis Tool 📈🛡️

A powerful, algorithmic stock screener and grading tool for **US and Canadian markets**.

This project automates the process of finding high-quality, fundamentally sound companies by filtering thousands of stocks through a rigorous financial sieve. It uses a hybrid approach—leveraging `yahooquery` for speed and `yfinance` for deep financial reliability to grade stocks into **Fortress**, **Strong**, or **Risky** tiers based on solvency, profitability, and capital efficiency.

## 🚀 Key Features

* **Multi-Market Support:** Dedicated scripts for **USA** (NYSE/NASDAQ) and **Canada** (TSX/TSX-V).
* **Hybrid Data Fetching:** Uses `yahooquery` for rapid initial screening of thousands of tickers and `yfinance` for deep-dive validation.
* **Automated Grading System:** Classifies stocks based on strict fundamental criteria (Z-Score, ROIC, Margins).
* **AI-Powered Analysis:** Integrated **Google Gemini 3.0** reasoning engine to generate institutional-grade research reports on top picks.
* **Specialized Scans:**
    * **🧠 Analyst Picks:** Cross-references "Fortress" stocks with Wall Street Buy ratings and upside potential.
    * **💎 Buffett Value:** Identifies deep value stocks trading below Book Value (P/B < 1) with positive ROE and manageable debt.
    * **📉 Burry "EBITDA" Value:** Finds stocks trading at an EV/EBITDA discount relative to their specific sector peers.
    * **🕵️ Insider Buying:** Detects companies with recent net positive insider accumulation.
* **Smart Caching:** Implements a JSON caching system to prevent API throttling and speed up subsequent runs.

---

## ⚙️ How It Works

The screener operates in a **4-Step Funnel**:

### 1. The Universe Fetch
* **USA:** Pulls the latest ticker list from the NASDAQ Trader exchange data.
* **Canada:** Scrapes official TSX/TSX-V indices or uses a pre-defined universe.

### 2. The "Lightweight" Sieve (Fast)
Eliminates thousands of "junk" stocks using rapid API calls.
* **Filters:** Price > $2.00, Market Cap > $300M, Current Ratio > 1.2, Positive Operating Margins.

### 3. The "Deep Dive" (Thorough)
Calculates complex metrics for the survivors:
* **Altman Z-Score:** Predicts bankruptcy risk.
* **ROIC (Return on Invested Capital):** Measures how efficiently management uses capital.
* **Interest Coverage:** Ensures the company can easily pay its interest expenses.
* **4-Year Margin Trend:** Checks for consistent profitability.

### 4. The "Deep Value" Intersection
Combines legendary value investing strategies to find the "Holy Grail" stocks:
* **The "Buffett" Filter:** Stocks trading for less than their accounting value (P/B < 1.0).
* **The "Burry" Filter:** Stocks that are cheaper than their sector average (EV/EBITDA).
* **The Intersection:** Stocks that pass **BOTH** filters are saved to `Deep_Value_Gems.csv`.

---

## 🛠️ Installation & Usage

### 1. Prerequisites
Ensure you have Python installed along with the following libraries:

```bash
pip install pandas numpy requests yahooquery yfinance finvizfinance plotly google-genai
