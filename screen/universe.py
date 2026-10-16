"""Knockout filters: liquidity, size, data completeness, analyzable sector (v1.1, D-001, D-010).

Reads market_snapshot, securities and the validation report; thresholds come
from config.yaml `universe`. This is the only stage that uses hard cut-offs.
"""
