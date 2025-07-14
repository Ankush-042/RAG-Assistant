import pandas as pd
import numpy as np
import re
from typing import List, Optional

# Analytical query parser (very simple for demo; can be replaced with ML intent classifier)
def parse_analysis_query(query: str):
    query = query.lower()
    # Common patterns
    if any(word in query for word in ["average", "mean"]):
        return "mean"
    if "sum" in query:
        return "sum"
    if "maximum" in query or "max" in query:
        return "max"
    if "minimum" in query or "min" in query:
        return "min"
    if "count" in query or "how many" in query:
        return "count"
    if "trend" in query or "over time" in query:
        return "trend"
    if "predict" in query:
        return "predict"
    return None

# Main analysis function
def analyze_structured_data(query: str, dfs: List[pd.DataFrame]) -> Optional[str]:
    intent = parse_analysis_query(query)
    if not intent:
        return None
    # Try to find the relevant column
    for df in dfs:
        for col in df.columns:
            if col.lower() in query.lower():
                # Found relevant column
                if intent == "mean":
                    try:
                        mean_val = df[col].astype(float).mean()
                        return f"Average {col}: {mean_val:.2f}"
                    except:
                        continue
                if intent == "sum":
                    try:
                        sum_val = df[col].astype(float).sum()
                        return f"Sum of {col}: {sum_val:.2f}"
                    except:
                        continue
                if intent == "max":
                    try:
                        max_val = df[col].astype(float).max()
                        return f"Maximum {col}: {max_val:.2f}"
                    except:
                        continue
                if intent == "min":
                    try:
                        min_val = df[col].astype(float).min()
                        return f"Minimum {col}: {min_val:.2f}"
                    except:
                        continue
                if intent == "count":
                    count_val = df[col].count()
                    return f"Count for {col}: {count_val}"
                if intent == "trend":
                    # Simple trend: show first and last value
                    try:
                        first, last = df[col].iloc[0], df[col].iloc[-1]
                        return f"Trend for {col}: {first} → {last}"
                    except:
                        continue
    return None
