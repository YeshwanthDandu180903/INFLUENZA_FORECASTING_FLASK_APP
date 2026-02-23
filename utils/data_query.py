"""
data_query.py — Live surveillance data query layer.

All functions return plain dicts of real numbers drawn directly from the CSV
datasets.  chatbot_logic.py injects this evidence into the LLM prompt so
answers are grounded in actual CDC-style data, not guesses.
"""

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "cleaned_datasets"

_MAIN_PATH   = DATA_DIR / "influenza_modeling_dataset_2015_present.csv"
_PUB_PATH    = DATA_DIR / "public_health_lab_cleaned_dataset.csv"
_CLIN_PATH   = DATA_DIR / "clinical_labs_cleaned_dataset.csv"
_SEASON_PATH = DATA_DIR / "Virus_season.csv"
_PRE_PATH    = DATA_DIR / "pre_2015_clincal_labs.csv"

# ──────────────────────────────────────────────────────────────
# LOAD ONCE AT IMPORT  (lazy but cached)
# ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_main():
    return pd.read_csv(_MAIN_PATH)

@lru_cache(maxsize=1)
def _load_pub():
    return pd.read_csv(_PUB_PATH)

@lru_cache(maxsize=1)
def _load_clin():
    return pd.read_csv(_CLIN_PATH)

@lru_cache(maxsize=1)
def _load_season():
    return pd.read_csv(_SEASON_PATH)

@lru_cache(maxsize=1)
def _load_pre():
    return pd.read_csv(_PRE_PATH)

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def _norm_region(region: str) -> str:
    """Normalise 'region 5' → 'Region 5'."""
    if not region:
        return ""
    m = re.search(r"(\d+)", str(region))
    return f"Region {m.group(1)}" if m else str(region).strip()


def _flu_season_label(year: int, week: int) -> str:
    """Return the epidemiological flu-season string e.g. '2022-23'."""
    if week >= 40:
        return f"{year}-{str(year + 1)[-2:]}"
    return f"{year - 1}-{str(year)[-2:]}"


# ──────────────────────────────────────────────────────────────
# 1.  TREND EXPLANATION
#     "Why is Region 5 increasing?"
# ──────────────────────────────────────────────────────────────
def query_trend_explanation(region: str) -> dict:
    """
    Returns a data-rich dict explaining the current trend in *region*.
    Uses the last 8 weeks of ILI data to compute direction, magnitude,
    and dominant virus type.
    """
    region = _norm_region(region)
    df = _load_main()
    df_r = df[df["region"] == region].sort_values(["year", "week"])
    if df_r.empty:
        return {"error": f"No ILI data found for {region}."}

    # Last 8 available rows
    tail8 = df_r.tail(8)
    last4 = tail8.tail(4)["ili_weighted_pct"].tolist()
    prev4 = tail8.head(4)["ili_weighted_pct"].tolist()

    avg_last4 = round(sum(last4) / len(last4), 2) if last4 else None
    avg_prev4 = round(sum(prev4) / len(prev4), 2) if prev4 else None
    pct_change = (
        round((avg_last4 - avg_prev4) / avg_prev4 * 100, 1)
        if avg_prev4 and avg_prev4 != 0
        else 0.0
    )

    latest_row = df_r.iloc[-1]
    latest_ili = round(float(latest_row["ili_weighted_pct"]), 2)
    latest_year = int(latest_row["year"])
    latest_week = int(latest_row["week"])
    season     = _flu_season_label(latest_year, latest_week)

    # Virus type breakdown (last 4 weeks)
    a_cases = int(last4[0]) if False else int(tail8.tail(4)["total_influenza_a_cases"].sum())
    b_cases = int(tail8.tail(4)["total_influenza_b_cases"].sum())
    dominant_type = "Influenza A" if a_cases >= b_cases else "Influenza B"
    total_cases   = a_cases + b_cases
    a_pct = round(a_cases / total_cases * 100, 1) if total_cases else 0
    b_pct = round(b_cases / total_cases * 100, 1) if total_cases else 0

    # Positivity rate
    pos_rate = round(float(latest_row.get("percent_positive_overall", 0)), 1)

    # Direction label
    if pct_change > 5:
        direction = "increasing"
    elif pct_change < -5:
        direction = "decreasing"
    else:
        direction = "stable"

    # Age group pressure (latest)
    age_0_4  = int(latest_row.get("cases_age_0_4", 0))
    age_5_24 = int(latest_row.get("cases_age_5_24", 0))
    age_25_64 = int(latest_row.get("AGE 25-64", 0))
    age_65p   = int(latest_row.get("cases_age_65_plus", 0))

    return {
        "region": region,
        "season": season,
        "latest_year": latest_year,
        "latest_week": latest_week,
        "latest_ili_pct": latest_ili,
        "avg_last_4_weeks_ili": avg_last4,
        "avg_prev_4_weeks_ili": avg_prev4,
        "pct_change_4w": pct_change,
        "direction": direction,
        "dominant_virus_type": dominant_type,
        "influenza_a_cases_4w": a_cases,
        "influenza_b_cases_4w": b_cases,
        "influenza_a_pct": a_pct,
        "influenza_b_pct": b_pct,
        "positivity_rate_pct": pos_rate,
        "age_cases": {
            "0-4": age_0_4,
            "5-24": age_5_24,
            "25-64": age_25_64,
            "65+": age_65p,
        },
    }


# ──────────────────────────────────────────────────────────────
# 2.  DOMINANT STRAIN LOOKUP
#     "Which strain dominated in 2018 in Region 3?"
# ──────────────────────────────────────────────────────────────
_STRAIN_COLS = [
    ("cases_a_h1n1_pdm09",  "A/H1N1pdm09"),
    ("cases_a_h3",          "A/H3N2"),
    ("cases_a_not_subtyped","A (not subtyped)"),
    ("cases_b_unspecified", "B (unspecified)"),
    ("cases_b_victoria",    "B/Victoria"),
    ("cases_b_yamagata",    "B/Yamagata"),
    ("cases_h3n2_variant",  "H3N2 variant"),
    ("cases_a_h5",          "A/H5"),
]

# Pre-2015 mappings from pre_2015 CSV
_PRE_STRAIN_COLS = [
    ("cases_a_h1n1_2009",         "A/H1N1pdm09"),
    ("cases_a_h1",                "A/H1 (seasonal)"),
    ("cases_a_h3",                "A/H3N2"),
    ("cases_a_not_subtyped",      "A (not subtyped)"),
    ("cases_a_unable_to_subtype", "A (unable to subtype)"),
    ("cases_b_total",             "B (total)"),
    ("cases_h3n2_variant",        "H3N2 variant"),
    ("cases_a_h5",                "A/H5"),
]


def query_dominant_strain(region: str, year: int = None, season: str = None) -> dict:
    """
    Returns dominant strain breakdown for a region × year (or season label).
    If year is None, uses the most-recent available year.
    Checks post-2015 data first, then pre-2015.
    """
    region = _norm_region(region)
    pub = _load_pub()
    pre = _load_pre()

    # Determine year
    if season:
        m = re.match(r"(\d{4})", season)
        if m:
            year = int(m.group(1))

    # Try post-2015 first
    df = pub[pub["region"] == region]
    if year:
        df = df[df["year"] == year]
    cols = _STRAIN_COLS

    # Fall back to pre-2015
    if df.empty or (year and int(year) < 2015):
        df   = pre[pre["region"] == region]
        cols = _PRE_STRAIN_COLS
        if year:
            df = df[df["year"] == year]

    if df.empty:
        return {
            "error": f"No strain data for {region}"
            + (f" in {year}" if year else "") + "."
        }

    used_year = int(df["year"].iloc[0]) if year is None else int(year)
    used_season = _flu_season_label(used_year, 40)  # approximate season

    totals: dict = {}
    for col, label in cols:
        if col in df.columns:
            val = int(df[col].sum())
            if val > 0:
                totals[label] = val

    if not totals:
        return {
            "error": f"No strain case counts found for {region} in {used_year}.",
            "region": region,
            "year": used_year,
        }

    total_cases = sum(totals.values())
    dominant    = max(totals, key=totals.get)
    dom_pct     = round(totals[dominant] / total_cases * 100, 1)

    # Rank strains
    ranked = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    strain_breakdown = {
        label: {"cases": cnt, "pct": round(cnt / total_cases * 100, 1)}
        for label, cnt in ranked
    }

    # Total specimens tested
    specimens = int(df["total_specimens_tested"].sum()) if "total_specimens_tested" in df.columns else None

    return {
        "region": region,
        "year": used_year,
        "season": used_season,
        "dominant_strain": dominant,
        "dominant_pct": dom_pct,
        "total_strain_cases": total_cases,
        "total_specimens_tested": specimens,
        "strain_breakdown": strain_breakdown,
    }


# ──────────────────────────────────────────────────────────────
# 3.  SEASON-OVER-SEASON COMPARISON
#     "Compare current activity with last season."
# ──────────────────────────────────────────────────────────────
def query_season_comparison(region: str = None, current_year: int = None) -> dict:
    """
    Compares ILI metrics between the current flu season and the prior season.
    If region is None, uses the national aggregate (all regions).
    """
    df = _load_main()
    if region:
        region = _norm_region(region)
        df = df[df["region"] == region]
        label = region
    else:
        label = "All Regions (National)"

    if df.empty:
        return {"error": "No data for comparison."}

    df = df.sort_values(["year", "week"])
    max_year = int(df["year"].max())
    if current_year is None:
        current_year = max_year
    prev_year = current_year - 1

    def _season_stats(yr):
        sub = df[df["year"] == yr]
        if sub.empty:
            return None
        return {
            "year"       : yr,
            "season"     : _flu_season_label(yr, 40),
            "avg_ili"    : round(float(sub["ili_weighted_pct"].mean()), 2),
            "peak_ili"   : round(float(sub["ili_weighted_pct"].max()), 2),
            "peak_week"  : int(sub.loc[sub["ili_weighted_pct"].idxmax(), "week"]),
            "total_a"    : int(sub["total_influenza_a_cases"].sum()),
            "total_b"    : int(sub["total_influenza_b_cases"].sum()),
            "avg_pospct" : round(float(sub["percent_positive_overall"].mean()), 1),
        }

    curr_stats = _season_stats(current_year)
    prev_stats = _season_stats(prev_year)

    if curr_stats is None:
        return {"error": f"No data for {current_year} in {label}."}

    pct_diff_avg = None
    comparison   = "similar"
    if prev_stats and prev_stats["avg_ili"]:
        pct_diff_avg = round(
            (curr_stats["avg_ili"] - prev_stats["avg_ili"]) / prev_stats["avg_ili"] * 100,
            1,
        )
        if pct_diff_avg > 5:
            comparison = "higher"
        elif pct_diff_avg < -5:
            comparison = "lower"

    peak_change = None
    if prev_stats and prev_stats["peak_ili"]:
        peak_change = round(curr_stats["peak_ili"] - prev_stats["peak_ili"], 2)

    return {
        "label"           : label,
        "current"         : curr_stats,
        "previous"        : prev_stats,
        "pct_change_avg"  : pct_diff_avg,
        "peak_ili_change" : peak_change,
        "comparison"      : comparison,
    }


# ──────────────────────────────────────────────────────────────
# 4.  AGE RISK PROFILE
#     "Which age group is most at risk this season?"
# ──────────────────────────────────────────────────────────────
def query_age_risk(season_label: str = None) -> dict:
    """
    Returns age-group disease burden for the given season (or the latest).
    Uses the Virus_season dataset.
    """
    df = _load_season()
    if season_label:
        sub = df[df["season"].str.lower() == season_label.lower()]
    else:
        latest = df["season"].iloc[-1] if not df.empty else None
        sub = df[df["season"] == latest] if latest else df

    if sub.empty:
        return {"error": "No seasonal age data available."}

    agg = (
        sub.groupby("season")[
            ["cases_age_0_4", "cases_age_5_24", "cases_age_25_64", "cases_age_65_plus", "total_cases"]
        ]
        .sum()
        .reset_index()
    )
    row = agg.iloc[-1]
    total = int(row["total_cases"]) if row["total_cases"] > 0 else 1
    season_name = str(row["season"])

    age_profile = {
        "0-4":   {"cases": int(row["cases_age_0_4"]),   "pct": round(row["cases_age_0_4"] / total * 100, 1)},
        "5-24":  {"cases": int(row["cases_age_5_24"]),  "pct": round(row["cases_age_5_24"] / total * 100, 1)},
        "25-64": {"cases": int(row["cases_age_25_64"]), "pct": round(row["cases_age_25_64"] / total * 100, 1)},
        "65+":   {"cases": int(row["cases_age_65_plus"]), "pct": round(row["cases_age_65_plus"] / total * 100, 1)},
    }
    highest_age = max(age_profile, key=lambda k: age_profile[k]["pct"])

    # Dominant strain for season
    strains_df = sub.groupby("virus")["total_cases"].sum().sort_values(ascending=False)
    dominant_virus = strains_df.index[0] if not strains_df.empty else "Unknown"

    return {
        "season"        : season_name,
        "dominant_strain": dominant_virus,
        "total_cases"   : total,
        "age_profile"   : age_profile,
        "highest_risk_age": highest_age,
    }


# ──────────────────────────────────────────────────────────────
# 5.  REGIONAL MULTI-WEEK HISTORY
#     Raw recent rows for any arbitrary query
# ──────────────────────────────────────────────────────────────
def query_recent_history(region: str, n_weeks: int = 10) -> dict:
    """Returns the last n_weeks of ILI + positivity for quick table context."""
    region = _norm_region(region)
    df = _load_main()
    df_r = df[df["region"] == region].sort_values(["year", "week"]).tail(n_weeks)
    if df_r.empty:
        return {"error": f"No data for {region}."}

    rows = []
    for _, row in df_r.iterrows():
        rows.append({
            "year" : int(row["year"]),
            "week" : int(row["week"]),
            "ili"  : round(float(row["ili_weighted_pct"]), 2),
            "pos_pct": round(float(row.get("percent_positive_overall", 0)), 1),
            "flu_a": int(row.get("total_influenza_a_cases", 0)),
            "flu_b": int(row.get("total_influenza_b_cases", 0)),
        })
    return {"region": region, "weeks": rows}


# ──────────────────────────────────────────────────────────────
# 6.  ALL REGIONS SNAPSHOT (for cross-region comparison)
# ──────────────────────────────────────────────────────────────
def query_all_regions_latest() -> dict:
    """Returns the most-recent ILI for every region, sorted descending."""
    df = _load_main()
    latest_rows = (
        df.sort_values(["year", "week"])
        .groupby("region")
        .tail(1)
    )
    result = []
    for _, row in latest_rows.sort_values("ili_weighted_pct", ascending=False).iterrows():
        result.append({
            "region" : str(row["region"]),
            "year"   : int(row["year"]),
            "week"   : int(row["week"]),
            "ili_pct": round(float(row["ili_weighted_pct"]), 2),
        })
    return {"regions": result}


# ──────────────────────────────────────────────────────────────
# 7.  CONVENIENCE: FORMAT EVIDENCE FOR LLM PROMPT
# ──────────────────────────────────────────────────────────────
def format_evidence(data: dict) -> str:
    """Convert a data-query result dict into a readable plain-text block."""
    if not data or "error" in data:
        return data.get("error", "No data available.") if data else "No data available."

    lines = []
    for key, val in data.items():
        if isinstance(val, dict):
            lines.append(f"{key}:")
            for k2, v2 in val.items():
                lines.append(f"  {k2}: {v2}")
        elif isinstance(val, list):
            lines.append(f"{key}:")
            for item in val:
                if isinstance(item, dict):
                    lines.append("  " + ", ".join(f"{k}={v}" for k, v in item.items()))
                else:
                    lines.append(f"  {item}")
        else:
            lines.append(f"{key}: {val}")
    return "\n".join(lines)
