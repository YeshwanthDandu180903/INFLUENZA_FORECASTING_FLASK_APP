from pathlib import Path

from flask import Flask, render_template, request
from collections import deque
import pandas as pd
import numpy as np
import joblib
from utils.chatbot_logic import region_aware_response

# ===============================
# CONFIG
# ===============================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "cleaned_datasets"
DATA_MAIN = DATA_DIR / "influenza_modeling_dataset_2015_present.csv"
DATA_CLINICAL = DATA_DIR / "clinical_labs_cleaned_dataset.csv"
DATA_PUBLIC = DATA_DIR / "public_health_lab_cleaned_dataset.csv"
DATA_SEASON = DATA_DIR / "Virus_season.csv"
DATA_PEDIATRIC = DATA_DIR / "Characteristics2.csv"
DATA_PRE2015 = DATA_DIR / "pre_2015_clincal_labs.csv"

RF_MODEL_PATH = BASE_DIR / "models" / "models_ml" / "rf_forecast_model.pkl"

TARGET = "ili_weighted_pct"
LAGS = [1, 2, 4, 8]

# ===============================
# LOAD DATA
# ===============================
df_main = pd.read_csv(DATA_MAIN).sort_values(["region", "year", "week"])
df_clinical = pd.read_csv(DATA_CLINICAL)
df_public = pd.read_csv(DATA_PUBLIC)
df_season = pd.read_csv(DATA_SEASON)
df_pre2015 = pd.read_csv(DATA_PRE2015)

# Load pediatric death characteristics (optional file)
if DATA_PEDIATRIC.exists():
    df_pediatric = pd.read_csv(DATA_PEDIATRIC, skiprows=1)  # Skip header row
    df_pediatric.columns = ['SEASON', 'CHARACTERISTIC', 'GROUP', 'COUNT', 'PERCENT']
else:
    df_pediatric = pd.DataFrame(columns=['SEASON', 'CHARACTERISTIC', 'GROUP', 'COUNT', 'PERCENT'])


# ===============================
# HELPER: DOMINANT VIRUS (LATEST WINDOW)
# ===============================
def infer_dominant_virus(region, window=4):
    df = df_public[df_public['region'] == region].sort_values(['year', 'week'])
    if df.empty:
        return None
    recent = df.tail(window)
    totals = {
        'A (H1N1)': recent.get('cases_a_h1n1_pdm09', pd.Series([0])).sum(),
        'A (H3)': recent.get('cases_a_h3', pd.Series([0])).sum(),
        'B (Victoria)': recent.get('cases_b_victoria', pd.Series([0])).sum(),
        'B (Yamagata)': recent.get('cases_b_yamagata', pd.Series([0])).sum()
    }
    top = max(totals, key=totals.get)
    return top if totals[top] > 0 else None

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load(RF_MODEL_PATH)

# ===============================
# REGION MAP
# ===============================
REGION_MAP = {
    "Region 1": ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont"],
    "Region 2": ["New Jersey", "New York", "Puerto Rico", "U.S. Virgin Islands"],
    "Region 3": ["Delaware", "District of Columbia", "Maryland", "Pennsylvania", "Virginia", "West Virginia"],
    "Region 4": ["Alabama", "Florida", "Georgia", "Kentucky", "Mississippi", "North Carolina", "South Carolina", "Tennessee"],
    "Region 5": ["Illinois", "Indiana", "Michigan", "Minnesota", "Ohio", "Wisconsin"],
    "Region 6": ["Arkansas", "Louisiana", "New Mexico", "Oklahoma", "Texas"],
    "Region 7": ["Iowa", "Kansas", "Missouri", "Nebraska"],
    "Region 8": ["Colorado", "Montana", "North Dakota", "South Dakota", "Utah", "Wyoming"],
    "Region 9": ["Arizona", "California", "Hawaii", "Nevada"],
    "Region 10": ["Alaska", "Idaho", "Oregon", "Washington"]
}


# ===============================
# FLASK
# ===============================
app = Flask(__name__)

# In-memory chat history (per app instance). For multi-user, move to session storage.
chat_history = deque(maxlen=12)

# ===============================
# FORECAST FUNCTION
# ===============================
def forecast_future(df_region, horizon):
    history = df_region[TARGET].tolist()
    forecasts = []
    feature_cols = [f"{TARGET}_lag_{l}" for l in LAGS]

    for _ in range(horizon):
        X = pd.DataFrame(
            [[history[-l] for l in LAGS]],
            columns=feature_cols
        )
        pred = model.predict(X)[0]
        forecasts.append(round(float(pred), 2))
        history.append(pred)

    return forecasts, history[-20:]


def build_region_snapshot(region, horizon=1):
    """Create a small, authoritative snapshot for a given region.

    Returns dict with keys: region, ili_forecast_pct, severity, trend, dominant_virus, age_risks
    If the region has no data, values will be None or empty where appropriate.
    """
    if not region:
        return {
            'region': None,
            'ili_forecast_pct': None,
            'severity': None,
            'trend': None,
            'dominant_virus': None,
            'age_risks': None
        }

    df_r = df_main[df_main['region'] == region]
    if df_r.empty:
        return {
            'region': region,
            'ili_forecast_pct': None,
            'severity': None,
            'trend': None,
            'dominant_virus': None,
            'age_risks': None
        }

    # Observed history
    observed = df_r[TARGET].tolist()
    last_obs = observed[-1] if observed else None

    # Forecast (keep forecasting logic unchanged)
    try:
        forecast_vals, _ = forecast_future(df_r, horizon)
        ili_forecast = float(forecast_vals[0]) if forecast_vals else None
    except Exception:
        ili_forecast = None

    # Severity from forecast if available
    if ili_forecast is None:
        severity = None
    elif ili_forecast < 2:
        severity = 'Low'
    elif ili_forecast < 5:
        severity = 'Medium'
    else:
        severity = 'High'

    # Trend: compare last observed to recent average (last 4 weeks)
    trend = None
    if len(observed) >= 2:
        recent = observed[-4:] if len(observed) >= 4 else observed
        recent_avg = sum(recent) / len(recent) if recent else None
        if recent_avg and last_obs is not None:
            if last_obs > recent_avg * 1.05:
                trend = 'increasing'
            elif last_obs < recent_avg * 0.95:
                trend = 'decreasing'
            else:
                trend = 'stable'

    dominant = infer_dominant_virus(region)

    # Age risks from season data (best-effort)
    age_risks = None
    try:
        latest_season = df_season['season'].max()
        row = df_season[df_season['season'] == latest_season] \
              .sort_values('total_cases', ascending=False).iloc[0]
        age_risks = {
            '0-4': float(row.get('cases_age_0_4_pct', 0)),
            '5-24': float(row.get('cases_age_5_24_pct', 0)),
            '25-64': float(row.get('cases_age_25_64_pct', 0)),
            '65+': float(row.get('cases_age_65_plus_pct', 0)),
        }
    except Exception:
        age_risks = None

    # Add jurisdictions (states) for the region when available
    jurisdictions = REGION_MAP.get(region) if region in REGION_MAP else None

    return {
        'region': region,
        'ili_forecast_pct': ili_forecast,
        'severity': severity,
        'trend': trend,
        'dominant_virus': dominant,
        'age_risks': age_risks,
        'jurisdictions': jurisdictions
    }

# ===============================
# AI EXPLANATION ENGINE
# ===============================
def generate_ai_explanation(forecast, severity, horizon, region, observed_history):
    """
    Rule-based AI explanation generator
    Provides natural language insights without external APIs
    """
    latest = forecast[0]

    # Use observed history (exclude forecasted values) for comparisons
    last_observed = observed_history[-1] if observed_history else None
    trend = "stable"
    if last_observed is not None:
        trend = "increasing" if latest > last_observed else "decreasing" if latest < last_observed else "stable"

    # Calculate trend strength using recent observed weeks only
    recent_window = observed_history[-4:] if len(observed_history) >= 1 else observed_history
    recent_avg = sum(recent_window) / len(recent_window) if recent_window else 0
    change_pct = ((latest - recent_avg) / recent_avg) * 100 if recent_avg > 0 else 0
    
    # Base explanation by severity
    if severity == "Low":
        base = f"The forecast indicates low influenza activity in {region} with ILI at {latest}%."
        action = "Continue routine surveillance. This is a favorable public health situation."
    elif severity == "Medium":
        base = f"Moderate influenza activity is forecasted for {region} with ILI at {latest}%."
        action = "Enhanced monitoring recommended. Healthcare facilities should prepare for increased patient volume."
    else:
        base = f"High influenza activity detected in {region} with ILI at {latest}%."
        action = "Urgent: Implement enhanced surveillance protocols. Consider public health advisories."
    
    # Trend analysis
    if abs(change_pct) < 5:
        trend_text = "Activity levels are relatively stable compared to recent weeks."
    elif change_pct > 5:
        trend_text = f"Activity is trending upward ({abs(change_pct):.1f}% increase), suggesting acceleration."
    else:
        trend_text = f"Activity is declining ({abs(change_pct):.1f}% decrease), indicating improvement."
    
    # Confidence statement
    if horizon <= 2:
        confidence = "High confidence: Short-term forecasts benefit from recent lag patterns."
    elif horizon == 3:
        confidence = "Medium confidence: 3-week forecasts have moderate reliability."
    else:
        confidence = "Moderate confidence: Extended horizons increase uncertainty."
    
    # Risk group alert (based on pediatric data)
    risk_alert = ""
    if severity in ["Medium", "High"]:
        risk_alert = "Children with neurologic or pulmonary conditions face elevated risk during high activity periods."
    
    # Precautions based on severity
    precautions = [
        "Wash hands frequently with soap and water for at least 20 seconds.",
        "Avoid close contact with people who are sick."
    ]
    
    if severity == "Medium":
        precautions.extend([
            "Consider wearing a mask in crowded indoor public spaces.",
            "Ensure your flu vaccination is up to date."
        ])
    elif severity == "High":
        precautions.extend([
            "Avoid large gatherings if you are in a high-risk group.",
            "Seek medical attention immediately if you experience severe symptoms.",
            "Wear a high-quality mask (N95/KN95) in public settings."
        ])
    else:
        precautions.append("Maintain healthy habits: sleep, exercise, and nutrition.")

    # Plain-language brief for non-technical users
    # Very short plain headline and action for non-technical users
    headline_map = {
        "Low": "Low sickness expected next week.",
        "Medium": "Some increase in sick visits expected.",
        "High": "Significant increase in sick visits expected."
    }
    action_map = {
        "Low": "No extra steps — continue normal monitoring.",
        "Medium": "Prepare staff and supplies; watch trends closely.",
        "High": "Activate response plans; inform stakeholders and consider public messaging."
    }

    # Compose a short, plain-language AI card focused on non-technical users.
    # Headline: concise statement of what will happen
    if severity == "Low":
        if change_pct > 5:
            headline = "Mild rise in sick visits expected."
        elif change_pct < -5:
            headline = "Sick visits likely to decline."
        else:
            headline = "Low sickness expected next week."
    elif severity == "Medium":
        if change_pct > 5:
            headline = "Some increase in sick visits expected."
        elif change_pct < -5:
            headline = "Sick visits may decrease slightly."
        else:
            headline = "Moderate sick visits expected."
    else:
        if change_pct > 0:
            headline = "Significant increase in sick visits likely."
        else:
            headline = "High levels of sick visits expected."

    # Key insight: one-line why (trend and recent comparison)
    if abs(change_pct) < 5:
        insight = f"Levels are similar to recent weeks in {region}."
    elif change_pct > 5:
        insight = f"Cases are rising compared to recent weeks in {region}."
    else:
        insight = f"Cases are declining compared to recent weeks in {region}."

    # Action: single, plain-language recommendation
    action = action_map.get(severity, action_map['Low'])

    # Confidence: short, user-friendly line
    if horizon <= 2:
        conf_text = "High confidence (short-term forecast)."
    elif horizon == 3:
        conf_text = "Medium confidence (3-week outlook)."
    else:
        conf_text = "Moderate confidence (longer horizon)."

    # Return only the compact AI card fields for UI rendering
    return {
        "headline": headline,
        "insight": insight,
        "action": action,
        "confidence": conf_text
    }
    


# ===============================
# UNIQUE FEATURE FUNCTIONS
# ===============================

def get_virus_evolution_timeline(region):
    """
    FEATURE 1: Virus Subtype Evolution Over Time
    Shows how H1N1, H3N2, and B strains have shifted dominance
    """
    df_reg = df_public[df_public['region'] == region].copy()
    df_reg = df_reg[(df_reg['year'] >= 2019) & (df_reg['year'] <= 2025)]
    
    timeline = df_reg.groupby('year').agg({
        'cases_a_h1n1_pdm09': 'sum',
        'cases_a_h3': 'sum',
        'cases_b_victoria': 'sum',
        'cases_b_yamagata': 'sum'
    }).reset_index()
    
    return timeline.to_dict('records')


def get_lab_testing_efficiency(region):
    """
    FEATURE 2: Clinical vs Public Health Lab Efficiency
    Compares positivity rates between lab types to identify testing gaps
    """
    clinical = df_clinical[df_clinical['region'] == region].tail(20)
    public   = df_public[df_public['region'] == region].tail(20)

    # Public lab: compute positivity from case columns / total specimens
    case_cols = [c for c in public.columns if c.startswith('cases_')]
    public_cases      = public[case_cols].sum(axis=1)
    public_total      = public['total_specimens_tested'].replace(0, float('nan'))
    public_positivity = (public_cases / public_total * 100).mean()
    public_positivity = float(public_positivity) if not pd.isna(public_positivity) else 0.0

    clinical_positivity = float(clinical['percent_positive_overall'].mean()) \
        if not clinical.empty else 0.0
    clinical_specimens  = int(clinical['total_specimens_tested'].sum()) \
        if 'total_specimens_tested' in clinical.columns else 0
    public_specimens    = int(public['total_specimens_tested'].sum()) \
        if not public.empty else 0

    efficiency = {
        'clinical_avg_positivity': clinical_positivity,
        'public_avg_positivity':   public_positivity,
        'clinical_specimens':      clinical_specimens,
        'public_specimens':        public_specimens,
        'efficiency_ratio': round(clinical_positivity / public_positivity, 2)
            if public_positivity > 0 else 1.0
    }
    return efficiency


def short_forecast_blurb(region, horizon=2):
    """Generates a short forecast blurb for the reports page."""
    df_region = df_main[df_main["region"] == region]
    if df_region.empty:
        return {"ili": [], "severity": "Unknown", "text": "No data for this region."}

    forecast, history = forecast_future(df_region, horizon)
    latest = forecast[0] if forecast else None
    severity = "High" if latest and latest > 5 else "Medium" if latest and latest >= 2 else "Low"

    text = f"ILI next week: {latest}% ({severity} risk)." if latest is not None else "No forecast available."
    return {"ili": forecast, "severity": severity, "text": text}


def forecast_series_for_report(region, horizon=3, history_window=8):
    """Returns recent actuals and forecast values for charting on the reports page."""
    df_region = df_main[df_main["region"] == region]
    if df_region.empty:
        return {"recent": [], "forecast": []}

    full = df_region[TARGET].tolist()
    recent = full[-history_window:] if len(full) >= history_window else full

    forecast, _ = forecast_future(df_region, horizon)
    return {"recent": recent, "forecast": forecast}


def forecast_specimen_volume(region, horizon):
    """
    FEATURE 3: Lab Workload Prediction
    Forecasts testing volume to help labs prepare resources
    """
    df_lab = df_clinical[df_clinical['region'] == region].tail(30)
    
    if len(df_lab) < 8:
        return []
    
    # Simple moving average forecast for specimen volume
    recent_avg = df_lab['total_specimens_tested'].tail(4).mean()
    seasonal_factor = df_lab['total_specimens_tested'].tail(8).std() / recent_avg if recent_avg > 0 else 0.1
    
    forecasts = []
    for i in range(horizon):
        # Add slight uncertainty
        pred = int(recent_avg * (1 + seasonal_factor * 0.1 * i))
        forecasts.append(pred)
    
    return forecasts


def get_age_virus_matrix():
    """
    FEATURE 4: Age-Risk Vulnerability Heat Map
    Shows which age groups are most vulnerable to each virus type
    """
    latest_season = df_season['season'].max()
    season_data = df_season[df_season['season'] == latest_season]
    
    matrix = []
    for _, row in season_data.iterrows():
        matrix.append({
            'virus': row['virus'],
            'age_0_4': float(row['cases_age_0_4_pct']),
            'age_5_24': float(row['cases_age_5_24_pct']),
            'age_25_64': float(row['cases_age_25_64_pct']),
            'age_65_plus': float(row['cases_age_65_plus_pct']),
            # Add opacity values for CSS
            'opacity_0_4': float(row['cases_age_0_4_pct']) / 100,
            'opacity_5_24': float(row['cases_age_5_24_pct']) / 100,
            'opacity_25_64': float(row['cases_age_25_64_pct']) / 100,
            'opacity_65_plus': float(row['cases_age_65_plus_pct']) / 100,
            # Pre-computed CSS styles to avoid Jinja syntax errors in editor
            'style_0_4': f"background-color: rgba(239, 68, 68, {float(row['cases_age_0_4_pct']) / 100:.2f})",
            'style_5_24': f"background-color: rgba(59, 130, 246, {float(row['cases_age_5_24_pct']) / 100:.2f})",
            'style_25_64': f"background-color: rgba(16, 185, 129, {float(row['cases_age_25_64_pct']) / 100:.2f})",
            'style_65_plus': f"background-color: rgba(139, 92, 246, {float(row['cases_age_65_plus_pct']) / 100:.2f})"
        })
    
    return matrix


def get_historical_comparison(region):
    """
    FEATURE 5: Pre-2015 vs Post-2015 Trend Analysis
    Shows how flu patterns have evolved over decades
    """
    # Pre-2015 data (aggregate 2010-2014)
    pre = df_pre2015[(df_pre2015['region'] == region) & 
                     (df_pre2015['year'] >= 2010) & 
                     (df_pre2015['year'] < 2015)]
    
    # Post-2015 data (aggregate 2020-2024)
    post_clinical = df_clinical[(df_clinical['region'] == region) & 
                                (df_clinical['year'] >= 2020) & 
                                (df_clinical['year'] < 2025)]
    
    if pre.empty or post_clinical.empty:
        return None
    
    comparison = {
        'pre2015_avg_positivity': float(pre['percent_positive_overall'].mean()),
        'post2015_avg_positivity': float(post_clinical['percent_positive_overall'].mean()),
        'pre2015_peak': float(pre['percent_positive_overall'].max()),
        'post2015_peak': float(post_clinical['percent_positive_overall'].max()),
        'evolution_pct': float((post_clinical['percent_positive_overall'].mean() - pre['percent_positive_overall'].mean()) / pre['percent_positive_overall'].mean() * 100) if pre['percent_positive_overall'].mean() > 0 else 0
    }
    
    return comparison


# ===============================
# ROUTE
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    regions = sorted(df_main["region"].unique())

    forecast = history = None
    severity = dominant_virus = None
    region_selected = None

    lab_trend = None
    subtype_data = None
    age_risks = None
    explanation = None
    virus_evolution = None
    lab_efficiency = None
    specimen_forecast = None
    age_virus_matrix = None
    historical_comp = None

    if request.method == "POST":
        region_selected = request.form["region"]
        horizon = int(request.form["horizon"])

        # ---------------------------
        # Forecast
        # ---------------------------
        df_r = df_main[df_main["region"] == region_selected]
        forecast, history = forecast_future(df_r, horizon)

        latest = forecast[0]
        if latest < 2:
            severity = "Low"
        elif latest < 5:
            severity = "Medium"
        else:
            severity = "High"

        # ---------------------------
        # AI Explanation (Rule-Based)
        # ---------------------------
        # Pass observed history (pre-forecast) so explanations compare forecasts to real past data
        observed_history = df_r[TARGET].tolist()
        # Keep recent window for display/analysis
        observed_history_recent = observed_history[-20:]

        explanation = generate_ai_explanation(
            forecast, severity, horizon, region_selected, observed_history_recent
        )

        # ---------------------------
        # Clinical lab positivity trend
        # ---------------------------
        df_lab = df_clinical[df_clinical["region"] == region_selected]
        lab_trend = df_lab.tail(12)["percent_positive_overall"].tolist()

        # ---------------------------
        # Virus subtype distribution
        # ---------------------------
        df_pub = df_public[df_public["region"] == region_selected]
        subtype_data = {
            "A/H1N1": int(df_pub["cases_a_h1n1_pdm09"].sum()),
            "A/H3": int(df_pub["cases_a_h3"].sum()),
            "B/Victoria": int(df_pub["cases_b_victoria"].sum()),
            "B/Yamagata": int(df_pub["cases_b_yamagata"].sum()),
        }

        # ---------------------------
        # Age risk explainability
        # ---------------------------
        latest_season = df_season["season"].max()
        row = df_season[df_season["season"] == latest_season] \
              .sort_values("total_cases", ascending=False).iloc[0]

        age_risks = {
            "0–4": float(row["cases_age_0_4_pct"]),
            "5–24": float(row["cases_age_5_24_pct"]),
            "25–64": float(row["cases_age_25_64_pct"]),
            "65+": float(row["cases_age_65_plus_pct"]),
        }

        # ---------------------------
        # Gender distribution removed (estimation not shown)
        # Previously the app estimated a gender breakdown from total patients; this is no longer provided.

        # ---------------------------
        # 🔥 NEW UNIQUE FEATURES
        # ---------------------------
        virus_evolution = get_virus_evolution_timeline(region_selected)
        lab_efficiency = get_lab_testing_efficiency(region_selected)
        specimen_forecast = forecast_specimen_volume(region_selected, horizon)
        age_virus_matrix = get_age_virus_matrix()
        historical_comp = get_historical_comparison(region_selected)

    return render_template(
        "index.html",
        regions=regions,
        region_selected=region_selected,
        forecast=forecast,
        history=history,
        severity=severity,
        lab_trend=lab_trend,
        subtype_data=subtype_data,
        age_risks=age_risks,
        explanation=explanation,
        region_map=REGION_MAP,
        virus_evolution=virus_evolution,
        lab_efficiency=lab_efficiency,
        specimen_forecast=specimen_forecast,
        age_virus_matrix=age_virus_matrix,
        historical_comp=historical_comp,
    )


@app.route("/insights", methods=["GET", "POST"])
def insights():
    seasons = sorted(df_season["season"].unique(), reverse=True)

    selected_season = seasons[0] if seasons else None

    dominant_virus = None
    high_risk_group = None
    risk_message = None
    age_risks = {}
    virus_totals = []

    # -----------------------------
    # Handle form submission
    # -----------------------------
    if request.method == "POST":
        selected_season = request.form.get("season")

    # -----------------------------
    # Core Insights Logic (FIXED)
    # -----------------------------
    if selected_season:
        season_df = df_season[df_season["season"] == selected_season]

        if not season_df.empty:
            # 🔹 Virus totals (for bar chart)
            virus_totals_df = (
                season_df
                .groupby("virus")["total_cases"]
                .sum()
                .reset_index()
            )

            virus_totals = virus_totals_df.to_dict("records")

            # 🔹 Dominant virus
            dominant_row = virus_totals_df.sort_values(
                "total_cases", ascending=False
            ).iloc[0]

            dominant_virus = dominant_row["virus"]

            # 🔹 Age risk (mean burden across season)
            age_risks = {
                "0–4": float(season_df["cases_age_0_4_pct"].mean()),
                "5–24": float(season_df["cases_age_5_24_pct"].mean()),
                "25–64": float(season_df["cases_age_25_64_pct"].mean()),
                "65+": float(season_df["cases_age_65_plus_pct"].mean()),
            }

            high_risk_group = max(age_risks, key=age_risks.get)

            # 🔹 Gender Distribution removed (estimation not shown)
            # The insights page previously reported an estimated gender breakdown; removed to avoid showing non-validated estimates.

            # 🔹 Interpretable AI-style explanation
            risk_message = (
                f"{dominant_virus} shows highest circulation during {selected_season}. "
                f"Age group {high_risk_group} experiences the greatest relative burden."
            )

            # 🔹 Pediatric risk factors (from mortality data)
            pediatric_season = df_pediatric[df_pediatric['SEASON'] == selected_season]
            
            risk_conditions = []
            coinfections = []
            
            if not pediatric_season.empty:
                # Top conditions
                conditions_df = pediatric_season[pediatric_season['CHARACTERISTIC'] == 'Conditions']
                if not conditions_df.empty:
                    top_conditions = conditions_df.nlargest(3, 'PERCENT')
                    risk_conditions = top_conditions[['GROUP', 'PERCENT']].to_dict('records')
                
                # Top bacterial co-infections
                coinfection_df = pediatric_season[pediatric_season['CHARACTERISTIC'] == 'Bacterial Coinfectio']
                if not coinfection_df.empty:
                    top_coinfections = coinfection_df.nlargest(3, 'PERCENT')
                    coinfections = top_coinfections[['GROUP', 'PERCENT']].to_dict('records')

    return render_template(
        "insights.html",
        seasons=seasons,
        selected_season=selected_season,
        dominant_virus=dominant_virus,
        high_risk_group=high_risk_group,
        age_risks=age_risks,
        virus_totals=virus_totals,
        risk_message=risk_message,
        risk_conditions=risk_conditions,
        coinfections=coinfections,
    )


@app.route("/reports", methods=["GET", "POST"])
def reports():
    regions = sorted(df_main["region"].unique())
    seasons = sorted(df_season["season"].unique(), reverse=True)

    region_selected = request.form.get("region") if request.method == "POST" else request.args.get("region")
    season_param = request.form.get("season") if request.method == "POST" else request.args.get("season")

    if not region_selected:
        region_selected = regions[0] if regions else None
    if not season_param:
        season_param = seasons[0] if seasons else None
    selected_season = season_param

    # Defaults
    virus_totals = []
    age_risks = {}
    dominant_virus = None
    high_risk_group = None
    lab_eff = None
    forecast_blurb = None

    if selected_season:
        season_df = df_season[df_season["season"] == selected_season]
        if not season_df.empty:
            virus_totals_df = (
                season_df
                .groupby("virus")["total_cases"]
                .sum()
                .reset_index()
            )
            virus_totals = virus_totals_df.to_dict("records")

            dominant_row = virus_totals_df.sort_values("total_cases", ascending=False).iloc[0]
            dominant_virus = dominant_row["virus"]

            age_risks = {
                "0–4": float(season_df["cases_age_0_4_pct"].mean()),
                "5–24": float(season_df["cases_age_5_24_pct"].mean()),
                "25–64": float(season_df["cases_age_25_64_pct"].mean()),
                "65+": float(season_df["cases_age_65_plus_pct"].mean()),
            }
            high_risk_group = max(age_risks, key=age_risks.get)

    if region_selected:
        lab_eff = get_lab_testing_efficiency(region_selected)
        forecast_blurb = short_forecast_blurb(region_selected, horizon=2)
        forecast_series = forecast_series_for_report(region_selected, horizon=3, history_window=8)
    else:
        forecast_series = {"recent": [], "forecast": []}

    return render_template(
        "reports.html",
        regions=regions,
        seasons=seasons,
        region_selected=region_selected,
        selected_season=selected_season,
        virus_totals=virus_totals,
        age_risks=age_risks,
        dominant_virus=dominant_virus,
        high_risk_group=high_risk_group,
        lab_eff=lab_eff,
        forecast_blurb=forecast_blurb,
        forecast_series=forecast_series,
    )

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_msg = data.get('message', '')
    # user-provided inputs
    user_msg = data.get('message', '')
    user_asked_region = data.get('user_region') or data.get('requested_region')
    dashboard_region = data.get('dashboard_region') or data.get('region')
    severity = data.get('severity')
    dominant_virus = data.get('dominant_virus')
    age_risks = data.get('age_risks')
    ili = data.get('ili_forecast_pct') or data.get('ili') or data.get('forecast_ili')
    trend = data.get('trend')

    # Determine which region to use (region override logic)
    # If the user explicitly asked about a different region, prefer that.
    target_region = None
    if user_asked_region and dashboard_region and user_asked_region != dashboard_region:
        target_region = user_asked_region
    else:
        target_region = dashboard_region or user_asked_region

    # Build a fresh snapshot for the target region (fetch region-specific values)
    snapshot = build_region_snapshot(target_region, horizon=int(data.get('horizon', 1)))

    # If no local data exists, region_aware_response will emit the polite missing-data phrasing
    reply = region_aware_response(
        user_msg,
        snapshot,
        history=list(chat_history)
    )

    chat_history.append({"user": user_msg, "assistant": reply})

    return {"response": reply}


if __name__ == "__main__":
    app.run(debug=True)
