"""
build_index.py  —  Build (or rebuild) the FAISS knowledge-base index.

What gets indexed
─────────────────
1. influenza_rag_knowledge.json     — existing expert narrative chunks
2. Per-region annual summaries      — computed from the real CSV datasets
3. Seasonal strain summaries        — from Virus_season.csv
4. Season-over-season comparisons   — for every region × year pair

Run this script directly whenever the CSV data or knowledge JSON changes:
    python utils/build_index.py
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "cleaned_datasets"
RAG_DIR  = BASE_DIR / "rag_model" / "rag_model_data"

KNOWLEDGE_PATH = RAG_DIR / "influenza_rag_knowledge.json"
INDEX_PATH     = RAG_DIR / "influenza.index"
MERGED_PATH    = RAG_DIR / "merged_knowledge.json"

# ──────────────────────────────────────────────────────────────
# LOADERS
# ──────────────────────────────────────────────────────────────
def _load_json_knowledge():
    try:
        with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
        return [{"content": d.get("content", ""), "source": "knowledge_json"} for d in docs if d.get("content")]
    except Exception as e:
        print(f"  [warn] Could not load knowledge JSON: {e}")
        return []


def _flu_season(year, week):
    return f"{year}-{str(year+1)[-2:]}" if week >= 40 else f"{year-1}-{str(year)[-2:]}"


# ──────────────────────────────────────────────────────────────
# CHUNK GENERATORS
# ──────────────────────────────────────────────────────────────
def _regional_annual_chunks(df):
    """One text chunk per region × year — summary of ILI + positivity."""
    chunks = []
    for (region, year), grp in df.groupby(["region", "year"]):
        avg_ili  = round(float(grp["ili_weighted_pct"].mean()), 2)
        peak_ili = round(float(grp["ili_weighted_pct"].max()), 2)
        peak_wk  = int(grp.loc[grp["ili_weighted_pct"].idxmax(), "week"])
        total_a  = int(grp["total_influenza_a_cases"].sum())
        total_b  = int(grp["total_influenza_b_cases"].sum())
        pos_avg  = round(float(grp["percent_positive_overall"].mean()), 1)
        dom      = "Influenza A" if total_a >= total_b else "Influenza B"
        season   = _flu_season(int(year), 40)

        text = (
            f"In {season} (calendar year {year}), {region} reported an average ILI of {avg_ili}% "
            f"(peak {peak_ili}% in week {peak_wk}). "
            f"Influenza A accounted for {total_a} cases versus {total_b} Influenza B cases — "
            f"dominant type: {dom}. "
            f"Average lab positivity was {pos_avg}%."
        )
        chunks.append({"content": text, "source": f"csv_annual_{region}_{year}"})
    return chunks


def _strain_season_chunks(df_pub):
    """One text chunk per region × year with subtype breakdown from PHL data."""
    strain_cols = [
        ("cases_a_h1n1_pdm09",  "A/H1N1pdm09"),
        ("cases_a_h3",          "A/H3N2"),
        ("cases_b_victoria",    "B/Victoria"),
        ("cases_b_yamagata",    "B/Yamagata"),
        ("cases_b_unspecified", "B (unspecified)"),
        ("cases_a_not_subtyped","A (not subtyped)"),
    ]
    chunks = []
    for (region, year), grp in df_pub.groupby(["region", "year"]):
        totals = {}
        for col, label in strain_cols:
            if col in grp.columns:
                v = int(grp[col].sum())
                if v > 0:
                    totals[label] = v
        if not totals:
            continue
        total_all = sum(totals.values())
        if total_all == 0:
            continue
        dominant  = max(totals, key=totals.get)
        dom_pct   = round(totals[dominant] / total_all * 100, 1)
        season    = _flu_season(int(year), 40)
        detail    = ", ".join(
            f"{lb}: {cnt} ({round(cnt/total_all*100,1)}%)"
            for lb, cnt in sorted(totals.items(), key=lambda x: x[1], reverse=True)
        )
        text = (
            f"Strain breakdown for {region} in {year} ({season} season): "
            f"dominant strain was {dominant} ({dom_pct}% of {total_all} typed cases). "
            f"Full breakdown — {detail}."
        )
        chunks.append({"content": text, "source": f"csv_strain_{region}_{year}"})
    return chunks


def _season_comparison_chunks(df):
    """One text chunk per region comparing two consecutive seasons."""
    chunks = []
    regions = df["region"].unique()
    years   = sorted(df["year"].unique())
    for region in regions:
        df_r = df[df["region"] == region]
        for i in range(1, len(years)):
            yr_curr = years[i]
            yr_prev = years[i - 1]
            curr = df_r[df_r["year"] == yr_curr]["ili_weighted_pct"]
            prev = df_r[df_r["year"] == yr_prev]["ili_weighted_pct"]
            if curr.empty or prev.empty:
                continue
            avg_c = round(float(curr.mean()), 2)
            avg_p = round(float(prev.mean()), 2)
            if avg_p == 0:
                continue
            pct_chg = round((avg_c - avg_p) / avg_p * 100, 1)
            direction = "higher" if pct_chg > 5 else "lower" if pct_chg < -5 else "similar"
            text = (
                f"Comparing {region}: {yr_curr} season averaged {avg_c}% ILI versus {avg_p}% in {yr_prev} "
                f"— activity was {direction} ({pct_chg:+.1f}%)."
            )
            chunks.append({"content": text, "source": f"csv_compare_{region}_{yr_curr}_vs_{yr_prev}"})
    return chunks


def _seasonal_age_chunks(df_season):
    """One text chunk per season with age-group burden."""
    chunks = []
    for season, grp in df_season.groupby("season"):
        agg = grp[["cases_age_0_4","cases_age_5_24","cases_age_25_64","cases_age_65_plus","total_cases"]].sum()
        total = int(agg["total_cases"]) or 1
        age_lines = (
            f"0-4 yrs: {round(agg['cases_age_0_4']/total*100,1)}%, "
            f"5-24 yrs: {round(agg['cases_age_5_24']/total*100,1)}%, "
            f"25-64 yrs: {round(agg['cases_age_25_64']/total*100,1)}%, "
            f"65+ yrs: {round(agg['cases_age_65_plus']/total*100,1)}%"
        )
        dominant_rows = grp.sort_values("total_cases", ascending=False)
        top_virus = dominant_rows["virus"].iloc[0] if not dominant_rows.empty else "Unknown"
        text = (
            f"In the {season} flu season, {total} total cases were distributed by age: {age_lines}. "
            f"Dominant virus type: {top_virus}."
        )
        chunks.append({"content": text, "source": f"csv_age_{season}"})
    return chunks


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def build_index():
    print("Loading datasets...")
    try:
        df_main   = pd.read_csv(DATA_DIR / "influenza_modeling_dataset_2015_present.csv")
        df_pub    = pd.read_csv(DATA_DIR / "public_health_lab_cleaned_dataset.csv")
        df_season = pd.read_csv(DATA_DIR / "Virus_season.csv")
    except Exception as e:
        print(f"ERROR loading CSVs: {e}")
        sys.exit(1)

    print("Generating knowledge chunks...")
    all_chunks = []

    # 1. Expert narrative from JSON
    json_chunks = _load_json_knowledge()
    all_chunks.extend(json_chunks)
    print(f"  {len(json_chunks):>5} chunks from knowledge JSON")

    # 2. Regional annual summaries
    reg_chunks = _regional_annual_chunks(df_main)
    all_chunks.extend(reg_chunks)
    print(f"  {len(reg_chunks):>5} regional-annual chunks")

    # 3. Strain subtype chunks
    strain_chunks = _strain_season_chunks(df_pub)
    all_chunks.extend(strain_chunks)
    print(f"  {len(strain_chunks):>5} strain-breakdown chunks")

    # 4. Season-over-season comparison chunks
    compare_chunks = _season_comparison_chunks(df_main)
    all_chunks.extend(compare_chunks)
    print(f"  {len(compare_chunks):>5} season-comparison chunks")

    # 5. Seasonal age burden chunks
    age_chunks = _seasonal_age_chunks(df_season)
    all_chunks.extend(age_chunks)
    print(f"  {len(age_chunks):>5} age-burden chunks")

    print(f"\n  Total chunks to index: {len(all_chunks)}")

    # Save merged knowledge for inspection
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    with open(MERGED_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"  Merged knowledge saved → {MERGED_PATH}")

    # Encode + build FAISS
    print("\nEncoding chunks with sentence-transformers...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["content"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64).astype(np.float32)

    dim = embeddings.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(embeddings)

    faiss.write_index(idx, str(INDEX_PATH))
    print(f"\n✅ FAISS index built: {len(all_chunks)} vectors, dim={dim}")
    print(f"   Index saved → {INDEX_PATH}")
    return all_chunks


if __name__ == "__main__":
    build_index()

