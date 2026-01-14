"""
Streamlit dashboard — Schema-aware, interactive analytics for Great Britain power market
Adapted to the provided schema and extended to:
- Normalize Hourly_Generation_Prices (remove duplicates, respect installed capacity)
- Visualize generation mix across time and year
- Detect irregularities where hourly generation > installed capacity
- Identify and resolve duplicate hourly records according to defined rules
- Compute utilization (capacity factor) by segment
- Peak demand analysis and linear regression forecast (2030-2040)
- Interactive hourly profile scaling using peak forecasts
- Dispatch order visualization using capture metrics (Annual_Price_Forecast)
- Monte Carlo simulation for hourly price forecasts (based on historical hourly prices)
- Narrative text and short descriptions for each visual

Usage:
    pip install -r requirements.txt
    streamlit run app.py

Notes / assumptions:
- Hourly generation data is in MWh. Installed capacity is expected in GW columns (2021..2050).
  We convert installed capacity GW -> MW by * 1000, and compare hourly MWh against capacity MW (1 hour).
- For duplicate rows (same Region + Year+Month+Day+Hour), selection rules:
    1) Prefer rows with zero exceedances (all generation types <= capacity for that year).
    2) If multiple with zero exceedances, choose the row with the largest total generation (most complete).
    3) If all rows exceed capacity for some types, choose the row minimizing the total exceedance (closest to capacities).
- If Market_Installed_Capacity does not contain a match for a Region+Type+Year, we mark capacity as Na and treat comparisons as unknown.
- Capture metrics in Annual_Price_Forecast are used as a proxy for dispatch priority; we explain the assumption in the UI.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(layout="wide", page_title="GB Power Markets — Interactive Dashboard", initial_sidebar_state="expanded")

BUNDLED_FILENAME = "power_sample_data.xlsx"

# -------------------------
# Helper / loading functions
# -------------------------
@st.cache_data
def load_excel(io_bytes) -> Dict[str, pd.DataFrame]:
    try:
        sheets = pd.read_excel(io_bytes, sheet_name=None, engine="openpyxl")
    except Exception:
        sheets = pd.read_excel(io_bytes, sheet_name=None)
    # normalize column names
    for k, df in sheets.items():
        df.columns = [str(c).strip() for c in df.columns]
        sheets[k] = df
    return sheets


def safe_col(df: pd.DataFrame, contains: str) -> Optional[str]:
    """Return column name that contains the substring (case-insensitive), or None."""
    contains_l = contains.lower()
    for c in df.columns:
        if contains_l in c.lower():
            return c
    return None


def build_timestamp_from_hmdh(df: pd.DataFrame, year_col="Year", month_col="Month", day_col="Day", hour_col="Hour") -> pd.Series:
    # find columns tolerant to case/spaces
    cols = {}
    for name in (year_col, month_col, day_col, hour_col):
        if name in df.columns:
            cols[name] = name
        else:
            found = safe_col(df, name)
            cols[name] = found
    if any(v is None for v in cols.values()):
        missing = [k for k, v in cols.items() if v is None]
        raise KeyError(f"Missing datetime columns: {missing}")
    tmp = df.copy()
    for k, col in cols.items():
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce").fillna(0).astype(int)
    tmp[cols[hour_col]] = tmp[cols[hour_col]] % 24
    ts = pd.to_datetime(dict(year=tmp[cols[year_col]], month=tmp[cols[month_col]], day=tmp[cols[day_col]], hour=tmp[cols[hour_col]]), errors="coerce")
    return ts


def detect_generation_columns(hgp: pd.DataFrame) -> List[str]:
    expected = [
        "System Load (MWh)",
        "Solar (MWh)",
        "Wind Onshore (MWh)",
        "Wind Offshore (MWh)",
        "Energy Storage (MWh)",
        "Coal (MWh)",
        "Gas (MWh)",
        "Hydro (MWh)",
        "Nuclear (MWh)",
        "Demand Response (MWh)",
        "Other (MWh)",
        "Net imports (exports) (MWh)",
    ]
    present = [c for c in expected if c in hgp.columns]
    # try fuzzy matches if not present
    if not present:
        tokens = ["solar", "onshore", "offshore", "storage", "coal", "gas", "hydro", "nuclear", "demand", "other", "imports", "system load"]
        for t in tokens:
            c = safe_col(hgp, t)
            if c and c not in present:
                present.append(c)
    # ensure System Load first if present
    present_sorted = sorted(present, key=lambda x: 0 if "system" in x.lower() else 1)
    return present_sorted


def map_gencol_to_type_key(gen_col: str) -> str:
    # map generation column name to a type token that likely appears in Market_Installed_Capacity.Type
    name = gen_col.lower()
    if "solar" in name:
        return "solar"
    if "onshore" in name or ("wind" in name and "onshore" in name):
        return "onshore"
    if "offshore" in name or ("wind" in name and "offshore" in name):
        return "offshore"
    if "nuclear" in name:
        return "nuclear"
    if "coal" in name:
        return "coal"
    if "gas" in name:
        return "gas"
    if "hydro" in name:
        return "hydro"
    if "storage" in name or "energy storage" in name:
        return "storage"
    if "demand" in name:
        return "demand response"
    if "import" in name or "net" in name:
        return "imports"
    return gen_col.lower()


def extract_year_cols(mic: pd.DataFrame) -> List[str]:
    # detect year columns from 2021 to 2050
    year_cols = [c for c in mic.columns if str(c).strip().isdigit() and 2021 <= int(str(c).strip()) <= 2050]
    if not year_cols:
        # fallback: any column that contains 2021..2050 substring
        for c in mic.columns:
            for y in range(2021, 2051):
                if str(y) in str(c):
                    if c not in year_cols:
                        year_cols.append(c)
    return year_cols


def capacity_lookup(mic: pd.DataFrame, region: Optional[str], type_key: str, year: int, region_col: Optional[str], type_col: Optional[str], year_cols: List[str]) -> Optional[float]:
    """
    Return installed capacity in MW (converted from GW*1000) for given region+type+year.
    If not found, return np.nan
    """
    if mic is None or year_cols is None or not year_cols:
        return np.nan
    # map year column name
    year_col_name = None
    for yc in year_cols:
        if str(year) in str(yc):
            year_col_name = yc
            break
    if year_col_name is None:
        # fallback to first year column
        year_col_name = year_cols[0]
    # filter by region if available
    df = mic.copy()
    if region_col and region in df.columns and pd.notna(region):
        df = df[df[region_col] == region]
    # filter by type; search for type_key substring in type_col
    if type_col and type_col in df.columns:
        mask = df[type_col].astype(str).str.lower().str.contains(type_key.lower(), na=False)
        df = df[mask]
    # if no matches, try more fuzzy matching across type_col
    if df.empty and type_col and type_col in mic.columns:
        # try partial tokens split
        token = type_key.split()[0]
        mask = mic[type_col].astype(str).str.lower().str.contains(token, na=False)
        df = mic[mask]
        if region_col and region in df.columns and pd.notna(region):
            df = df[df[region_col] == region]
    if df.empty:
        return np.nan
    # take first match
    val = df[year_col_name].iloc[0]
    try:
        val = float(val)
    except Exception:
        return np.nan
    # assume val is GW -> convert to MW by * 1000
    return val * 1000.0


# -------------------------
# Sidebar: load file
# -------------------------
st.sidebar.title("Data Source")
uploaded = st.sidebar.file_uploader("Upload power_sample_data.xlsx (.xlsx) with the schema", type=["xlsx", "xls"])
use_bundled = False
if uploaded is None:
    if Path(BUNDLED_FILENAME).exists():
        st.sidebar.success(f"Using bundled file: {BUNDLED_FILENAME}")
        use_bundled = True
    else:
        st.sidebar.info("No bundled file found. Upload one to get started.")

sheets = {}
if uploaded is not None:
    try:
        sheets = load_excel(uploaded)
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded file: {e}")
elif use_bundled:
    try:
        sheets = load_excel(BUNDLED_FILENAME)
    except Exception as e:
        st.sidebar.error(f"Failed to load bundled file: {e}")

if not sheets:
    st.title("GB Power Markets — Data Explorer")
    st.warning("No dataset loaded. Upload 'power_sample_data.xlsx' or place it in this folder.")
    st.stop()

st.title("GB Power Markets — Interactive Dashboard")
st.markdown("Interactive analysis of generation, capacity, utilization and price forecasts. "
            "Narratives next to charts explain assumptions and insights. Use the controls on the left to filter and tune analyses.")

# -------------------------
# Prepare primary sheets
# -------------------------
hgp = sheets.get("Hourly_Generation_Prices")
mic = sheets.get("Market_Installed_Capacity")
apf = sheets.get("Annual_Price_Forecast")
mg = sheets.get("Market_Generation")

# -------------------------
# Process Hourly_Generation_Prices
# -------------------------
if hgp is not None:
    st.header("Hourly Generation (schema-aware)")
    st.subheader("Raw preview (top 200 rows)")
    st.dataframe(hgp.head(200))

    # build timestamp
    try:
        hgp["timestamp"] = build_timestamp_from_hmdh(hgp)
    except Exception as e:
        st.error(f"Could not build timestamp from Year/Month/Day/Hour columns: {e}")
        # attempt fallback if Date column exists
        date_col = safe_col(hgp, "date")
        if date_col:
            hgp["timestamp"] = pd.to_datetime(hgp[date_col], errors="coerce")
        else:
            hgp["timestamp"] = pd.NaT

    # detect key columns
    region_col = safe_col(hgp, "region") or ("Region" if "Region" in hgp.columns else None)
    gen_cols = detect_generation_columns(hgp)  # includes System Load first if present
    price_col = safe_col(hgp, "price")
    st.write("Detected region column:", region_col)
    st.write("Detected generation columns:", gen_cols)
    st.write("Detected price column:", price_col)

    # Prepare Market Installed Capacity lookup metadata
    if mic is not None:
        mic_cols = mic.columns.tolist()
        mic_region_col = safe_col(mic, "region") or (mic_cols[0] if len(mic_cols) > 0 else None)
        mic_type_col = safe_col(mic, "type") or safe_col(mic, "technology") or safe_col(mic, "category")
        mic_year_cols = extract_year_cols(mic)
        st.sidebar.write("Installed capacity source:", "Market_Installed_Capacity sheet found")
    else:
        mic_region_col = mic_type_col = None
        mic_year_cols = []
        st.sidebar.warning("Market_Installed_Capacity sheet not found — capacity checks will be limited.")

    st.markdown("### Duplicate handling and capacity-check logic")
    st.write(
        "We detect duplicate records by (Region, Year, Month, Day, Hour). "
        "Selection rules prioritize rows that respect installed capacity; if all exceed capacities we pick the row closest to capacities."
    )

    # identify duplicate sets
    key_cols = []
    for c in ["Region", "Year", "Month", "Day", "Hour"]:
        if c in hgp.columns:
            key_cols.append(c)
    if region_col and "Region" not in key_cols:
        # ensure region is included
        if "Region" in hgp.columns:
            if "Region" not in key_cols:
                key_cols = ["Region"] + [k for k in key_cols]

    # fallback: use timestamp + region
    if not key_cols:
        if "timestamp" in hgp.columns and region_col:
            key_cols = [region_col, "timestamp"]
        elif "timestamp" in hgp.columns:
            key_cols = ["timestamp"]
        else:
            st.error("Unable to build duplicate detection key (missing time columns).")
            key_cols = []

    # function to compute capacity vector for a row (per generation column)
    def row_capacities_for_row(row) -> Dict[str, float]:
        caps = {}
        if mic is None or not mic_year_cols:
            # can't lookup capacities -> return NaN set
            for g in gen_cols:
                caps[g] = np.nan
            return caps
        # determine region and year
        region_val = row.get(region_col) if region_col in row.index else None
        year_val = int(row.get("Year")) if "Year" in row.index and not pd.isna(row.get("Year")) else (row.get("timestamp").year if pd.notna(row.get("timestamp")) else None)
        for g in gen_cols:
            # map generation column to type token
            tk = map_gencol_to_type_key(g)
            cap_mw = capacity_lookup(mic, region_val, tk, int(year_val) if year_val else 2021, mic_region_col, mic_type_col, mic_year_cols)
            caps[g] = cap_mw
        return caps

    # build capacities cache for unique (region, year) combos to avoid repeated lookups
    caps_cache: Dict[Tuple[Optional[str], Optional[int]], Dict[str, float]] = {}

    # Find duplicates groups
    if key_cols:
        dup_groups = hgp.duplicated(subset=key_cols, keep=False)
        dup_df = hgp[dup_groups].copy()
        n_dup_groups = dup_df.groupby(key_cols).ngroups if not dup_df.empty else 0
    else:
        dup_df = pd.DataFrame()
        n_dup_groups = 0

    st.write(f"Duplicate groups detected: {n_dup_groups}")

    # Selection algorithm for duplicates
    def select_best_row_for_group(group: pd.DataFrame) -> pd.Series:
        # for each row compute exceedance sum (sum of positive gen - cap across gen_cols)
        scores = []
        for idx, row in group.iterrows():
            region_val = row.get(region_col) if region_col else None
            year_val = int(row.get("Year")) if "Year" in row and not pd.isna(row.get("Year")) else (row.get("timestamp").year if pd.notna(row.get("timestamp")) else None)
            cache_key = (region_val, int(year_val) if year_val else None)
            if cache_key in caps_cache:
                caps = caps_cache[cache_key]
            else:
                caps = row_capacities_for_row(row)
                caps_cache[cache_key] = caps
            # compute per column exceedance (MWh - capacity_MW) for each generation type
            exceedances = {}
            total_exceed = 0.0
            total_gen = 0.0
            n_cols_used = 0
            for g in gen_cols:
                gen_val = pd.to_numeric(row.get(g), errors="coerce")
                if pd.isna(gen_val):
                    gen_val = 0.0
                cap = caps.get(g, np.nan)
                if pd.isna(cap):
                    # missing capacity: treat as no-exceed (neutral) but mark unknown
                    exceed = 0.0
                else:
                    exceed = max(0.0, gen_val - cap)  # MWh vs MW (capacity in MW equates to MWh potential for 1 hour)
                total_exceed += exceed
                total_gen += gen_val
                n_cols_used += 1
                exceedances[g] = exceed
            scores.append({
                "index": idx,
                "total_exceed": total_exceed,
                "total_gen": total_gen,
                "exceedances": exceedances,
                "n_cols": n_cols_used
            })
        # prefer total_exceed == 0 (no exceedances); among them choose max total_gen
        zero_exceed = [s for s in scores if s["total_exceed"] == 0]
        if zero_exceed:
            chosen = max(zero_exceed, key=lambda x: x["total_gen"])
        else:
            # choose the one minimizing total_exceed, tie-breaker: highest total_gen
            chosen = min(scores, key=lambda x: (x["total_exceed"], -x["total_gen"]))
        return group.loc[chosen["index"]]

    # Build normalized (deduplicated) dataframe
    if not dup_df.empty:
        # iterate groups and pick best
        groups = []
        for _, grp in dup_df.groupby(key_cols):
            best = select_best_row_for_group(grp)
            groups.append(best)
        # non-duplicate rows
        non_dup = hgp[~dup_groups].copy()
        selected = pd.DataFrame(groups)
        normalized = pd.concat([non_dup, selected], ignore_index=True).sort_values(by="timestamp")
    else:
        normalized = hgp.copy()

    st.subheader("Normalized hourly data (duplicates resolved)")
    st.write("Preview of normalized data:")
    st.dataframe(normalized.head(200))

    # After selection, clip per-generation values to installed capacity where capacity available
    st.markdown("### Enforce capacity bounds on normalized data")
    enforce = st.checkbox("Clip generation values to installed capacity where capacity exists (cap to capacity) — recommended", value=True)
    normalized_clipped = normalized.copy()

    # Vectorized clipping approach using a capacities table per (Region, Year)
    if mic is not None and mic_year_cols:
        # build keys for each row in normalized_clipped: (region_val, year_val)
        def _row_key(row):
            region_val = row.get(region_col) if region_col else None
            year_val = None
            if "Year" in row.index and not pd.isna(row.get("Year")):
                try:
                    year_val = int(row.get("Year"))
                except Exception:
                    year_val = None
            if year_val is None and pd.notna(row.get("timestamp")):
                try:
                    year_val = int(row.get("timestamp").year)
                except Exception:
                    year_val = None
            return (region_val, year_val)

        key_series = normalized_clipped.apply(_row_key, axis=1)
        unique_keys = list({k for k in key_series.tolist()})

        # populate caps_cache for unique keys (if not already)
        for region_val, year_val in unique_keys:
            cache_key = (region_val, int(year_val) if year_val else None)
            if cache_key in caps_cache:
                continue
            caps = {}
            for g in gen_cols:
                tk = map_gencol_to_type_key(g)
                cap_mw = capacity_lookup(mic, region_val, tk, int(year_val) if year_val else 2021, mic_region_col, mic_type_col, mic_year_cols)
                caps[g] = cap_mw
            caps_cache[cache_key] = caps

        # build capacities DataFrame to merge
        caps_rows = []
        for (region_val, year_val), caps in caps_cache.items():
            row = {"_key_region": region_val, "_key_year": year_val}
            for g in gen_cols:
                row[f"cap__{g}"] = caps.get(g, np.nan)
            caps_rows.append(row)
        caps_df = pd.DataFrame(caps_rows)

        # attach key columns to normalized_clipped for merge
        key_cols_df = normalized_clipped.reset_index().rename(columns={"index": "_orig_index"})
        key_cols_df["_key_region"] = key_series.apply(lambda t: t[0]).values
        key_cols_df["_key_year"] = key_series.apply(lambda t: t[1]).values

        merged = key_cols_df.merge(caps_df, on=["_key_region", "_key_year"], how="left")

        # perform clipping
        if enforce:
            clip_counts_total = 0
            for g in gen_cols:
                gen_vals = pd.to_numeric(merged[g], errors="coerce").fillna(0.0)
                cap_col = f"cap__{g}"
                cap_vals = pd.to_numeric(merged[cap_col], errors="coerce")
                # only clip where capacity is known (not NaN)
                mask_known = cap_vals.notna()
                # count clips
                clipped_mask = mask_known & (gen_vals > cap_vals)
                clip_counts_total += int(clipped_mask.sum())
                # apply clip (where cap known, else keep original)
                merged[g] = np.where(mask_known, np.minimum(gen_vals, cap_vals), gen_vals)
            # restore order and drop helper cols
            # reassign normalized_clipped from merged
            cols_keep = [c for c in normalized_clipped.columns]
            normalized_clipped = merged[cols_keep].set_index("_orig_index").sort_index().reset_index(drop=True)
            st.write("Number of clipped generation values (est.):", clip_counts_total)
        else:
            st.info("Clipping is disabled — normalized dataset left unchanged.")
    else:
        st.info("Market_Installed_Capacity not provided or missing year columns — clipping unavailable.")

    st.subheader("Irregularities: hourly generation > installed capacity")
    # compute irregularities for normalized (or raw) full set to display where gen > capacity
    def compute_exceedances(df_in: pd.DataFrame) -> pd.DataFrame:
        records = []
        for idx, row in df_in.iterrows():
            region_val = row.get(region_col) if region_col else None
            year_val = int(row.get("Year")) if "Year" in row and not pd.isna(row.get("Year")) else (row.get("timestamp").year if pd.notna(row.get("timestamp")) else None)
            cache_key = (region_val, int(year_val) if year_val else None)
            if cache_key in caps_cache:
                caps = caps_cache[cache_key]
            else:
                caps = row_capacities_for_row(row)
                caps_cache[cache_key] = caps
            for g in gen_cols:
                gen_val = pd.to_numeric(row.get(g), errors="coerce")
                if pd.isna(gen_val):
                    continue
                cap = caps.get(g, np.nan)
                if pd.isna(cap):
                    continue
                exceed = gen_val - cap
                if exceed > 0:
                    records.append({
                        "index": idx,
                        "timestamp": row.get("timestamp"),
                        "Region": region_val,
                        "Year": year_val,
                        "GenType": g,
                        "Generation_MWh": gen_val,
                        "Capacity_MW": cap,
                        "Exceed_MWh": exceed
                    })
        return pd.DataFrame(records)

    irregularities = compute_exceedances(normalized if not enforce else normalized_clipped)
    st.write(f"Irregular records found (generation > capacity): {len(irregularities)}")
    if not irregularities.empty:
        st.dataframe(irregularities.head(200))
        # plot aggregated exceedances by gen type
        ex_by_type = irregularities.groupby("GenType")["Exceed_MWh"].sum().reset_index().sort_values("Exceed_MWh", ascending=False)
        fig_ex = px.bar(ex_by_type, x="GenType", y="Exceed_MWh", title="Total hourly exceedances by generation type (MWh)")
        st.plotly_chart(fig_ex, use_container_width=True)
    else:
        st.info("No exceedances detected (after selection/clipping).")

    st.subheader("Duplicates diagnostics")
    if not dup_df.empty:
        st.write("Sample of duplicate groups and selection outcome (first 50):")
        # show group comparisons for a sample of groups
        sample_groups = []
        for _, grp in dup_df.groupby(key_cols):
            grp_display = grp.copy()
            # compute total generation and a simple exceed metric per row for display
            grp_display["_total_gen"] = grp_display[gen_cols].sum(axis=1, numeric_only=True)
            # annotate chosen row index
            best = select_best_row_for_group(grp)
            grp_display["_selected"] = grp_display.index == best.name
            sample_groups.append(grp_display)
            if len(sample_groups) >= 10:
                break
        if sample_groups:
            comp = pd.concat(sample_groups, ignore_index=True)
            st.dataframe(comp.head(200))
    else:
        st.info("No duplicate groups detected.")

    # Export normalized dataset
    st.markdown("### Download normalized hourly dataset")
    export_df = normalized_clipped if enforce else normalized
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download normalized Hourly_Generation_Prices CSV", data=csv, file_name="Hourly_Generation_Prices_normalized.csv", mime="text/csv")

    # -------------------------
    # Time-series visuals: generation mix and year/time filters
    # -------------------------
    st.header("Generation mix across time and by year")
    with st.sidebar.expander("Time / Year selection"):
        min_ts = export_df["timestamp"].min()
        max_ts = export_df["timestamp"].max()
        st.write(f"Data range: {min_ts} — {max_ts}")
        date_range = st.sidebar.date_input("Select date range", value=(min_ts.date() if pd.notna(min_ts) else None, max_ts.date() if pd.notna(max_ts) else None))
        year_filter = st.sidebar.multiselect("Select years to include", options=sorted(export_df["timestamp"].dt.year.dropna().unique().tolist()), default=sorted(export_df["timestamp"].dt.year.dropna().unique().tolist()))
        resample_freq = st.sidebar.selectbox("Resample frequency for visuals", options=["H", "D", "W", "M"], index=1)
        stack_group_by_region = st.sidebar.checkbox("Stack area per region (if multiple regions)", value=False)

    # filter by date range and year
    df_vis = export_df.copy()
    if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df_vis = df_vis[(df_vis["timestamp"] >= start) & (df_vis["timestamp"] <= end)]
    if year_filter:
        df_vis = df_vis[df_vis["timestamp"].dt.year.isin(year_filter)]

    if stack_group_by_region and region_col and region_col in df_vis.columns:
        # aggregate per region then stack by region (sum of all gen types)
        st.write("Stacked area per region (generation mix across selected gen types).")
        agg = df_vis.set_index("timestamp").groupby(region_col)[gen_cols].resample(resample_freq).sum().reset_index()
        # create stacked area per region: show total generation per region (sum of gen cols) or separate by gen type?
        agg["Total_MWh"] = agg[gen_cols].sum(axis=1, numeric_only=True)
        fig = px.area(agg, x="timestamp", y="Total_MWh", color=region_col, line_group=region_col, title=f"Total generation by region ({resample_freq} resampled)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        # stacked area for generation types across whole dataset (or filtered region)
        st.write("Stacked area of generation mix by technology (selected gen types).")
        if region_col and region_col in df_vis.columns:
            # allow region selection for this visual
            sel_region = st.selectbox("Region for generation mix", options=["All"] + sorted(df_vis[region_col].dropna().unique().tolist()))
            if sel_region != "All":
                df_vis_plot = df_vis[df_vis[region_col] == sel_region].copy()
            else:
                df_vis_plot = df_vis.copy()
        else:
            df_vis_plot = df_vis.copy()

        if df_vis_plot.empty:
            st.info("No data selected.")
        else:
            g = df_vis_plot.set_index("timestamp")[gen_cols].resample(resample_freq).sum().fillna(0)
            # stacked area by generation type
            fig = go.Figure()
            for col in gen_cols:
                fig.add_trace(go.Scatter(x=g.index, y=g[col], mode="lines", stackgroup="one", name=col))
            fig.update_layout(title=f"Generation mix ({resample_freq} resampled)", xaxis_title="Date", yaxis_title="Generation (MWh)")
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Utilization by market segment
    # -------------------------
    st.header("Utilization (capacity factor) by market segment")
    # compute capacity for each generation type by region+year and compute utilization per hour = generation / capacity_MW
    if mic is not None and mic_year_cols:
        st.write("Capacity data found — computing utilization (hourly generation / capacity).")
        util_df = []
        for idx, row in export_df.iterrows():
            region_val = row.get(region_col) if region_col else None
            year_val = int(row.get("Year")) if "Year" in row and not pd.isna(row.get("Year")) else (row.get("timestamp").year if pd.notna(row.get("timestamp")) else None)
            cache_key = (region_val, int(year_val) if year_val else None)
            if cache_key in caps_cache:
                caps = caps_cache[cache_key]
            else:
                caps = row_capacities_for_row(row)
                caps_cache[cache_key] = caps
            rec = {"timestamp": row.get("timestamp"), "Region": region_val, "Year": year_val}
            for g in gen_cols:
                gen_val = pd.to_numeric(row.get(g), errors="coerce")
                cap = caps.get(g, np.nan)
                if pd.isna(gen_val) or pd.isna(cap) or cap == 0:
                    util = np.nan
                else:
                    util = gen_val / cap  # since cap in MW, gen_val in MWh for the hour -> fraction
                rec[g + "_util"] = util
            util_df.append(rec)
        util_df = pd.DataFrame(util_df)
        # average utilization per generation type
        util_cols = [c for c in util_df.columns if c.endswith("_util")]
        avg_util = util_df[util_cols].mean(skipna=True).reset_index()
        avg_util.columns = ["GenType_util", "Avg_util"]
        # map names
        avg_util["GenType"] = avg_util["GenType_util"].str.replace("_util", "")
        st.subheader("Average utilization (capacity factor) by technology")
        fig_util = px.bar(avg_util.sort_values("Avg_util", ascending=False), x="GenType", y="Avg_util", labels={"Avg_util": "Average utilization (fraction)", "GenType": "Technology"})
        st.plotly_chart(fig_util, use_container_width=True)
    else:
        st.info("Market_Installed_Capacity sheet missing — utilization cannot be computed reliably.")

    # -------------------------
    # Peak demand & regression forecast
    # -------------------------
    st.header("Peak demand analysis & forecast (annual peaks)")
    if "System Load (MWh)" in gen_cols or safe_col(export_df, "system load"):
        load_col = next(c for c in gen_cols if "system" in c.lower())
        # compute annual peaks
        peak_ann = export_df.dropna(subset=["timestamp"]).copy()
        peak_ann["Year"] = peak_ann["timestamp"].dt.year
        ann_peaks = peak_ann.groupby("Year")[load_col].max().reset_index().sort_values("Year")
        st.write("Annual peak demand (MWh):")
        st.dataframe(ann_peaks)
        # regression (linear) for trend
        X = ann_peaks[["Year"]].values
        y = ann_peaks[load_col].values
        if len(ann_peaks) >= 3:
            lr = LinearRegression()
            lr.fit(X, y)
            y_pred = lr.predict(X)
            r2 = r2_score(y, y_pred)
            st.write(f"Linear trend: slope = {lr.coef_[0]:.2f} MWh/year, intercept = {lr.intercept_:.2f}. R^2 = {r2:.3f}")
            # forecast 2030-2040
            years_future = np.arange(2030, 2041).reshape(-1, 1)
            y_fore = lr.predict(years_future)
            fc = pd.DataFrame({"Year": years_future.flatten(), "Peak_forecast_MWh": y_fore})
            st.subheader("Peak forecast 2030-2040 (linear projection)")
            st.dataframe(fc)
            # plot historical + forecast
            fig_peak = go.Figure()
            fig_peak.add_trace(go.Scatter(x=ann_peaks["Year"], y=ann_peaks[load_col], mode="markers+lines", name="Historical peaks"))
            fig_peak.add_trace(go.Scatter(x=years_future.flatten(), y=y_fore, mode="lines+markers", name="Linear forecast", line=dict(dash="dash")))
            st.plotly_chart(fig_peak, use_container_width=True)
        else:
            st.info("Insufficient years for regression (need >=3 years). Displaying historical peaks only.")
            fig_peak = px.line(ann_peaks, x="Year", y=load_col, title="Annual peak demand (MWh)")
            st.plotly_chart(fig_peak, use_container_width=True)
    else:
        st.info("No System Load column detected for peak demand analysis.")

    # -------------------------
    # Hourly profile scaling using peak forecast
    # -------------------------
    st.header("Interactive hourly demand profile using peak forecasts")
    st.write("We create an average hourly demand profile (fractional distribution across 24 hours), then scale it to forecasted peak demand to show expected hourly profile.")
    # compute average hourly profile from historical data
    if "timestamp" in export_df.columns and (("System Load (MWh)" in gen_cols) or safe_col(export_df, "system load")):
        load_col = next(c for c in gen_cols if "system" in c.lower())
        tmp = export_df.dropna(subset=["timestamp", load_col]).copy()
        tmp["hour"] = tmp["timestamp"].dt.hour
        avg_by_hour = tmp.groupby("hour")[load_col].mean().reset_index()
        avg_by_hour["fraction"] = avg_by_hour[load_col] / avg_by_hour[load_col].sum()
        st.write("Average hourly distribution (fraction of daily total):")
        st.dataframe(avg_by_hour)
        # allow user to pick forecast year and show scaled hourly profile
        if len(ann_peaks) >= 3:
            years_to_show = list(range(2021, 2051))
            sel_year = st.selectbox("Select forecast year for hourly profile (uses linear peak forecast if selected)", options=[2025, 2030, 2035, 2040], index=1)
            # if regression exists, use lr prediction for that year; else use historical peak median
            if 'lr' in locals():
                peak_for_year = lr.predict(np.array([[sel_year]]))[0]
            else:
                peak_for_year = ann_peaks[load_col].median()
            # we interpret peak as maximum hourly System Load. Scale hourly fractions so maximum equals peak_for_year:
            profile = avg_by_hour.copy()
            # scale such that max(profile*scale) == peak_for_year
            scale = peak_for_year / profile[load_col].max()
            profile["Scaled_MWh"] = profile[load_col] * scale
            fig_prof = px.line(profile, x="hour", y="Scaled_MWh", title=f"Estimated hourly demand profile scaled to peak {sel_year} = {peak_for_year:.1f} MWh")
            st.plotly_chart(fig_prof, use_container_width=True)
        else:
            st.info("Insufficient annual data for peak forecasting — using historic median peak for scaling.")
            peak_for_year = tmp[load_col].max()
            profile = avg_by_hour.copy()
            scale = peak_for_year / profile[load_col].max()
            profile["Scaled_MWh"] = profile[load_col] * scale
            fig_prof = px.line(profile, x="hour", y="Scaled_MWh", title=f"Estimated hourly demand profile scaled to recent peak")
            st.plotly_chart(fig_prof, use_container_width=True)
    else:
        st.info("Cannot compute hourly demand profile (missing System Load or timestamp).")

    # -------------------------
    # Dispatch order using capture/prices (Annual_Price_Forecast)
    # -------------------------
    st.header("Dispatch order proxy using capture metrics")
    st.write(
        "We use capture metrics from Annual_Price_Forecast as a proxy for dispatch priority. "
        "Assumption: technologies with higher capture rates capture more of the market price and are therefore less likely to be curtailed. "
        "This is a proxy — dispatch priority in real systems depends on marginal costs and market rules."
    )
    if apf is not None:
        st.subheader("Annual Price Forecast preview")
        st.dataframe(apf.head(200))
        # find capture-like columns
        capture_cols = [c for c in apf.columns if "capture" in str(c).lower() or "capture" in str(c).lower() or "capture" in str(c)]
        # also include columns with generation type names
        extra_candidates = []
        for token in ["wind", "solar", "gas", "offshore", "onshore", "nuclear"]:
            c = safe_col(apf, token)
            if c and c not in capture_cols:
                extra_candidates.append(c)
        capture_cols = list(dict.fromkeys(capture_cols + extra_candidates))
        st.write("Detected capture/metric columns:", capture_cols)
        cap_df = apf.copy()
        # try to coerce to numeric for visualization
        for c in capture_cols:
            cap_df[c + "_num"] = pd.to_numeric(cap_df[c], errors="coerce")
        num_caps = [c + "_num" for c in capture_cols if c + "_num" in cap_df.columns]
        if num_caps:
            # average across categories (if multiple rows)
            avg_caps = cap_df[num_caps].mean(skipna=True).sort_values()
            dc = pd.DataFrame({"Tech": [c.replace("_num", "") for c in avg_caps.index], "Capture": avg_caps.values})
            dc = dc.sort_values("Capture", ascending=False)
            fig_disp = px.bar(dc, x="Tech", y="Capture", title="Proxy dispatch priority (higher capture assumed higher priority)")
            st.plotly_chart(fig_disp, use_container_width=True)
        else:
            st.info("No numeric capture-like columns found in Annual_Price_Forecast (they may be strings).")
    else:
        st.info("Annual_Price_Forecast sheet missing — cannot build dispatch proxy.")

    # -------------------------
    # Monte Carlo simulation for price forecast
    # -------------------------
    st.header("Monte Carlo price forecasts (ensemble) — hourly")
    st.write(
        "We use historical hourly Price (if available) to build a simple hourly seasonal + residual model. "
        "Residuals are bootstrapped (non-parametric) and used to create multiple simulated future ensembles. "
        "This is a lightweight demonstration — for production use consider richer time-series models."
    )
    if price_col and price_col in export_df.columns:
        price_ser = export_df.dropna(subset=["timestamp", price_col]).set_index("timestamp")[price_col].sort_index()
        # build hourly-of-year seasonal profile (0..8759)
        # use hour-of-year index: combine dayofyear and hour to get seasonality
        pyr = price_ser.copy().to_frame("price")
        pyr["hour_of_day"] = pyr.index.hour
        pyr["day_of_year"] = pyr.index.dayofyear
        pyr["hour_of_year"] = (pyr["day_of_year"] - 1) * 24 + pyr["hour_of_day"]
        # compute mean price per hour_of_year
        mean_by_hoy = pyr.groupby("hour_of_year")["price"].mean()
        # compute residuals: subtract mean for that hoy
        pyr = pyr.reset_index().set_index("hour_of_year")
        pyr["season_mean"] = mean_by_hoy
        pyr["resid"] = pyr["price"] - pyr["season_mean"]
        resid = pyr["resid"].dropna().values
        st.write(f"Historical hourly observations: {len(price_ser)}. Residuals available: {len(resid)}")
        # monte carlo params
        n_sims = st.slider("Number of Monte Carlo simulations", min_value=100, max_value=5000, value=500, step=100)
        n_years = st.slider("Forecast horizon (years)", min_value=1, max_value=10, value=3, step=1)
        hours_per_year = 8760
        forecast_hours = hours_per_year * n_years
        run_mc = st.button("Run Monte Carlo price forecast")
        if run_mc:
            st.info("Running Monte Carlo simulations — this may take a moment.")
            # build base seasonal profile repeated for forecast horizon: use last full year's pattern if available, else use mean_by_hoy
            base_cycle = mean_by_hoy.values
            # if mean_by_hoy missing some hours (short history), fallback to hourly means
            if len(base_cycle) < 8760:
                # expand by repeating the 24-hour mean many times
                # quick fallback: use hour-of-day mean to fill 8760
                hour_of_day_mean = price_ser.groupby(price_ser.index.hour).mean()
                base_cycle = np.tile(hour_of_day_mean.values, int(np.ceil(8760 / 24)))[:8760]
            # create seasonal vector
            seasonal = np.tile(base_cycle, n_years)
            # bootstrap residuals for each simulation
            sims = np.empty((n_sims, len(seasonal)))
            rng = np.random.default_rng(seed=42)
            for i in range(n_sims):
                draws = rng.choice(resid, size=len(seasonal), replace=True)
                sims[i, :] = seasonal + draws
            # compute ensemble statistics (annual mean, percentiles)
            sims_df_mean = sims.mean(axis=1)
            # prepare visualization: show percentiles across time for ensemble
            p10 = np.percentile(sims, 10, axis=0)
            p50 = np.percentile(sims, 50, axis=0)
            p90 = np.percentile(sims, 90, axis=0)
            # build time index for forecast
            last_ts = price_ser.index.max()
            forecast_index = pd.date_range(start=(last_ts + pd.Timedelta(hours=1)).floor("H"), periods=len(seasonal), freq="H")
            df_ens = pd.DataFrame({"timestamp": forecast_index, "p10": p10, "p50": p50, "p90": p90})
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(x=df_ens["timestamp"], y=df_ens["p50"], name="Median forecast", line=dict(color="black")))
            fig_mc.add_trace(go.Scatter(x=df_ens["timestamp"], y=df_ens["p10"], name="10th percentile", line=dict(color="lightgray")))
            fig_mc.add_trace(go.Scatter(x=df_ens["timestamp"], y=df_ens["p90"], name="90th percentile", line=dict(color="lightgray")))
            fig_mc.update_layout(title=f"Monte Carlo hourly price forecasts: {n_sims} sims, {n_years} years horizon", xaxis_title="Date", yaxis_title=f"{price_col}")
            st.plotly_chart(fig_mc, use_container_width=True)
            # show distribution of annual means across sims
            annual_means = sims.reshape(n_sims, n_years, hours_per_year).mean(axis=2).mean(axis=1) if hours_per_year * n_years == sims.shape[1] else sims.mean(axis=1)
            fig_hist = px.histogram(annual_means, nbins=50, title="Distribution of mean price across simulations")
            st.plotly_chart(fig_hist, use_container_width=True)
            st.write("Monte Carlo summary statistics (mean of simulated means):", np.mean(annual_means), "Std:", np.std(annual_means))
        else:
            st.info("Configure simulations and click 'Run Monte Carlo price forecast' to run.")
    else:
        st.info("No Price column detected in Hourly_Generation_Prices — Monte Carlo price forecast cannot be run.")

# -------------------------
# Market Installed Capacity sheet visuals (separate)
# -------------------------
if mic is not None:
    st.header("Installed Capacity — details and sanity checks")
    st.subheader("Preview")
    st.dataframe(mic.head(200))
    st.markdown("If Market_Installed_Capacity contains region/type/year columns we melt to long form for visualization.")
    year_cols = extract_year_cols(mic)
    mic_region_col = safe_col(mic, "region") or (mic.columns[0] if len(mic.columns) > 0 else None)
    mic_type_col = safe_col(mic, "type") or safe_col(mic, "technology") or safe_col(mic, "category")
    st.write("Detected region column:", mic_region_col, "type column:", mic_type_col, "year columns:", year_cols)
    if year_cols:
        long_mic = mic.melt(id_vars=[c for c in [mic_region_col, mic_type_col] if c], value_vars=year_cols, var_name="Year", value_name="Capacity_GW")
        long_mic["Year_int"] = long_mic["Year"].astype(str).str.extract("(\d{4})").astype(float)
        st.subheader("Installed capacity trajectories")
        if mic_type_col:
            fig = px.area(long_mic, x="Year_int", y="Capacity_GW", color=mic_type_col, facet_col=mic_region_col if mic_region_col else None, title="Installed capacity by technology (GW)")
        else:
            fig = px.line(long_mic.groupby("Year_int")["Capacity_GW"].sum().reset_index(), x="Year_int", y="Capacity_GW", title="Total installed capacity (GW)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No year columns found for installed capacity (2021-2050 expected).")

# -------------------------
# Market_Generation sheet
# -------------------------
if mg is not None:
    st.header("Market Generation (aggregate categories)")
    st.dataframe(mg.head(200))
    try:
        mg["timestamp"] = build_timestamp_from_hmdh(mg)
    except Exception:
        date_col = safe_col(mg, "date")
        if date_col:
            mg["timestamp"] = pd.to_datetime(mg[date_col], errors="coerce")
        else:
            mg["timestamp"] = pd.NaT

    category_col = safe_col(mg, "category")
    supply_col = safe_col(mg, "supply") or safe_col(mg, "supply mwh")
    st.write("Detected category column:", category_col, "supply column:", supply_col)
    if category_col and supply_col:
        freq_mg = st.selectbox("Resample freq for Market_Generation", options=["D", "W", "M"], index=2)
        tmp = mg.dropna(subset=["timestamp", category_col, supply_col]).set_index("timestamp")
        grouped = tmp.groupby(category_col)[supply_col].resample(freq_mg).sum().reset_index()
        fig_mg = px.line(grouped, x="timestamp", y=supply_col, color=category_col, title="Market generation by category")
        st.plotly_chart(fig_mg, use_container_width=True)
    else:
        st.info("Market_Generation missing category or supply columns.")

# -------------------------
# Footer / narrative
# -------------------------
st.markdown("---")
st.subheader("Summary & next steps")
st.write(
    "This interactive dashboard provides:\n"
    "- Normalized hourly generation data with duplicate resolution and capacity checks.\n"
    "- Visual tools to explore generation mix across time, utilization (capacity factors), and irregularities.\n"
    "- Peak demand historical analysis and simple linear forecasts for 2030-2040.\n"
    "- Monte Carlo price forecast demo based on hourly price seasonality + bootstrapped residuals.\n\n"
    "To improve forecasts and analytics further, consider:\n"
    "- Using a more sophisticated time-series model (SARIMA, Prophet, or state-space models) for prices.\n"
    "- Incorporating more explicit marginal cost / merit-order data for dispatch ordering.\n"
    "- Adding regional network constraints / interconnector details for net imports analysis.\n"
    "- Validating types/tokens in Market_Installed_Capacity and enforcing a technology mapping table for robust capacity lookups."
)
st.caption("Developed as an interactive demo for power market research. For further customization, provide the actual spreadsheet and notes on any deviations in column names or technology type labels.")