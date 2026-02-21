"""
NYC Yellow Taxi Trip Dashboard
Streamlit application for interactive exploration of January 2024 taxi trip data.
"""

import streamlit as st
import pandas as pd
import polars as pl
import plotly.express as px
import requests
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Yellow Taxi Dashboard",
    page_icon="🚕",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
TRIP_URL  = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
ZONE_URL  = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
RAW_DIR   = "data/raw"
TRIP_PATH = os.path.join(RAW_DIR, "yellow_tripdata_2024-01.parquet")
ZONE_PATH = os.path.join(RAW_DIR, "taxi_zone_lookup.csv")

PAYMENT_MAP = {1: "Credit Card", 2: "Cash", 3: "No Charge", 4: "Dispute", 5: "Unknown"}
DAY_ORDER   = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _download_if_missing(url: str, path: str) -> None:
    """Stream-download a file only if it doesn't already exist."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)


@st.cache_data(show_spinner="Downloading & loading data…")
def load_data() -> pd.DataFrame:
    """Download (if needed), clean, and return the full trips DataFrame."""
    _download_if_missing(TRIP_URL, TRIP_PATH)
    _download_if_missing(ZONE_URL, ZONE_PATH)

    # Load with Polars for speed
    df    = pl.read_parquet(TRIP_PATH)
    zones = pl.read_csv(ZONE_PATH)

    # ── Cleaning ──────────────────────────────────────────────────────────────
    critical_cols = [
        "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "PULocationID", "DOLocationID", "fare_amount",
    ]
    df = df.drop_nulls(subset=critical_cols)
    df = df.filter(
        (pl.col("trip_distance") > 0)
        & (pl.col("fare_amount") > 0)
        & (pl.col("fare_amount") <= 500)
        & (pl.col("tpep_dropoff_datetime") > pl.col("tpep_pickup_datetime"))
    )

    # ── Feature engineering ───────────────────────────────────────────────────
    df = df.with_columns([
        ((pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime"))
         .dt.total_seconds() / 60).alias("trip_duration_minutes"),
        pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
        pl.col("tpep_pickup_datetime").dt.strftime("%A").alias("pickup_day_of_week"),
    ])

    # Speed – guard against division by zero
    df = df.with_columns(
        pl.when(pl.col("trip_duration_minutes") > 0)
        .then(pl.col("trip_distance") / (pl.col("trip_duration_minutes") / 60))
        .otherwise(None)
        .alias("trip_speed_mph")
    )

    # ── Join zone names ───────────────────────────────────────────────────────
    zones_pu = (zones
                .rename({"LocationID": "PULocationID", "Zone": "pickup_zone",
                         "Borough": "pickup_borough"})
                .select(["PULocationID", "pickup_zone", "pickup_borough"]))
    zones_do = (zones
                .rename({"LocationID": "DOLocationID", "Zone": "dropoff_zone",
                         "Borough": "dropoff_borough"})
                .select(["DOLocationID", "dropoff_zone", "dropoff_borough"]))
    df = df.join(zones_pu, on="PULocationID", how="left")
    df = df.join(zones_do, on="DOLocationID", how="left")

    # Payment label – use map_elements for broad Polars version compatibility
    df = df.with_columns(
        pl.col("payment_type")
        .cast(pl.Int64)
        .map_elements(lambda x: PAYMENT_MAP.get(x, "Unknown"), return_dtype=pl.Utf8)
        .alias("payment_label")
    )

    return df.to_pandas()


# ── Load data ─────────────────────────────────────────────────────────────────
df_full = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.title("🔍 Filters")

# Date range – use normalize() for reliable date comparison across Pandas versions
min_date = df_full["tpep_pickup_datetime"].dt.normalize().min().date()
max_date = df_full["tpep_pickup_datetime"].dt.normalize().max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Hour range slider
hour_range = st.sidebar.slider("Pickup Hour Range", 0, 23, (0, 23))

# Payment type multiselect
all_payment_types = sorted(df_full["payment_label"].dropna().unique().tolist())
selected_payments = st.sidebar.multiselect(
    "Payment Type",
    options=all_payment_types,
    default=all_payment_types,
)

# ── Apply filters ─────────────────────────────────────────────────────────────
# Handle the case where the user has only picked one date (date_input returns a single value)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0] if isinstance(date_range, (list, tuple)) else date_range

# Build boolean mask across all three filters
mask = (
    (df_full["tpep_pickup_datetime"].dt.normalize().dt.date >= start_date)
    & (df_full["tpep_pickup_datetime"].dt.normalize().dt.date <= end_date)
    & (df_full["pickup_hour"] >= hour_range[0])
    & (df_full["pickup_hour"] <= hour_range[1])
    & (df_full["payment_label"].isin(selected_payments if selected_payments else all_payment_types))
)
df = df_full[mask].copy()

# Guard against empty DataFrame (e.g. all payment types deselected)
if df.empty:
    st.warning("No trips match the current filters. Please adjust your sidebar selections.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚕 NYC Yellow Taxi Trip Dashboard — January 2024")
st.markdown(
    """
    This dashboard explores **~3 million Yellow Taxi trips** recorded in New York City during
    January 2024. Use the sidebar filters to slice by date, hour, and payment type.
    All charts update dynamically with your selection.
    """
)

# ── KPI metrics ───────────────────────────────────────────────────────────────
st.subheader("📊 Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Trips",        f"{len(df):,}")
col2.metric("Avg Fare",           f"${df['fare_amount'].mean():.2f}")
col3.metric("Total Revenue",      f"${df['total_amount'].sum():,.0f}")
col4.metric("Avg Distance (mi)",  f"{df['trip_distance'].mean():.2f}")
col5.metric("Avg Duration (min)", f"{df['trip_duration_minutes'].mean():.1f}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Top Pickup Zones",
    " Hourly Fare Patterns",
    " Trip Distance",
    " Payment Types",
    " Weekly Heatmap",
])

# ── Tab 1: Top 10 pickup zones ────────────────────────────────────────────────
with tab1:
    st.subheader("Top 10 Pickup Zones by Trip Count")
    top_zones = (
        df.groupby("pickup_zone", dropna=True)
        .size()
        .reset_index(name="trip_count")
        .nlargest(10, "trip_count")
        .sort_values("trip_count")
    )
    fig1 = px.bar(
        top_zones,
        x="trip_count",
        y="pickup_zone",
        orientation="h",
        labels={"trip_count": "Number of Trips", "pickup_zone": "Pickup Zone"},
        color="trip_count",
        color_continuous_scale="Oranges",
        template="plotly_white",
    )
    fig1.update_layout(coloraxis_showscale=False, height=420)
    st.plotly_chart(fig1, use_container_width=True)
    st.caption(
        "Midtown Manhattan zones (Penn Station, Times Square, Grand Central) dominate trip pickups, "
        "reflecting the density of hotels, transit hubs, and offices in the area. "
        "JFK Airport also ranks highly, driven by airport-to-city rides that are a staple of NYC cab demand."
    )

# ── Tab 2: Average fare by hour ───────────────────────────────────────────────
with tab2:
    st.subheader("Average Fare Amount by Hour of Day")
    fare_by_hour = (
        df.groupby("pickup_hour")["fare_amount"]
        .mean()
        .reset_index()
        .rename(columns={"fare_amount": "avg_fare"})
        .sort_values("pickup_hour")
    )
    fig2 = px.line(
        fare_by_hour,
        x="pickup_hour",
        y="avg_fare",
        markers=True,
        labels={"pickup_hour": "Hour of Day (0–23)", "avg_fare": "Average Fare ($)"},
        template="plotly_white",
    )
    fig2.update_traces(line_color="#F4A522", line_width=3)
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Fares peak in the early morning hours (4–6 AM), likely driven by airport runs and long-haul night trips. "
        "A secondary peak appears during evening rush hour (5–7 PM) when demand increases. "
        "Midday trips tend to be shorter in distance, pulling the average fare down."
    )

# ── Tab 3: Trip distance distribution ────────────────────────────────────────
with tab3:
    st.subheader("Distribution of Trip Distances")
    # Clip at 30 miles so outliers don't squash the chart
    dist_clipped = df[df["trip_distance"] <= 30]["trip_distance"]
    fig3 = px.histogram(
        dist_clipped,
        x="trip_distance",
        nbins=60,
        labels={"trip_distance": "Trip Distance (miles)", "count": "Number of Trips"},
        color_discrete_sequence=["#F4A522"],
        template="plotly_white",
    )
    fig3.update_layout(height=400, bargap=0.05)
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "The distribution is strongly right-skewed: the vast majority of trips are under 5 miles, "
        "consistent with short urban hops within Manhattan. "
        "The long tail represents inter-borough and airport trips, which are far less frequent but significantly longer."
    )

# ── Tab 4: Payment types ──────────────────────────────────────────────────────
with tab4:
    st.subheader("Trip Breakdown by Payment Type")
    pay_counts = (
        df.groupby("payment_label", dropna=True)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    fig4 = px.pie(
        pay_counts,
        names="payment_label",
        values="count",
        color_discrete_sequence=px.colors.qualitative.Set2,
        template="plotly_white",
    )
    fig4.update_traces(textposition="inside", textinfo="percent+label")
    fig4.update_layout(height=420)
    st.plotly_chart(fig4, use_container_width=True)
    st.caption(
        "Credit card payments account for the large majority of trips, reflecting the widespread adoption "
        "of in-cab payment terminals. Cash remains significant but is a minority. "
        "Note that tip data is only captured electronically for credit card transactions, "
        "meaning tip analysis under-represents actual tipping behaviour."
    )

# ── Tab 5: Day-of-week × hour heatmap ────────────────────────────────────────
with tab5:
    st.subheader("Trip Count by Day of Week and Hour")
    heatmap_data = (
        df.groupby(["pickup_day_of_week", "pickup_hour"])
        .size()
        .reset_index(name="trip_count")
    )

    # Guard: heatmap needs at least some data to pivot
    if heatmap_data.empty:
        st.info("Not enough data to display the heatmap for the current filters.")
    else:
        heatmap_pivot = (
            heatmap_data
            .pivot(index="pickup_day_of_week", columns="pickup_hour", values="trip_count")
            .reindex([d for d in DAY_ORDER if d in heatmap_data["pickup_day_of_week"].unique()])
        )
        fig5 = px.imshow(
            heatmap_pivot,
            labels={"x": "Hour of Day", "y": "Day of Week", "color": "Trip Count"},
            color_continuous_scale="YlOrRd",
            aspect="auto",
            template="plotly_white",
        )
        fig5.update_layout(height=420)
        st.plotly_chart(fig5, use_container_width=True)
        st.caption(
            "Weekday mornings (7–9 AM) and evenings (5–8 PM) show the classic commuter surge pattern, "
            "with Friday evenings being the single busiest period of the week. "
            "Weekend nights (Friday and Saturday, 10 PM – 2 AM) also show elevated demand, "
            "driven by nightlife activity."
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<small>Data source: NYC Taxi & Limousine Commission (TLC) — January 2024</small>",
    unsafe_allow_html=True,
)
