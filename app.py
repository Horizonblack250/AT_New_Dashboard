import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Airtrol Flow Accuracy Dashboard", layout="wide")

# Calibration parameters (from your fit)
A_GAIN = 0.958426
B_OFFSET = 6.549856

@st.cache_data
def load_data():
    df = pd.read_csv("df_clean.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Choose actual flow column: prefer shifted, fallback to raw VFM
    if "VFM_Flow_shifted" in df.columns:
        actual_col = "VFM_Flow_shifted"
    else:
        actual_col = "VFM Flow Rate (SCFM)"

    # Ensure required column exists
    if "Flow_Rate_Calculated_SCFM" not in df.columns:
        raise ValueError("Column 'Flow_Rate_Calculated_SCFM' not found in df_clean.csv")

    # Compute corrected flow using your calibration equation
    df["Flow_Rate_Corrected_SCFM"] = (
        A_GAIN * df["Flow_Rate_Calculated_SCFM"].astype(float) + B_OFFSET
    )

    # Compute error % vs actual (using corrected flow)
    actual = df[actual_col].astype(float)
    corrected = df["Flow_Rate_Corrected_SCFM"].astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        df["Error_Pct"] = np.where(
            actual > 0,
            (actual - corrected) / actual * 100.0,
            np.nan,
        )

    return df, actual_col

df, actual_col = load_data()

st.title("Airtrol – Corrected Flow vs VFM")

# ---------------- Sidebar Controls ----------------
st.sidebar.header("Filters")

min_time = df["Timestamp"].min()
max_time = df["Timestamp"].max()

start_date = st.sidebar.date_input("Start Date", min_time.date())
start_time = st.sidebar.time_input("Start Time", min_time.time())

end_date = st.sidebar.date_input("End Date", max_time.date())
end_time = st.sidebar.time_input("End Time", max_time.time())

start_dt = datetime.combine(start_date, start_time)
end_dt = datetime.combine(end_date, end_time)

show_error = st.sidebar.checkbox("Show Error % Line", value=True)

# Filter range
mask = (df["Timestamp"] >= start_dt) & (df["Timestamp"] <= end_dt)
df_view = df.loc[mask].copy()

if df_view.empty:
    st.warning("No data available in the selected time window. Try expanding the range.")
    st.stop()

# ---------------- Plot ----------------
fig = go.Figure()

# Actual VFM flow
fig.add_trace(go.Scatter(
    x=df_view["Timestamp"],
    y=df_view[actual_col],
    mode="lines",
    name="Actual VFM Flow (SCFM)",
    line=dict(width=2)  # solid
))

# Corrected calculated flow
fig.add_trace(go.Scatter(
    x=df_view["Timestamp"],
    y=df_view["Flow_Rate_Corrected_SCFM"],
    mode="lines",
    name="Corrected Calculated Flow (SCFM)",
    line=dict(width=2, color="royalblue")  # solid
))

# Error % (secondary axis) with toggle
if show_error:
    err = df_view["Error_Pct"].copy()

    # For plotting only: clip extreme spikes so scale is readable
    # (keeps KPIs using full true error)
    q_low, q_high = np.nanpercentile(err, [1, 99])
    err_clipped = err.clip(q_low, q_high)

    fig.add_trace(go.Scatter(
        x=df_view["Timestamp"],
        y=err_clipped,
        mode="lines",
        name="Error % (clipped for view)",
        yaxis="y2",
        line=dict(width=2, color="red")  # solid
    ))

    fig.update_layout(
        yaxis2=dict(
            title="Error % (visualized, clipped)",
            overlaying="y",
            side="right",
            showgrid=False,
        )
    )

fig.update_layout(
    title="Actual vs Corrected Calculated Flow",
    xaxis=dict(title="Timestamp"),
    yaxis=dict(title="Flow (SCFM)"),
    hovermode="x unified",
    legend=dict(x=0, y=1.1, orientation="h"),
    margin=dict(l=40, r=40, t=60, b=40),
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- KPIs ----------------
st.subheader("Performance Metrics (Selected Window)")

err_window = df_view["Error_Pct"]
valid_err_mask = err_window.notna() & np.isfinite(err_window)

if valid_err_mask.any():
    mean_abs_err = err_window[valid_err_mask].abs().mean()
    pct_within_10 = (err_window[valid_err_mask].abs() <= 10).mean() * 100.0
else:
    mean_abs_err = np.nan
    pct_within_10 = np.nan

col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error %", f"{mean_abs_err:.2f}%" if np.isfinite(mean_abs_err) else "N/A")
col2.metric("% Within ±10%", f"{pct_within_10:.1f}%" if np.isfinite(pct_within_10) else "N/A")

# Optional: show small data preview
with st.expander("Show Data Preview"):
    st.dataframe(df_view[["Timestamp", actual_col, "Flow_Rate_Calculated_SCFM",
                          "Flow_Rate_Corrected_SCFM", "Error_Pct"]].head(50))
