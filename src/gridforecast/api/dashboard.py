"""
GridForecast — Streamlit Dashboard.

Features:
  - Single prediction: генерира 168h context, вика /predict, показва резултат
  - Batch prediction: качва CSV, вика /predict/batch, изтегля predictions
  - Model info: показва drift summary с emoji mapping
  - History: session state с предишни predictions

Пусни:
  streamlit run src/gridforecast/api/dashboard.py
"""
from __future__ import annotations

import io
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════
API_URL = os.getenv("API_URL", "http://localhost:9001")

# ASCII → emoji mapping (от API-то идват ASCII кодове)
STATUS_EMOJI = {
    "DRIFT": "🚨",
    "OK": "✅",
    "WARN": "⚠️",
    "MODERATE": "⚠️",
    "IGNORE": "🔕",
    "RETRAIN_NOW": "🚨",
    "MONITOR": "⚠️",
    "NO_ACTION_STABLE": "✅",
}


def status_badge(status: str) -> str:
    """Връща emoji + ASCII status."""
    emoji = STATUS_EMOJI.get(status, "❓")
    return f"{emoji} {status}"


# ═══════════════════════════════════════════════════════════
# Page setup
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GridForecast",
    page_icon="⚡",
    layout="wide",
)


# ═══════════════════════════════════════════════════════════
# Sidebar: API status
# ═══════════════════════════════════════════════════════════
st.sidebar.header("⚡ GridForecast")

st.sidebar.code(API_URL, language="text")

try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    if health.get("model_loaded"):
        st.sidebar.success(f"API: {health.get('status', 'healthy')}")
        st.sidebar.caption(f"Last check: {health.get('timestamp', 'N/A')[:19]}")
    else:
        st.sidebar.warning("API: degraded")
except Exception as e:
    st.sidebar.error(f"API unreachable: {e}")

st.sidebar.divider()

# Model info (ако API-то работи)
try:
    info = requests.get(f"{API_URL}/model-info", timeout=3).json()
    st.sidebar.metric("Model", info.get("model_name", "N/A"))
    st.sidebar.metric("MAPE", f"{info.get('fit_mape', 0):.2f}%")
    st.sidebar.metric("Features", info.get("n_features", 0))
except Exception:
    pass


# ═══════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════
tab_single, tab_batch, tab_drift, tab_history = st.tabs([
    "🔮 Single prediction",
    "📦 Batch",
    "📊 Drift summary",
    "📜 History",
])


# ═══════════════════════════════════════════════════════════
# Tab 1: Single prediction
# ═══════════════════════════════════════════════════════════
with tab_single:
    st.subheader("Predict next hour from 168h history")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Inputs**")
        base_consumption = st.number_input(
            "Base consumption (MW)",
            value=1100.0,
            step=10.0,
            help="Средна консумация за последните 168h",
        )
        base_temp = st.number_input(
            "Temperature (°C)",
            value=15.0,
            step=1.0,
        )
        is_weekend = st.checkbox("Is weekend?", value=False)
        is_holiday = st.checkbox("Is holiday?", value=False)
        start_date = st.date_input(
            "Context end date",
            value=datetime(2024, 12, 1).date(),
        )
        st.caption(
            f"Context: 168h ending {start_date} 23:00. "
            f"Prediction: {start_date} + 1 day 00:00."
        )

    with col2:
        if st.button("🚀 Predict", type="primary", use_container_width=True):
            # 1. Build synthetic 168h history
            end_ts = datetime.combine(start_date, datetime.min.time()) + timedelta(hours=167)
            ts_list = pd.date_range(end=end_ts, periods=168, freq="h")
            rng = np.random.default_rng(42)
            consumption = (
                base_consumption
                + 50 * np.sin(2 * np.pi * np.arange(168) / 24)
                + rng.normal(0, 20, 168)
            )
            consumption = np.maximum(consumption, 100.0)

            # 2. Prediction timestamp = last + 1h
            pred_ts = ts_list[-1] + pd.Timedelta(hours=1)

            # 3. Build payload
            payload = {
                "timestamp": pred_ts.isoformat(),
                "temperature_c": float(base_temp),
                "is_holiday": bool(is_holiday),
                "is_weekend": bool(is_weekend),
                "history": [
                    {
                        "timestamp": ts.isoformat(),
                        "consumption_mw": float(c),
                    }
                    for ts, c in zip(ts_list, consumption)
                ],
            }

            with st.spinner("Calling API..."):
                try:
                    r = requests.post(
                        f"{API_URL}/predict",
                        json=payload,
                        timeout=30,
                    )
                    r.raise_for_status()
                    data = r.json()

                    # 4. Show result
                    st.success(f"✅ Predicted: **{data['predicted_mw']:.2f} MW**")
                    st.caption(
                        f"Timestamp: {data['timestamp']} | "
                        f"Latency: {data['latency_ms']:.1f} ms | "
                        f"Model: {data['model_name']} {data['model_version']}"
                    )

                    # 5. Plot history + prediction
                    hist_df = pd.DataFrame({
                        "timestamp": ts_list,
                        "consumption_mw": consumption,
                    })
                    pred_df = pd.DataFrame([{
                        "timestamp": pd.Timestamp(pred_ts),
                        "consumption_mw": data["predicted_mw"],
                    }])
                    chart_df = pd.concat([hist_df, pred_df], ignore_index=True)

                    st.line_chart(
                        chart_df,
                        x="timestamp",
                        y="consumption_mw",
                        height=300,
                    )

                    # 6. Save to history
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({
                        "timestamp": data["timestamp"],
                        "predicted_mw": data["predicted_mw"],
                        "latency_ms": data["latency_ms"],
                        "model": data["model_name"],
                        "source": "single",
                    })

                except requests.HTTPError as e:
                    st.error(f"API error: {e}")
                    try:
                        st.code(r.json(), language="json")
                    except Exception:
                        st.code(r.text)
                except Exception as e:
                    st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════
# Tab 2: Batch
# ═══════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("Batch prediction from CSV")
    st.caption(
        "CSV format: `group_id, timestamp, consumption_mw` "
        "(всяка група трябва да има ≥ 168 реда history)"
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded, parse_dates=["timestamp"])
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Total rows: {len(df):,} | Groups: {df['group_id'].nunique()}")

        if st.button("📦 Predict batch", type="primary"):
            # 1. Group by group_id, take last 168 rows
            requests_list = []
            skipped = 0

            for gid, g in df.groupby("group_id"):
                g = g.sort_values("timestamp").tail(168)
                if len(g) < 168:
                    skipped += 1
                    continue

                last_ts = g["timestamp"].iloc[-1]
                pred_ts = last_ts + pd.Timedelta(hours=1)

                context = [
                    {
                        "timestamp": row.timestamp.isoformat(),
                        "consumption_mw": float(row.consumption_mw),
                    }
                    for row in g.itertuples()
                ]

                requests_list.append({
                    "timestamp": pred_ts.isoformat(),
                    "temperature_c": 15.0,  # default
                    "is_holiday": False,
                    "is_weekend": pred_ts.dayofweek >= 5,
                    "history": context,
                })

            if skipped > 0:
                st.warning(f"Skipped {skipped} groups (history < 168)")

            if not requests_list:
                st.error("No valid groups (all have < 168 rows)")
            else:
                with st.spinner(f"Calling API for {len(requests_list)} groups..."):
                    try:
                        r = requests.post(
                            f"{API_URL}/predict/batch",
                            json={"requests": requests_list},
                            timeout=120,
                        )
                        r.raise_for_status()
                        data = r.json()

                        rows = []
                        for i, pred in enumerate(data["predictions"]):
                            rows.append({
                                "group_id": i,
                                "timestamp": pred["timestamp"],
                                "predicted_mw": pred["predicted_mw"],
                            })
                        out = pd.DataFrame(rows)

                        st.success(
                            f"✅ Generated {len(out)} predictions "
                            f"in {data['total_latency_ms']:.0f} ms"
                        )
                        st.dataframe(out, use_container_width=True)

                        csv = out.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download CSV",
                            csv,
                            "predictions.csv",
                            "text/csv",
                        )

                        # Save to history
                        if "history" not in st.session_state:
                            st.session_state.history = []
                        for _, row in out.iterrows():
                            st.session_state.history.append({
                                "timestamp": row["timestamp"],
                                "predicted_mw": row["predicted_mw"],
                                "latency_ms": data["total_latency_ms"] / len(out),
                                "model": "LightGBM",
                                "source": f"batch (group {row['group_id']})",
                            })

                    except Exception as e:
                        st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════
# Tab 3: Drift summary
# ═══════════════════════════════════════════════════════════
with tab_drift:
    st.subheader("Drift summary (from /model-info)")

    try:
        info = requests.get(f"{API_URL}/model-info", timeout=5).json()
        drift = info.get("drift_summary", [])

        if not drift:
            st.info("No drift summary available.")
        else:
            # Overall verdict
            verdict_row = next(
                (r for r in drift if r["method"] == "FINAL VERDICT"),
                None,
            )
            if verdict_row:
                st.markdown(
                    f"### {status_badge(verdict_row['status'])}"
                )
                st.caption(verdict_row.get("note", ""))

            # Table with emoji mapping
            table_rows = []
            for r in drift:
                table_rows.append({
                    "Method": r["method"],
                    "Value": f"{r['value']:.4g}",
                    "Threshold": f"{r['threshold']:.2f}",
                    "Status": status_badge(r["status"]),
                    "Weight": r["weight"],
                    "Note": r.get("note", ""),
                })
            st.dataframe(
                pd.DataFrame(table_rows),
                use_container_width=True,
                hide_index=True,
            )

            # Metrics (само HIGH/CRIT)
            st.markdown("### Key metrics (HIGH/CRIT weight)")
            high_metrics = [r for r in drift if r["weight"] in ("HIGH", "CRIT")]

            cols = st.columns(len(high_metrics))
            for col, r in zip(cols, high_metrics):
                with col:
                    emoji = STATUS_EMOJI.get(r["status"], "❓")
                    st.metric(
                        label=r["method"],
                        value=f"{r['value']:.4f}",
                        delta=f"{emoji} {r['status']}",
                    )

    except Exception as e:
        st.error(f"Error loading drift summary: {e}")


# ═══════════════════════════════════════════════════════════
# Tab 4: History
# ═══════════════════════════════════════════════════════════
with tab_history:
    st.subheader("Session history")

    if "history" not in st.session_state:
        st.session_state.history = []

    if not st.session_state.history:
        st.info("No predictions yet in this session.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Clear history"):
                st.session_state.history = []
                st.rerun()
        with col2:
            csv = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download history CSV",
                csv,
                "history.csv",
                "text/csv",
            )