
"""
Quick-Commerce Delivery Intelligence
AI-Powered ML Dashboard

Main Streamlit application entry point.
"""

import sys
from pathlib import Path

# ============================================================
# PROJECT PATH CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
import config

from app.styles import apply_custom_theme

from app.views import (
    page_executive,
    page_data_explorer,
    page_eta_prediction,
    page_late_risk,
    page_demand_zones,
    page_rider_segmentation,
    page_model_performance,
    page_business_recommendations
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Quick-Commerce ML Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM THEME
# ============================================================

apply_custom_theme()


# ============================================================
# NAVIGATION CONFIGURATION
# ============================================================

NAVIGATION_OPTIONS = {
    "📊 ML Dashboard": page_executive.render_page,

    "⚡ ETA Prediction": page_eta_prediction.render_page,

    "🚨 Late-Order Prediction": page_late_risk.render_page,

    "🔍 Data Explorer": page_data_explorer.render_page,

    "🗺️ Demand Zone Analysis": page_demand_zones.render_page,

    "🚴 Rider Segmentation": page_rider_segmentation.render_page,

    "🧪 Model Performance": page_model_performance.render_page,

    "💡 Business Recommendations": (
        page_business_recommendations.render_page
    )
}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚡ QUICK-COMMERCE")

st.sidebar.caption(
    "AI-Powered Delivery Intelligence"
)

st.sidebar.divider()

st.sidebar.subheader("Navigation")

selected_page = st.sidebar.radio(
    "Select a dashboard",
    options=list(NAVIGATION_OPTIONS.keys()),
    index=0,
    key="main_navigation"
)

st.sidebar.divider()


# ============================================================
# SYSTEM CONFIGURATION
# ============================================================

st.sidebar.subheader("System Configuration")

st.sidebar.metric(
    label="Default Delivery SLA",
    value=f"{config.DEFAULT_SLA_MINUTES:.0f} min"
)

st.sidebar.caption(
    "Delivery Intelligence Platform"
)

st.sidebar.caption(
    "ML Dashboard v1.1"
)

st.sidebar.divider()

st.sidebar.caption(
    "Powered by Python, Streamlit and XGBoost"
)


# ============================================================
# MAIN APPLICATION HEADER
# ============================================================

st.title("⚡ Quick-Commerce Delivery Intelligence")

st.caption(
    "AI-powered delivery analytics, ETA prediction, "
    "late-order risk assessment and operational intelligence."
)

st.divider()


# ============================================================
# PAGE ROUTING
# ============================================================

try:

    render_selected_page = NAVIGATION_OPTIONS.get(
        selected_page
    )

    if render_selected_page is None:
        st.error("Selected page is not available.")

    else:
        render_selected_page()


except Exception as e:

    st.error(
        f"Unable to load {selected_page}."
    )

    with st.expander(
        "View technical error details",
        expanded=True
    ):
        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Quick-Commerce Delivery Intelligence | "
    "AI-Powered ML Dashboard"
)
