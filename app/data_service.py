
"""
Centralized Data and Model Access Service.

Handles:
- Streamlit caching
- Dataset loading
- Model deserialization
- Existing ETA and late-risk inference
- Unified ETA and late-risk inference
- Spatial and rider clustering model access
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging

import joblib
import pandas as pd
import numpy as np
import streamlit as st

import config
from src.feature_engineering import FeatureEngineer
from src.recommendations import DeliveryOptimizer
from src.prediction import DeliveryInferencePipeline


logger = logging.getLogger(__name__)

FEATURED_CSV = config.DATA_DIR / "featured_delivery_data.csv"
COMPARISON_CSV = config.REPORTS_DIR / "model_comparison.csv"


# ============================================================
# DATASET LOADERS
# ============================================================

@st.cache_data(show_spinner="Loading featured delivery dataset...")
def load_featured_dataset() -> pd.DataFrame:
    """Load and cache the cleaned and featured delivery dataset."""

    if not FEATURED_CSV.exists():
        st.error(
            f"Dataset not found at {FEATURED_CSV}. "
            "Please run the pipeline cleaning process."
        )
        return pd.DataFrame()

    df = pd.read_csv(FEATURED_CSV)

    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(
            df["order_date"],
            errors="coerce"
        )

    return df


@st.cache_data(show_spinner="Loading model performance ledger...")
def load_model_comparison_df() -> pd.DataFrame:
    """Load the evaluated model comparison CSV."""

    if not COMPARISON_CSV.exists():
        return pd.DataFrame()

    return pd.read_csv(COMPARISON_CSV)


# ============================================================
# TRAINED MODEL LOADERS
# ============================================================

@st.cache_resource(show_spinner="Loading best ETA regression model...")
def load_regressor_bundle() -> Optional[Dict[str, Any]]:
    """Load and cache the trained ETA regressor bundle."""

    model_path = config.MODELS_DIR / "best_eta_regressor.joblib"

    if not model_path.exists():
        logger.warning("Regressor artifact missing: %s", model_path)
        return None

    try:
        return joblib.load(model_path)

    except Exception as e:
        logger.exception("Failed to load ETA regressor: %s", e)
        return None


@st.cache_resource(
    show_spinner="Loading best late-order classification model..."
)
def load_classifier_bundle() -> Optional[Dict[str, Any]]:
    """Load and cache the trained late-order classifier bundle."""

    model_path = config.MODELS_DIR / "best_late_classifier.joblib"

    if not model_path.exists():
        logger.warning("Classifier artifact missing: %s", model_path)
        return None

    try:
        return joblib.load(model_path)

    except Exception as e:
        logger.exception("Failed to load late-order classifier: %s", e)
        return None


@st.cache_resource(
    show_spinner="Loading dark store spatial clustering model..."
)
def load_dark_store_bundle() -> Optional[Dict[str, Any]]:
    """Load and cache the spatial K-Means clustering artifact."""

    model_path = config.MODELS_DIR / "dark_store_kmeans.joblib"

    if not model_path.exists():
        logger.warning("Dark store clustering artifact missing.")
        return None

    try:
        return joblib.load(model_path)

    except Exception as e:
        logger.exception("Failed to load spatial clusters: %s", e)
        return None


@st.cache_resource(
    show_spinner="Loading courier segmentation model..."
)
def load_rider_segmentation_bundle() -> Optional[Dict[str, Any]]:
    """Load and cache the delivery partner segmentation artifact."""

    model_path = config.MODELS_DIR / "rider_segmentation_kmeans.joblib"

    if not model_path.exists():
        logger.warning("Rider segmentation artifact missing.")
        return None

    try:
        return joblib.load(model_path)

    except Exception as e:
        logger.exception("Failed to load rider segmentation: %s", e)
        return None


# ============================================================
# UNIFIED ML INFERENCE PIPELINE
# ============================================================

@st.cache_resource(
    show_spinner="Initializing unified ML inference pipeline..."
)
def load_unified_prediction_pipeline() -> Optional[DeliveryInferencePipeline]:
    """
    Load one shared DeliveryInferencePipeline instance.

    The pipeline handles:
    - Data cleaning
    - Feature engineering
    - ETA regression
    - Late-order classification
    - Calibrated classification threshold
    - Fallback handling
    - Operational recommendations
    """

    try:
        pipeline = DeliveryInferencePipeline()

        status = pipeline.get_model_status()

        logger.info(
            "Unified inference pipeline initialized. Model status: %s",
            status
        )

        if not status.get("eta_model_loaded", False):
            logger.warning("ETA model is not available.")

        if not status.get("classifier_model_loaded", False):
            logger.warning("Late-order classifier is not available.")

        return pipeline

    except Exception as e:
        logger.exception(
            "Failed to initialize unified inference pipeline: %s",
            e
        )
        return None


def get_unified_model_status() -> Dict[str, Any]:
    """Return the loading status of both trained models."""

    pipeline = load_unified_prediction_pipeline()

    if pipeline is None:
        return {
            "pipeline_loaded": False,
            "eta_model_loaded": False,
            "classifier_model_loaded": False,
            "error": "Unified inference pipeline could not be initialized."
        }

    status = pipeline.get_model_status()
    status["pipeline_loaded"] = True

    return status


def execute_unified_prediction(
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate ETA and late-order predictions for one delivery order.

    Both predictions use the same DeliveryInferencePipeline.

    Args:
        payload: Dictionary containing delivery order features.

    Returns:
        Dictionary containing ETA, late-risk probability,
        risk category, SLA, fleet recommendation and mitigation plan.

    Raises:
        ValueError: If the input payload is invalid.
        RuntimeError: If the inference pipeline cannot initialize.
    """

    if not isinstance(payload, dict) or not payload:
        raise ValueError(
            "Prediction payload must be a non-empty dictionary."
        )

    pipeline = load_unified_prediction_pipeline()

    if pipeline is None:
        raise RuntimeError(
            "Unified ML inference pipeline could not be initialized."
        )

    try:
        result = pipeline.predict_single(payload)

        logger.info(
            "Unified prediction completed. ETA=%s, Late risk=%s, "
            "Fallback=%s",
            result.get("predicted_eta_minutes"),
            result.get("late_risk_probability"),
            result.get("is_fallback")
        )

        return result

    except Exception as e:
        logger.exception("Unified prediction failed: %s", e)
        raise


# ============================================================
# DISTANCE CALCULATION
# ============================================================

def calculate_haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """Calculate great-circle distance between two GPS coordinates."""

    earth_radius_km = 6371.0

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)

    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1)
        * np.cos(phi2)
        * np.sin(delta_lambda / 2.0) ** 2
    )

    c = 2.0 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1.0 - a)
    )

    return round(float(earth_radius_km * c), 2)


# ============================================================
# LEGACY ETA INFERENCE
# Retained for compatibility with the existing ETA page.
# ============================================================

def execute_eta_inference(
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute ETA prediction using the existing standalone
    regressor implementation.

    Retained so the current ETA page continues to work.
    """

    engineer = FeatureEngineer()
    optimizer = DeliveryOptimizer()

    df_single = pd.DataFrame([payload])

    df_single["order_date"] = pd.to_datetime(
        df_single.get("order_date", "2022-04-01")
    )

    lat1 = float(payload.get("restaurant_latitude", 0))
    lon1 = float(payload.get("restaurant_longitude", 0))

    lat2 = float(
        payload.get("delivery_location_latitude", 0)
    )
    lon2 = float(
        payload.get("delivery_location_longitude", 0)
    )

    distance_input = payload.get("distance_km")

    if distance_input is None or float(distance_input) <= 0:
        dist_km = calculate_haversine_km(
            lat1, lon1, lat2, lon2
        )

        if dist_km <= 0 or np.isnan(dist_km):
            dist_km = 4.5

    else:
        dist_km = float(distance_input)

    df_single["distance_km"] = dist_km

    order_hour = int(payload.get("order_hour", 18))

    df_single["order_hour"] = order_hour

    df_single["order_day_of_week"] = int(
        payload.get("order_day_of_week", 4)
    )

    df_single["is_weekend"] = int(
        payload.get("is_weekend", 0)
    )

    df_single["is_rush_hour"] = int(
        order_hour in [12, 13, 14, 18, 19, 20, 21]
    )

    traffic_density = str(
        payload.get("road_traffic_density", "Medium")
    ).strip()

    traffic_map = {
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Jam": 4
    }

    df_single["traffic_density_ordinal"] = traffic_map.get(
        traffic_density,
        2
    )

    reg_bundle = load_regressor_bundle()

    if reg_bundle is not None:

        try:
            X, _ = engineer.get_feature_matrices(
                df_single,
                task="regression"
            )

            X_proc = reg_bundle["preprocessor"].transform(X)

            pred_eta = float(
                reg_bundle["model"].predict(X_proc)[0]
            )

            pred_eta = max(5.0, round(pred_eta, 1))

            model_name = "XGBoost Regressor (Trained ML Pipeline)"
            is_fallback = False

        except Exception as e:
            logger.warning(
                "Regressor inference failed, falling back: %s",
                e
            )

            pred_eta = _fallback_heuristic(
                df_single.iloc[0]
            )

            model_name = "Domain Physics Heuristic (Fallback)"
            is_fallback = True

    else:
        pred_eta = _fallback_heuristic(
            df_single.iloc[0]
        )

        model_name = "Domain Physics Heuristic (Fallback)"
        is_fallback = True

    weather = str(payload.get("weather", "Sunny"))

    dynamic_sla = optimizer.compute_dynamic_eta(
        pred_eta,
        weather,
        traffic_density
    )

    fleet_rec = optimizer.recommend_fleet_dispatch(
        dist_km,
        traffic_density,
        weather
    )

    return {
        "predicted_eta_min": pred_eta,
        "distance_km": round(dist_km, 2),
        "model_name": model_name,
        "is_fallback": is_fallback,
        "dynamic_sla": dynamic_sla,
        "fleet_recommendation": fleet_rec
    }


# ============================================================
# LEGACY LATE-RISK INFERENCE
# Retained for compatibility with the existing late-risk page.
# ============================================================

def execute_late_risk_inference(
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute late-order risk prediction using the existing
    standalone classifier implementation.

    Retained so the current late-risk page continues to work.
    """

    engineer = FeatureEngineer()
    optimizer = DeliveryOptimizer()

    df_single = pd.DataFrame([payload])

    df_single["order_date"] = pd.to_datetime(
        df_single.get("order_date", "2022-04-01")
    )

    lat1 = float(payload.get("restaurant_latitude", 0))
    lon1 = float(payload.get("restaurant_longitude", 0))

    lat2 = float(
        payload.get("delivery_location_latitude", 0)
    )
    lon2 = float(
        payload.get("delivery_location_longitude", 0)
    )

    distance_input = payload.get("distance_km")

    if distance_input is None or float(distance_input) <= 0:
        dist_km = calculate_haversine_km(
            lat1, lon1, lat2, lon2
        )

        if dist_km <= 0 or np.isnan(dist_km):
            dist_km = 4.5

    else:
        dist_km = float(distance_input)

    df_single["distance_km"] = dist_km

    order_hour = int(payload.get("order_hour", 18))

    df_single["order_hour"] = order_hour

    df_single["order_day_of_week"] = int(
        payload.get("order_day_of_week", 4)
    )

    df_single["is_weekend"] = int(
        payload.get("is_weekend", 0)
    )

    df_single["is_rush_hour"] = int(
        order_hour in [12, 13, 14, 18, 19, 20, 21]
    )

    traffic_density = str(
        payload.get("road_traffic_density", "Medium")
    ).strip()

    traffic_map = {
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Jam": 4
    }

    df_single["traffic_density_ordinal"] = traffic_map.get(
        traffic_density,
        2
    )

    clf_bundle = load_classifier_bundle()

    threshold = (
        clf_bundle.get("threshold", 0.59)
        if clf_bundle
        else 0.59
    )

    if clf_bundle is not None:

        try:
            X, _ = engineer.get_feature_matrices(
                df_single,
                task="classification"
            )

            X_proc = clf_bundle["preprocessor"].transform(X)

            probs = clf_bundle["model"].predict_proba(
                X_proc
            )[0]

            classes = getattr(
                clf_bundle["model"],
                "classes_",
                [0, 1]
            )

            positive_index = list(classes).index(1)

            late_prob = float(
                probs[positive_index]
            )

            is_late_pred = int(
                late_prob >= threshold
            )

            model_name = (
                "XGBoost Classifier (Trained ML Pipeline)"
            )

            is_fallback = False

        except Exception as e:
            logger.warning(
                "Classifier inference failed: %s",
                e
            )

            late_prob = 0.50
            is_late_pred = 0

            model_name = "Heuristic Fallback"
            is_fallback = True

    else:
        late_prob = 0.50
        is_late_pred = 0

        model_name = "Heuristic Fallback"
        is_fallback = True

    if late_prob < 0.30:
        risk_category = "LOW"

    elif late_prob < threshold:
        risk_category = "MODERATE"

    elif late_prob < 0.80:
        risk_category = "HIGH"

    else:
        risk_category = "CRITICAL"

    mitigation = optimizer.generate_risk_mitigation_plan(
        late_risk_probability=late_prob,
        predicted_eta_min=32.0 if is_late_pred else 22.0,
        prep_time_min=14.0
    )

    return {
        "late_probability": round(late_prob, 4),
        "late_probability_pct": round(
            late_prob * 100,
            1
        ),
        "threshold_used": threshold,
        "is_late_decision": is_late_pred,
        "risk_category": risk_category,
        "model_name": model_name,
        "is_fallback": is_fallback,
        "mitigation_plan": mitigation
    }


# ============================================================
# FALLBACK ETA HEURISTIC
# ============================================================

def _fallback_heuristic(row: pd.Series) -> float:
    """Physics-informed speed and preparation-time fallback."""

    dist = float(
        row.get("distance_km", 4.0)
    )

    traffic = str(
        row.get("road_traffic_density", "Medium")
    )

    weather = str(
        row.get("weather", "Sunny")
    )

    speed_map = {
        "Low": 28.0,
        "Medium": 22.0,
        "High": 16.0,
        "Jam": 10.0
    }

    speed = speed_map.get(
        traffic,
        20.0
    )

    if weather in ["Stormy", "Sandstorms", "Fog"]:
        speed *= 0.75

    transit_time = (
        dist / max(speed, 5.0)
    ) * 60.0

    return round(
        12.0 + transit_time,
        1
    )
