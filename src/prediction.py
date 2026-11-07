
"""
Unified Inference Pipeline for Real-Time and Batch Delivery Intelligence.

Supports:
1. Trained ETA Regression Model
2. Trained Late-Order Classification Model
3. Calibrated Classification Threshold
4. Single Order Prediction
5. Batch Order Prediction
6. Physics-Informed ETA Fallback
7. Operational Dispatch Recommendations

Model artifacts:
    models/best_eta_regressor.joblib
    models/best_late_classifier.joblib
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

import joblib
import numpy as np
import pandas as pd

import config
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.recommendations import DeliveryOptimizer


logger = logging.getLogger(__name__)


class DeliveryInferencePipeline:
    """Inference service for real-time and batch delivery predictions."""

    def __init__(
        self,
        regressor_path: Optional[Path] = None,
        classifier_path: Optional[Path] = None
    ):
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.optimizer = DeliveryOptimizer()

        self.regressor = None
        self.classifier = None

        self.classifier_threshold = 0.50

        reg_candidates = [
            regressor_path,
            config.MODELS_DIR / "best_eta_regressor.joblib",
            config.MODELS_DIR / "eta_regressor_xgboost.joblib"
        ]

        clf_candidates = [
            classifier_path,
            config.MODELS_DIR / "best_late_classifier.joblib",
            config.MODELS_DIR / "late_classifier_xgboost.joblib"
        ]

        reg_file = next(
            (Path(p) for p in reg_candidates if p and Path(p).exists()),
            None
        )

        clf_file = next(
            (Path(p) for p in clf_candidates if p and Path(p).exists()),
            None
        )

        self._load_models_if_available(reg_file, clf_file)

    # ---------------------------------------------------------
    # MODEL LOADING
    # ---------------------------------------------------------

    def _load_models_if_available(
        self,
        reg_file: Optional[Path],
        clf_file: Optional[Path]
    ):
        """Load trained model artifacts."""

        if reg_file is not None:
            try:
                self.regressor = joblib.load(reg_file)

                logger.info(
                    "Loaded ETA model from %s",
                    reg_file
                )

            except Exception as e:
                logger.exception("ETA model loading failed: %s", e)

        if clf_file is not None:
            try:
                self.classifier = joblib.load(clf_file)

                if isinstance(self.classifier, dict):
                    self.classifier_threshold = float(
                        self.classifier.get("threshold", 0.50)
                    )

                logger.info(
                    "Loaded late-order classifier from %s",
                    clf_file
                )

                logger.info(
                    "Classification threshold: %.3f",
                    self.classifier_threshold
                )

            except Exception as e:
                logger.exception(
                    "Classifier loading failed: %s",
                    e
                )

    # ---------------------------------------------------------
    # MODEL STATUS
    # ---------------------------------------------------------

    def get_model_status(self) -> Dict[str, Any]:
        """Return the current status of both prediction models."""

        return {
            "eta_model_loaded": self.regressor is not None,
            "classifier_model_loaded": self.classifier is not None,
            "classifier_threshold": self.classifier_threshold,
            "eta_model_type": self._get_model_type(self.regressor),
            "classifier_model_type": self._get_model_type(
                self.classifier
            )
        }

    @staticmethod
    def _get_model_type(model) -> str:
        if model is None:
            return "Not Loaded"

        if isinstance(model, dict):
            if "model" in model:
                return type(model["model"]).__name__

            if "pipeline" in model:
                return type(model["pipeline"]).__name__

        return type(model).__name__

    # ---------------------------------------------------------
    # MODEL INPUT PREPARATION
    # ---------------------------------------------------------

    def _prepare_model_input(
        self,
        df: pd.DataFrame,
        task: str
    ):
        """Prepare feature matrices using the existing feature engineer."""

        X, _ = self.engineer.get_feature_matrices(
            df,
            task=task
        )

        return X

    # ---------------------------------------------------------
    # ETA MODEL PREDICTION
    # ---------------------------------------------------------

    def _predict_eta_ml(self, X: pd.DataFrame) -> np.ndarray:
        """Execute prediction using the trained ETA model."""

        if self.regressor is None:
            raise RuntimeError("ETA regression model is unavailable.")

        if isinstance(self.regressor, dict):

            if "model" in self.regressor:
                model = self.regressor["model"]

                preprocessor = self.regressor.get("preprocessor")

                if preprocessor is not None:
                    X = preprocessor.transform(X)

                predictions = model.predict(X)

            elif "pipeline" in self.regressor:
                predictions = self.regressor["pipeline"].predict(X)

            else:
                raise ValueError(
                    "Unsupported ETA model dictionary structure."
                )

        else:
            predictions = self.regressor.predict(X)

        predictions = np.asarray(predictions, dtype=float).reshape(-1)

        if not np.all(np.isfinite(predictions)):
            raise ValueError("ETA model returned invalid predictions.")

        return np.maximum(predictions, 5.0)

    # ---------------------------------------------------------
    # LATE CLASSIFIER PREDICTION
    # ---------------------------------------------------------

    def _predict_late_probability(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """Execute probability prediction using the trained classifier."""

        if self.classifier is None:
            raise RuntimeError("Late-order classifier is unavailable.")

        if isinstance(self.classifier, dict):

            if "model" in self.classifier:
                model = self.classifier["model"]

                preprocessor = self.classifier.get("preprocessor")

                if preprocessor is not None:
                    X = preprocessor.transform(X)

            elif "pipeline" in self.classifier:
                model = self.classifier["pipeline"]

            else:
                raise ValueError(
                    "Unsupported classifier dictionary structure."
                )

        else:
            model = self.classifier

        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                "Classifier does not support predict_proba()."
            )

        probabilities = model.predict_proba(X)

        classes = getattr(model, "classes_", None)

        if classes is not None:
            positive_indices = np.where(np.asarray(classes) == 1)[0]

            if len(positive_indices) == 0:
                raise ValueError(
                    "Positive class (1) not found in classifier."
                )

            positive_index = int(positive_indices[0])

        else:
            positive_index = 1

        if probabilities.shape[1] <= positive_index:
            raise ValueError("Invalid classifier probability output.")

        late_probabilities = np.asarray(
            probabilities[:, positive_index],
            dtype=float
        )

        if not np.all(np.isfinite(late_probabilities)):
            raise ValueError(
                "Classifier returned invalid probabilities."
            )

        return np.clip(late_probabilities, 0.0, 1.0)

    # ---------------------------------------------------------
    # ETA FALLBACK
    # ---------------------------------------------------------

    def _heuristic_eta_fallback(self, row: pd.Series) -> float:
        """Physics-informed fallback ETA calculation."""

        distance = float(row.get("distance_km", 4.0))

        traffic = str(
            row.get("road_traffic_density", "Medium")
        ).strip()

        weather = str(
            row.get("weather", "Sunny")
        ).strip()

        order_type = str(
            row.get("type_of_order", "Meal")
        ).strip()

        prep_estimates = {
            "Snack": 10.0,
            "Drinks": 6.0,
            "Meal": 14.0,
            "Buffet": 20.0
        }

        prep_time = prep_estimates.get(order_type, 12.0)

        speed_map = {
            "Low": 28.0,
            "Medium": 22.0,
            "High": 16.0,
            "Jam": 10.0
        }

        speed = speed_map.get(traffic, 20.0)

        if weather in ["Stormy", "Sandstorms", "Fog"]:
            speed *= 0.75

        elif weather in ["Windy", "Cloudy"]:
            speed *= 0.90

        transit_time = (
            distance / max(speed, 5.0)
        ) * 60.0

        total_eta = prep_time + transit_time

        return round(float(total_eta), 1)

    # ---------------------------------------------------------
    # CLASSIFICATION FALLBACK
    # ---------------------------------------------------------

    @staticmethod
    def _heuristic_late_probability(
        eta: float,
        sla: float
    ) -> float:
        """Estimate late risk when classifier is unavailable."""

        difference = eta - sla

        probability = 1.0 / (
            1.0 + np.exp(-difference / 5.0)
        )

        return float(np.clip(probability, 0.0, 1.0))

    # ---------------------------------------------------------
    # SINGLE ORDER PREDICTION
    # ---------------------------------------------------------

    def predict_single(
        self,
        order_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate ETA, late-risk and operational recommendations
        for one delivery order.
        """

        if not isinstance(order_data, dict) or not order_data:
            raise ValueError("order_data must be a non-empty dictionary.")

        df_single = pd.DataFrame([order_data])

        # Step 1: Data cleaning
        df_cleaned = self.cleaner.clean(
            df_single,
            is_training=False
        )

        # Step 2: Feature engineering
        df_featured = self.engineer.transform_features(
            df_cleaned,
            is_training=False
        )

        row = df_featured.iloc[0]

        # Step 3: ETA regression
        eta_fallback = False

        if self.regressor is not None:
            try:
                X_eta = self._prepare_model_input(
                    df_featured,
                    task="regression"
                )

                predicted_eta = float(
                    self._predict_eta_ml(X_eta)[0]
                )

                eta_model_type = "Trained ML Model"

            except Exception as e:
                logger.exception("ETA inference failed: %s", e)

                predicted_eta = self._heuristic_eta_fallback(row)

                eta_model_type = "Baseline Domain Heuristic"

                eta_fallback = True

        else:
            predicted_eta = self._heuristic_eta_fallback(row)

            eta_model_type = "Baseline Domain Heuristic"

            eta_fallback = True

        # Step 4: Late-order classification
        classifier_fallback = False

        if self.classifier is not None:
            try:
                X_clf = self._prepare_model_input(
                    df_featured,
                    task="classification"
                )

                late_prob = float(
                    self._predict_late_probability(X_clf)[0]
                )

                classifier_model_type = "Trained ML Model"

            except Exception as e:
                logger.exception(
                    "Classifier inference failed: %s",
                    e
                )

                late_prob = self._heuristic_late_probability(
                    predicted_eta,
                    config.DEFAULT_SLA_MINUTES
                )

                classifier_model_type = "Baseline Domain Heuristic"

                classifier_fallback = True

        else:
            late_prob = self._heuristic_late_probability(
                predicted_eta,
                config.DEFAULT_SLA_MINUTES
            )

            classifier_model_type = "Baseline Domain Heuristic"

            classifier_fallback = True

        # Step 5: Calibrated classification decision
        is_late_predicted = int(
            late_prob >= self.classifier_threshold
        )

        # Step 6: Risk categorization
        if late_prob < 0.30:
            risk_category = "LOW"

        elif late_prob < self.classifier_threshold:
            risk_category = "MODERATE"

        elif late_prob < 0.80:
            risk_category = "HIGH"

        else:
            risk_category = "CRITICAL"

        # Step 7: Operational recommendations
        weather = str(
            row.get("weather", "Sunny")
        ).strip()

        traffic = str(
            row.get("road_traffic_density", "Medium")
        ).strip()

        distance = float(
            row.get("distance_km", 3.5)
        )

        order_type = str(
            row.get("type_of_order", "Meal")
        ).strip()

        prep_estimates = {
            "Snack": 10.0,
            "Drinks": 6.0,
            "Meal": 14.0,
            "Buffet": 20.0
        }

        prep_time = float(
            row.get(
                "prep_time_min",
                prep_estimates.get(order_type, 12.0)
            )
        )

        if pd.isna(prep_time):
            prep_time = prep_estimates.get(order_type, 12.0)

        dynamic_sla = self.optimizer.compute_dynamic_eta(
            predicted_eta,
            weather,
            traffic
        )

        fleet_recommendation = (
            self.optimizer.recommend_fleet_dispatch(
                distance,
                traffic,
                weather
            )
        )

        mitigation_plan = (
            self.optimizer.generate_risk_mitigation_plan(
                late_risk_probability=late_prob,
                predicted_eta_min=predicted_eta,
                prep_time_min=prep_time
            )
        )

        # Step 8: Final prediction response
        return {
            "predicted_eta_minutes": round(predicted_eta, 1),

            "late_risk_probability": round(late_prob, 4),

            "late_risk_percentage": round(
                late_prob * 100, 1
            ),

            "is_late_predicted": is_late_predicted,

            "classifier_threshold": self.classifier_threshold,

            "risk_category": risk_category,

            "eta_model_type": eta_model_type,

            "classifier_model_type": classifier_model_type,

            "eta_is_fallback": eta_fallback,

            "classifier_is_fallback": classifier_fallback,

            "is_fallback": (
                eta_fallback or classifier_fallback
            ),

            "distance_km": round(distance, 2),

            "estimated_prep_time_min": round(
                prep_time, 1
            ),

            "dynamic_sla": dynamic_sla,

            "fleet_recommendation": fleet_recommendation,

            "mitigation_plan": mitigation_plan
        }

    # ---------------------------------------------------------
    # BATCH PREDICTION
    # ---------------------------------------------------------

    def predict_batch(
        self,
        df_raw: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate predictions for multiple delivery orders.

        Returns original engineered features with:
            predicted_eta_min
            late_risk_probability
            is_late_predicted
            eta_model_type
            classifier_model_type
        """

        if not isinstance(df_raw, pd.DataFrame):
            raise TypeError("df_raw must be a pandas DataFrame.")

        if df_raw.empty:
            return df_raw.copy()

        # Step 1: Cleaning
        df_cleaned = self.cleaner.clean(
            df_raw.copy(),
            is_training=False
        )

        # Step 2: Feature engineering
        df_featured = self.engineer.transform_features(
            df_cleaned,
            is_training=False
        )

        # Step 3: ETA prediction
        eta_fallback = False

        if self.regressor is not None:
            try:
                X_eta = self._prepare_model_input(
                    df_featured,
                    task="regression"
                )

                eta_predictions = self._predict_eta_ml(X_eta)

                eta_model_type = "Trained ML Model"

            except Exception as e:
                logger.exception(
                    "Batch ETA prediction failed: %s",
                    e
                )

                eta_predictions = np.array([
                    self._heuristic_eta_fallback(
                        df_featured.iloc[i]
                    )
                    for i in range(len(df_featured))
                ])

                eta_model_type = "Baseline Domain Heuristic"

                eta_fallback = True

        else:
            eta_predictions = np.array([
                self._heuristic_eta_fallback(
                    df_featured.iloc[i]
                )
                for i in range(len(df_featured))
            ])

            eta_model_type = "Baseline Domain Heuristic"

            eta_fallback = True

        # Step 4: Late-order probability
        classifier_fallback = False

        if self.classifier is not None:
            try:
                X_clf = self._prepare_model_input(
                    df_featured,
                    task="classification"
                )

                late_probabilities = (
                    self._predict_late_probability(X_clf)
                )

                classifier_model_type = "Trained ML Model"

            except Exception as e:
                logger.exception(
                    "Batch classification failed: %s",
                    e
                )

                late_probabilities = np.array([
                    self._heuristic_late_probability(
                        eta,
                        config.DEFAULT_SLA_MINUTES
                    )
                    for eta in eta_predictions
                ])

                classifier_model_type = "Baseline Domain Heuristic"

                classifier_fallback = True

        else:
            late_probabilities = np.array([
                self._heuristic_late_probability(
                    eta,
                    config.DEFAULT_SLA_MINUTES
                )
                for eta in eta_predictions
            ])

            classifier_model_type = "Baseline Domain Heuristic"

            classifier_fallback = True

        # Step 5: Apply calibrated threshold
        late_decisions = (
            late_probabilities >= self.classifier_threshold
        ).astype(int)

        # Step 6: Add results
        result_df = df_featured.reset_index(drop=True).copy()

        result_df["predicted_eta_min"] = np.round(
            eta_predictions, 1
        )

        result_df["late_risk_probability"] = np.round(
            late_probabilities, 4
        )

        result_df["late_risk_percentage"] = np.round(
            late_probabilities * 100, 1
        )

        result_df["is_late_predicted"] = late_decisions

        result_df["eta_model_type"] = eta_model_type

        result_df["classifier_model_type"] = (
            classifier_model_type
        )

        result_df["eta_is_fallback"] = eta_fallback

        result_df["classifier_is_fallback"] = (
            classifier_fallback
        )

        result_df["is_fallback"] = (
            eta_fallback or classifier_fallback
        )

        result_df["classifier_threshold"] = (
            self.classifier_threshold
        )

        return result_df
