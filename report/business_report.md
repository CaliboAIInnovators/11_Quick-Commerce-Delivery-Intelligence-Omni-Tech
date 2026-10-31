# Quick-Commerce Delivery Intelligence — Business Report

**Hackathon:** KLU + PATH3 · ML-01  
**Project:** Quick-Commerce Delivery Intelligence System

> This is the editable source version of the business report. Update the text and figures after checking the latest notebook outputs and Power BI exports.

## Executive Summary

The project develops a delivery-intelligence workflow around four business questions: how long an order may take, whether it is likely to miss the configured service threshold, where demand and delivery delays cluster geographically, and which rider groups may need different operational support.

The Power BI overview shared during the project work displayed approximately **46K deliveries**, **26.29 minutes average delivery time**, approximately **14K late orders**, a **0.30 (30%) late-order rate**, and an **average rider rating of 4.64**. These are rounded dashboard values and should be checked against the final cleaned-data export before submission.

The workflow combines exploratory data analysis, regression, classification, K-Means geographic clustering, rider segmentation, and business recommendations.

## 1. Business Problem and Scope

Delivery platforms must balance speed, reliability, rider capacity and operating cost. The project aims to turn order, location, traffic, weather and rider-related data into operational insights.

Project outputs:
- Delivery-time estimation using regression.
- Late-order risk scoring using classification.
- Geographic demand clustering to identify candidate service zones.
- Rider performance segmentation.
- Three prioritized management recommendations.

## 2. Data Preparation

The project notebook includes a cleaning and preparation workflow intended to:
- Convert delivery-time text into numeric minutes.
- Normalize category labels and data types.
- Inspect missing values and duplicate records.
- Validate geographic coordinates and calculate distance.
- Document imputation decisions.
- Exclude IDs, target leakage and information unavailable at order-placement time.

**Leakage control:** pickup time should not be used as an input for a prediction made at order placement because it is only known after the order is placed. Preprocessing should be fitted within the training pipeline, not on the full dataset.

## 3. Exploratory Data Analysis

The planned EDA includes:
- Delivery-time distribution.
- Delivery time by weather and traffic.
- Delivery time by order hour and distance.
- Late-order rate by weather and traffic.
- Rider rating versus delivery time.
- Delivery time by vehicle type and city type.

Interpretation should describe observed associations and should not claim that correlation proves causation.

## 4. Machine-Learning Approach

### 4.1 Delivery-Time Regression

The notebook compares a mean baseline, Ridge/Linear Regression, Random Forest, Gradient Boosting and XGBoost when available. Evaluation metrics are:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score

**Selected model:** update from the latest `reports/model_comparison_regression.csv` output.  
**Test MAE:** update  
**Test RMSE:** update  
**Test R²:** update

The model-selection workflow chooses the model with the lowest held-out test MAE.

### 4.2 Late-Order Classification

The classifier predicts whether delivery time exceeds a configurable SLA threshold. Candidate models include Logistic Regression, Random Forest, Gradient Boosting and optional XGBoost.

Evaluation metrics:
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix and ROC curve

**Important:** the challenge brief does not provide a fixed numerical SLA. State the threshold used in the final run and treat it as a project assumption, not an official benchmark.

**Selected model:** update from the latest `reports/model_comparison_classification.csv`.  
**Test ROC-AUC:** update  
**Test F1-score:** update  
**Test Recall:** update

## 5. Geographic Demand and Dark-Store Intelligence

The project applies K-Means clustering to valid geographic coordinates. The intended cluster summary includes order volume, average delivery time, late-order rate and demand concentration.

The recommendation-generation script identifies Zone 1 and Zone 6 as candidate bottleneck zones, with the following code-reported values:

| Candidate zone | Orders | Late rate | Average delivery time |
|---|---:|---:|---:|
| Zone 1 | 5,848 | 31.65% | 26.57 min |
| Zone 6 | 6,326 | 29.97% | Not stated in retrieved recommendation text |

The script also reports that these zones together represent 29.0% of valid deliveries and 34.2% of network SLA delays, with an average order distance of 10.4 km.

**Validation required:** compare all these figures with the latest `powerbi/zone_clusters.csv` before using them as final claims. The underlying food-delivery dataset uses restaurant/customer coordinates as proxies; they are not confirmed quick-commerce dark-store locations. Treat proposed sites as candidates for feasibility assessment only.

## 6. Rider Performance Segmentation

Riders are grouped using delivery volume, average delivery time, average rating, late-order rate and vehicle condition where available.

The recommendation script reports this provisional comparison:

| Segment | Riders | Late rate | Avg. time | Rating | Vehicle condition |
|---|---:|---:|---:|---:|---:|
| Tier 1 — Elite | 349 | 20.65% | 24.61 min | 4.69 | 1.8 |
| Tier 3 — At-risk | 335 | 39.68% | 28.19 min | 4.60 | 0.8 |

The script reports Tier 3 as 25.38% of the rider fleet. Validate these figures against `powerbi/rider_segments.csv` before final submission.

## 7. Business Recommendations

### Recommendation 1 — Dynamic Traffic-Aware ETA and Peak Rider Pre-Allocation

**Priority:** P1 — Critical  
**Domain:** Delivery operations  
**Timeline:** Weeks 1–4

**Finding and evidence:** The recommendation script reports average delivery time of 31.14 minutes and a 51.4% late rate under Jam traffic, compared with 21.41 minutes and an 11.2% late rate under Low traffic. It also reports traffic density and weather as contributing over 25% of tree-model predictive importance.

**Recommended action:** Use the trained ETA model to adjust customer-facing estimates during severe traffic and weather. Pilot pre-allocation of additional floating riders before evening peaks. Measure late rate, ETA error and rider utilization against a comparable control period. Any proposed extra rider allocation is a pilot hypothesis, not a proven improvement.

### Recommendation 2 — Validate Zone 1 and Zone 6 as Micro-Fulfilment Candidates

**Priority:** P2 — High  
**Domain:** Zone planning  
**Timeline:** Months 2–4

**Finding and evidence:** The recommendation script reports 5,848 orders and a 31.65% late rate for Zone 1, and 6,326 orders with a 29.97% late rate for Zone 6.

**Recommended action:** Conduct a feasibility assessment for micro-hubs or smaller fulfilment points in these zones. Validate actual demand, road access, property cost and replenishment requirements before selecting a site. Do not treat cluster centroids as verified property locations.

### Recommendation 3 — Tiered Rider Coaching, Equipment Checks and Dispatch Support

**Priority:** P3 — Medium  
**Domain:** Rider management  
**Timeline:** Weeks 2–8

**Finding and evidence:** The recommendation script reports a 39.68% late rate and 28.19-minute average delivery time for Tier 3, compared with 20.65% and 24.61 minutes for Tier 1.

**Recommended action:** Offer navigation coaching, vehicle-condition checks and maintenance support to at-risk riders. Test dispatch rules that reserve complex or high-priority orders for riders with demonstrated reliability, while monitoring fairness and rider workload.

## 8. Limitations and Assumptions

- The challenge brief does not prescribe a fixed SLA threshold.
- The dataset is food-delivery data; coordinates are proxies, not verified quick-commerce dark-store locations.
- Recommendation-script figures must be checked against the latest generated CSV exports.
- Do not add model scores until the final model-comparison CSVs are available.
- Observational associations do not establish causation.
- Operational improvements are hypotheses until tested in a controlled pilot.

## 9. Conclusion

The Quick-Commerce Delivery Intelligence project provides a practical framework for monitoring delivery performance, estimating ETA, identifying late-order risk, prioritizing geographic bottlenecks and supporting rider development. The next step is to validate all provisional figures against the latest exports and insert the final held-out model metrics before judging or business use.

## Final Validation Checklist

- [ ] Confirm final row count and KPI totals in `powerbi/cleaned_data.csv`.
- [ ] Add final regression metrics from `reports/model_comparison_regression.csv`.
- [ ] Add final classification metrics from `reports/model_comparison_classification.csv`.
- [ ] Validate geographic figures against `powerbi/zone_clusters.csv`.
- [ ] Validate rider figures against `powerbi/rider_segments.csv`.
- [ ] Validate the three recommendations against `powerbi/business_findings.csv`.
- [ ] Confirm the SLA threshold used in the final classifier.
