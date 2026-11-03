# Quick-Commerce Delivery Intelligence ⚡

### AI-Powered ETA Prediction, Late-Delivery Risk Detection & Delivery Optimization

## Overview

Quick-Commerce Delivery Intelligence is an end-to-end data analytics and machine learning project designed to help quick-commerce operations understand delivery performance, estimate delivery times, identify potentially late orders, and support better delivery planning.

The project combines data cleaning, exploratory data analysis (EDA), feature engineering, supervised machine learning, unsupervised clustering, model evaluation, and an interactive Streamlit dashboard.

## Problem Statement

Quick-commerce platforms operate with tight delivery expectations. Delivery times can be affected by distance, traffic, weather, order timing, and delivery-partner characteristics. Delays can reduce customer satisfaction and create operational inefficiencies.

This project explores how historical delivery data can be used to:

- Predict estimated delivery time.
- Identify orders at risk of exceeding a delivery-time threshold.
- Discover geographic demand clusters that can inform dark-store planning.
- Segment delivery partners based on operational performance.
- Present insights through an interactive dashboard.

## Key Features

### 1. Exploratory Data Analysis

- Dataset profiling and data-quality checks.
- Analysis of delivery-time distributions.
- Study of delivery patterns across traffic, weather, distance, and other variables.
- Correlation analysis and visual reporting.

### 2. ETA Prediction — Regression

A regression workflow estimates delivery duration in minutes using relevant order, location, environmental, and delivery-partner features.

The model comparison report records the selected XGBoost Regressor results:

| Metric |        Result |
| ------ | ------------: |
| MAE    | 3.317 minutes |
| RMSE   | 4.150 minutes |
| R²    |        0.8013 |

### 3. Late-Delivery Risk — Classification

A classification workflow estimates whether an order may exceed the project's configurable 30-minute demonstration threshold.

The model comparison report records the selected XGBoost Classifier results:

| Metric   | Result |
| -------- | -----: |
| F1-score | 0.8709 |
| Recall   | 88.32% |
| ROC-AUC  | 0.9814 |
| PR-AUC   | 0.9614 |

The 30-minute threshold is an analytical benchmark used in this project, not a claim about a universal or official delivery SLA.

### 4. Geographic Demand-Zone Clustering

K-Means clustering is used to group valid delivery coordinates into geographic clusters. These clusters can support exploratory demand-zone and dark-store planning analysis.

### 5. Delivery-Partner Segmentation

Clustering is used to explore delivery-partner performance groups and identify patterns that may support operational coaching and allocation decisions.

### 6. Interactive Streamlit Dashboard

The Streamlit application brings together:

- Executive overview and business KPIs
- Data exploration
- ETA prediction
- Late-delivery risk analysis
- Geographic demand zones
- Delivery-partner segmentation
- Model performance and explainability
- Business recommendations

## Technology Stack

| Area                     | Tools                         |
| ------------------------ | ----------------------------- |
| Language                 | Python                        |
| Data processing          | Pandas, NumPy                 |
| Visualization            | Matplotlib, Seaborn, Plotly   |
| Machine learning         | Scikit-learn, XGBoost         |
| Explainability           | SHAP                          |
| Dashboard                | Streamlit                     |
| Geographic visualization | Folium                        |
| Model persistence        | Joblib                        |
| Development              | Jupyter Notebook, Git, GitHub |

## Project Structure

```text
Quick-Commerce-Delivery-Intelligence/
│
├── app/
│   ├── __init__.py
│   ├── streamlit_app.py
│   ├── data_service.py
│   ├── styles.py
│   └── views/
│
├── data/
│   ├── train.xlsx
│   ├── cleaned_delivery_data.csv
│   └── featured_delivery_data.csv
│
├── dataset/
├── ml/
├── models/
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── reports/
│   ├── figures/
│   ├── model_comparison.csv
│   ├── eda_findings.md
│   ├── regression_report.md
│   ├── classification_report.md
│   ├── clustering_report.md
│   └── model_selection.md
│
├── scripts/
│   ├── generate_evaluation_curves.py
│   ├── prepublication_file_audit.py
│   ├── train_and_evaluate.py
│   └── validate_streamlit_app.py
│
├── src/
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── regression.py
│   ├── classification.py
│   ├── clustering.py
│   ├── model_evaluation.py
│   └── prediction.py
│
├── tests/
├── powerbi/
├── PPT/
├── report/
│
├── .streamlit/
├── config.py
├── main.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

*Note: The exact files available may vary by project version. Large generated artifacts, local virtual environments, and temporary outputs should not be committed to the repository.*

## Getting Started

### Prerequisites

- Python 3.10 or later
- Git
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/PradeepthiPathuri/11_Quick-Commerce-Delivery-Intelligence-Omni-Tech.git

cd 11_Quick-Commerce-Delivery-Intelligence-Omni-Tech
```

### 2. Create and Activate a Virtual Environment

**Windows:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Place the source dataset at the path expected by the project configuration:

```text
data/train.xlsx
```

The project also uses generated cleaned and feature-engineered files under `data/`.

Check `config.py` if your dataset is stored under a different filename or location.

### 5. Run the Data Workflows

Run commands from the repository root:

```bash
python main.py --inspect
python main.py --clean
python main.py --features
python main.py --eda
```

### 6. Launch the Streamlit Dashboard

Run the following command from the repository root:

```bash
streamlit run app/streamlit_app.py
```

Open the local URL displayed in your terminal:

```text
http://localhost:8501
```

If your project version exposes the dashboard through the CLI entry point, you can also try:

```bash
python main.py --app
```

## Model Training and Evaluation

The repository includes model-training and evaluation workflows under `scripts/` and model implementation modules under `src/`.

Depending on the project version and available dataset/model artifacts, run:

```bash
python scripts/train_and_evaluate.py
```

Review the generated evaluation outputs in `reports/`.

Model metrics can vary when the dataset, split strategy, features, random seed, or training configuration changes.

## Important Notes on Data Leakage

For a realistic prediction scenario, only use information available at the intended prediction time.

The project configuration identifies post-dispatch or outcome-related fields that must be excluded from early prediction features, such as:

- Actual delivery duration and late-delivery target.
- Order-picked timestamps and preparation-time values.
- Other fields that are only known after dispatch or during delivery.

Always review `config.py` and the training pipeline before changing the feature set.

## Reports and Outputs

The `reports/` directory contains analytical findings, model comparisons, evaluation reports, and generated visualizations.

These files document the current experiment results and provide context for the dashboard.

## Limitations

- Results depend on the quality, coverage, and representativeness of the available historical dataset.
- Model metrics are offline evaluation results and do not guarantee production performance.
- Geographic clusters are analytical groupings; they are not automatically optimized store locations.
- The 30-minute late-delivery threshold is configurable and should be adapted to the business context.
- A production deployment would require live traffic/weather feeds, monitoring, retraining, security controls, and integration with order-dispatch systems.

## Future Enhancements

- Integrate live traffic and weather information.
- Connect predictions to a real-time order-management or dispatch system.
- Add model monitoring, drift detection, and scheduled retraining.
- Improve geographic optimization using demand, capacity, and service constraints.
- Deploy the dashboard and inference service to a cloud platform.

## Contributors

Developed as a collaborative project. Add the names and contributions of all team members here.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
