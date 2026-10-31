# Update Guide

## Files to edit
- `report/business_report_editable.docx` — easiest to edit in Microsoft Word.
- `report/business_report.md` — plain-text Markdown source.

## Numbers that must be checked
- Dashboard KPI values: verify against `powerbi/cleaned_data.csv`.
- Regression scores: `reports/model_comparison_regression.csv`.
- Classification scores: `reports/model_comparison_classification.csv`.
- Zone metrics: `powerbi/zone_clusters.csv`.
- Rider metrics: `powerbi/rider_segments.csv`.
- Recommendations: `powerbi/business_findings.csv`.

## Keep these caveats
- State the SLA threshold used by the classifier.
- Treat the dark-store cluster locations as candidate areas, not confirmed sites.
- Do not report projected savings as achieved results.
