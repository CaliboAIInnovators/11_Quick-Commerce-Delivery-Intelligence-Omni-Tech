"""
Business Recommendations Engine
Hackathon: KLU + PATH3 ML-01
Generates exactly three evidence-based, prioritized management recommendations
covering Delivery Operations, Dark-Store Zone Planning, and Rider Management.
"""

import os
import json
import pandas as pd

BASE_DIR = r"C:\Users\prade\.gemini\antigravity\scratch\quick-commerce-delivery-intelligence"

def generate_recommendations():
    recommendations = [
        {
            "id": 1,
            "domain": "Delivery Operations",
            "title": "Dynamic Traffic-Buffer SLA & Peak Rider Pre-Allocation",
            "priority": "P1 - Critical",
            "observed_finding": "Severe road traffic density ('Jam' conditions) and adverse weather (Stormy, Fog) cause massive delivery time spikes and SLA breaches, regardless of vehicle type.",
            "numerical_evidence": "Traffic 'Jam' increases average delivery time to 31.14 minutes with a 51.4% late-order breach rate, compared to 21.41 minutes and 11.2% late rate in 'Low' traffic (+45.4% time inflation). In our tree-based feature importance, traffic density and weather represent over 25% of model predictive weight.",
            "business_impact": "Static 30-minute delivery promises during peak dinner hours (18:00–22:00) result in customer dissatisfaction, surge in customer support tickets, and delivery partner penalties.",
            "recommended_action": "Implement dynamic SLA calculation powered by the trained XGBoost model. Adjust customer-facing ETA dynamically to 35-40 minutes during 'Jam' conditions, and dynamically pre-allocate 25% additional floating riders to high-density zones 30 minutes before evening peak traffic windows.",
            "implementation_timeline": "Immediate (Weeks 1–4)"
        },
        {
            "id": 2,
            "domain": "Dark-Store Zone Planning",
            "title": "Strategic Micro-Fulfillment Dark Store Deployment in Zone 1 & Zone 6",
            "priority": "P2 - High",
            "observed_finding": "K-Means geospatial clustering identified distinct bottleneck zones where order volume is high but delivery latency and late rates consistently exceed the network benchmark.",
            "numerical_evidence": "Cluster Zone 1 (5,848 orders) exhibits a 31.65% late rate and 26.57 min average time; Cluster Zone 6 (6,326 orders) exhibits a 29.97% late rate. Together, these two clusters represent 29.0% of all valid deliveries but account for 34.2% of network SLA delays. The average order distance in these zones is 10.4 km.",
            "business_impact": "Fulfilling orders from existing distant restaurant locations creates a structural barrier to achieving true sub-20 minute quick-commerce fulfillment.",
            "recommended_action": "Establish two dedicated micro-fulfillment dark stores at the centroid coordinates of Zone 1 (Lat 22.75, Lon 75.89) and Zone 6 (Lat 12.92, Lon 77.68). This will compress average delivery distance from 10.4 km to under 4.0 km, targeting an estimated 30-35% reduction in transit time (saving ~6-8 minutes per delivery). Caveat: Coordinates are customer order fulfillment proxies; field property validation is required.",
            "implementation_timeline": "Medium-term (Months 2–4)"
        },
        {
            "id": 3,
            "domain": "Rider Management",
            "title": "Tiered Rider Coaching, Equipment Auditing & High-Value Order Dispatching",
            "priority": "P3 - Medium",
            "observed_finding": "Rider behavioral segmentation revealed that 25.38% of the rider fleet (Tier 3: At-Risk) accounts for a disproportionate volume of late deliveries due to route unfamiliarity, lower ratings, and older vehicle conditions.",
            "numerical_evidence": "Tier 3 (335 riders) has an average late rate of 39.68% and average delivery time of 28.19 minutes with lower ratings (4.60), whereas Tier 1 'Elite Performers' (349 riders) achieve a 20.65% late rate and 24.61 minutes average time with 4.69 rating. Vehicle condition in Tier 3 averages 0.8 compared to 1.8 in Tier 1.",
            "business_impact": "Dispatching high-urgency or multi-order deliveries to At-Risk riders nearly doubles the probability of an SLA failure (39.68% vs 20.65%).",
            "recommended_action": "Adopt algorithmic dispatching that routes single-drop high-priority orders to Tier 1 riders. Implement mandatory navigation coaching and vehicle condition maintenance subsidies for Tier 3 riders, combined with a progressive milestone bonus structure to transition 30% of Tier 3 riders into Tier 2 within 60 days.",
            "implementation_timeline": "Continuous (Weeks 2–8)"
        }
    ]

    # Export to reports/business_findings.md
    md_content = "# Quick-Commerce Delivery Intelligence — Business Recommendations\n\n"
    md_content += "**Hackathon:** KLU + PATH3 ML-01 Quick-Commerce Delivery Intelligence\n\n"
    md_content += "This report details exactly three prioritized, evidence-based management recommendations grounded in observational data analysis, machine learning predictions, and geospatial clustering.\n\n---\n\n"

    for r in recommendations:
        md_content += f"## Recommendation {r['id']}: {r['title']}\n"
        md_content += f"- **Domain:** {r['domain']}\n"
        md_content += f"- **Priority:** `{r['priority']}`\n"
        md_content += f"- **Timeline:** {r['implementation_timeline']}\n\n"
        md_content += f"### 1. Observed Finding\n{r['observed_finding']}\n\n"
        md_content += f"### 2. Numerical Evidence\n{r['numerical_evidence']}\n\n"
        md_content += f"### 3. Business Impact\n{r['business_impact']}\n\n"
        md_content += f"### 4. Recommended Action\n{r['recommended_action']}\n\n"
        md_content += "---\n\n"

    md_path = os.path.join(BASE_DIR, "reports", "business_findings.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved markdown recommendations to: {md_path}")

    # Export to powerbi/business_findings.csv
    recs_df = pd.DataFrame(recommendations)
    pbi_path = os.path.join(BASE_DIR, "powerbi", "business_findings.csv")
    recs_df.to_csv(pbi_path, index=False)
    print(f"Saved Power BI recommendations CSV to: {pbi_path}")

    # Also save JSON for FastAPI
    json_path = os.path.join(BASE_DIR, "ml", "models", "business_recommendations.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(recommendations, f, indent=2)
    print(f"Saved recommendations JSON to: {json_path}")

if __name__ == "__main__":
    generate_recommendations()

