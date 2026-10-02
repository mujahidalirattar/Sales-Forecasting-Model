# Product Requirement Document (PRD)

## Project Title
**Walmart Executive Sales Analytics & Predictive Optimization Dashboard**

---

## 1. Executive Summary & Objective
The objective of this project is to deliver an enterprise-grade Data Analytics (80%) and Machine Learning (20%) dashboard for Walmart store managers and executives. The platform transitions traditional raw sales forecasting into an actionable decision-making tool. It combines predictive ML modeling with diagnostic descriptive analytics, interactive regression trends, inventory risk indicators, labor allocation models, and a self-service model retraining module.

---

## 2. Target Audience & Business Value
* **Store Operations Managers:** Optimize weekly staff scheduling and reduce cashier bottleneck risks.
* **Supply Chain & Inventory Analysts:** Mitigate stockouts and excess holding costs via dynamic inventory buffer alerts.
* **Executive Decision Makers:** Analyze macro-economic variables (CPI, Fuel Prices, Unemployment) and baseline linear trends against machine learning predictions.

---

## 3. Tech Stack & Architecture

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.13 |
| **Frontend Framework** | Streamlit |
| **Data Visualization** | Plotly Express & Plotly Graph Objects |
| **Machine Learning** | Scikit-Learn (RandomForestRegressor, LinearRegression) |
| **Statistical Analysis** | Statsmodels / NumPy / Pandas |
| **Model Serialization** | Joblib |
| **Styling & Effects** | Custom CSS3 (Animations, Keyframe Loaders, Transitions) |

---

## 4. Key Functional Modules

### Module A: Executive Sales Forecasting Simulator (ML 20%)
* **Single & Batch Inference:** Predict weekly sales using store parameters, date pickers, and regional economic indicators.
* **Scenario Modeling:** Interactive sliders to test "What-If" scenarios (e.g., impact of +10% Fuel Price hike or Holiday surges).

### Module B: Core Data Analytics & Statistical Diagnostics (DA 80%)
* **Linear Regression Baseline vs. RF Model:** Interactive scatter plot with Linear Regression trendline ($y = \beta_0 + \beta_1 x + \epsilon$) to visualize overall baseline trajectory versus non-linear Random Forest predictions.
* **Economic Factor Correlation Matrix:** Dynamic heatmap showing correlation between CPI, Unemployment, Fuel Price, Temperature, and Weekly Sales.
* **Holiday Surge Analysis:** Comparative bar and box plots showing holiday week sales distribution vs. non-holiday baselines.
* **Store & Dept Multi-metric Comparison:** Drill-down filter to compare performance across stores (1–45) and department verticals.

### Module C: Operational Resource Allocation (Prescriptive Analytics)
* **Inventory Stock Risk Engine:**
  * $\text{Stockout Risk}$: $\text{Current Stock} < \text{Predicted Demand}$
  * $\text{Overstock Risk}$: $\text{Current Stock} > (\text{Predicted Demand} \times 1.20)$
* **Labor & Staffing Engine:**
  * Staff Ratio: 1 worker per $5,000 predicted sales.
  * Allocation Split: 35% Cashiers, 65% Floor & Inventory Staff.

### Module D: Continuous Learning Pipeline (Data Upload & Retraining)
* **File Uploader:** Support for `.csv` and `.xlsx` dataset uploads matching the schema.
* **Data Validation Engine:** Automated check for required columns and null values before model fitting.
* **On-the-Fly Retraining:** User-triggered retraining interface that updates the local `.pkl` model file with fresh data and outputs updated $R^2$, MAE, and RMSE metrics.

---

## 5. UI/UX Specifications

### Color Palette (Strict Light Mode)
* **Primary Background:** `#F8FAFC` (Clean Slate)
* **Card & Container Background:** `#FFFFFF` (Pure White with subtle `#E2E8F0` border)
* **Primary Accent:** `#0284C7` (Walmart Navy/Sky Blue)
* **Secondary Accent:** `#0EA5E9` (Vibrant Cyan)
* **Success/Warning/Alerts:** `#10B981` (Emerald), `#F59E0B` (Amber), `#EF4444` (Rose)
* **Forbidden Colors:** Pure Dark Mode Backgrounds (`#000000`, `#121212`, `#1E1E1E`).

### Visual Effects & Interactivity
* **Hover State:** All metric cards and containers feature smooth CSS transition (`transform: translateY(-3px)`, `box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05)`).
* **Custom Typography:** Clean sans-serif hierarchy with bold metric numbers.
* **Custom Loader:** Dynamic CSS glowing/shimmering text animation on the phrase **"SALES"** displayed during data processing and model retraining steps.

---

## 6. Non-Functional Requirements
* **Response Time:** Predictions and chart renderings must complete in $< 1.5$ seconds.
* **Modular Code Structure:** Clean separation between layout configuration, styling, data pipeline, and prediction logic.
* **Cross-Browser Compatibility:** Responsive layout rendered cleanly across Chrome, Edge, and Safari.
