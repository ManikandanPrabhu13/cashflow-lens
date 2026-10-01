<div align="center">

# 💰 CashFlow-Lens

### Explainable & Evidence-Aware Credit Risk Analytics for MSMEs

*From financial evidence to explainable risk — without turning lending into a black box.*

[![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB?style=for-the-badge&logo=react&logoColor=black)](#)  
[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](#)  
[![ML](https://img.shields.io/badge/ML-LightGBM%20%7C%20Random%20Forest-FF6F00?style=for-the-badge&logo=scikit-learn&logoColor=white)](#)  
[![Explainability](https://img.shields.io/badge/Explainability-SHAP-8A2BE2?style=for-the-badge)](#)  
[![Fairness](https://img.shields.io/badge/Fairness-Fairlearn-00A86B?style=for-the-badge)](#)  
[![Status](https://img.shields.io/badge/Status-Prototype-orange?style=for-the-badge)](#)

</div>

---

## ✨ What is CashFlow-Lens?

CashFlow-Lens is a **modular credit risk analytics platform** designed for Micro, Small, and Medium Enterprises (MSMEs).

It combines financial data processing, machine-learning-based credit-risk analysis, explainability, evidence-oriented checks, forecasting, fairness analysis, and financial stress testing into a single analytical workflow.

The platform is designed to provide an underwriter with **more context than a single credit score**.

> **PREDICT → VERIFY → TRACE → EXPLAIN → FORECAST → AUDIT → STRESS TEST → HUMAN DECISION SUPPORT**

> ⚠️ **Important:** CashFlow-Lens does **not** autonomously approve or reject loans. It is designed as decision support for human underwriters.

---

# 🎯 Why CashFlow-Lens?

MSME financial information can be fragmented across multiple financial records.

CashFlow-Lens therefore considers relationships between:

```text
        Cash Flow
            │
      ┌─────┴─────┐
      │           │
   Invoices      GST
      │           │
      └─────┬─────┘
            │
     Bank Settlement
            │
       Loan History
            │
            ▼
      Credit Risk Context
```

Instead of asking only:

> **What is the risk?**

CashFlow-Lens also attempts to provide:

> **Why does the model see this risk?**

> **What evidence supports the financial information?**

> **How could the cash position evolve?**

> **How could risk change under financial stress?**

> **Does model behaviour differ across configured groups?**

---

# ✨ Key Features

## 1. 🧠 Credit Risk Modeling & Explainability

The project implements multiple machine-learning approaches:

- Logistic Regression
- Random Forest
- LightGBM

The modeling pipeline supports:

- Borrower-level risk prediction
- Probability-based risk estimates
- Global feature analysis
- Local explanations
- SHAP-based explainability
- Model comparison

### Evaluation Metrics

The project evaluates models using multiple metrics:

```text
ROC-AUC
PR-AUC
Precision
Recall
F1
Brier Score
KS Statistic
Confusion Matrix
```

This allows the system to look beyond simple classification accuracy.

---

# 2. 🔍 Evidence Integrity

CashFlow-Lens includes an evidence-oriented layer for checking consistency between financial records.

Current evidence relationships represented by the project include:

```text
Invoice ↔ GST
Invoice ↔ Bank Settlement
Purchase ↔ Inventory
Vendor ↔ Transaction
Revenue ↔ Cash Settlement
```

The project also contains synthetic scenarios for inconsistencies such as:

- GST/revenue mismatch
- Unsettled invoices
- Duplicate invoices
- Vendor identity mismatch
- Inventory inconsistencies
- Unusual transaction/inflow patterns

The philosophy is:

> **An inconsistency is a signal for investigation — not an automatic declaration of fraud.**

### Current Implementation Status

The evidence layer is currently at **prototype/integration stage**.

The project contains the underlying financial relationships and scenario generation, while some API-level evidence responses are still demonstration-oriented rather than fully dynamic borrower-specific reconciliation.

---

# 3. 🔗 Data Provenance

CashFlow-Lens includes provenance-oriented structures for tracing financial information toward its underlying records.

Conceptually:

```text
Financial Claim
      │
      ▼
 Source Record
      │
      ▼
Transaction / Invoice
      │
      ▼
Supporting Evidence
```

This provides an evidence-oriented view of where important financial information originates.

---

# 4. 📈 Cash-Flow Forecasting

The forecasting layer projects future cash-flow conditions over:

- **30 days**
- **60 days**
- **90 days**

The current approach uses **exponentially weighted historical averages**.

The forecasting output includes indicators such as:

```text
Expected Inflow
Expected Outflow
Net Cash Flow
Projected Balance
Liquidity Pressure
Debt-Service Capacity Indicators
```

These projections are intended as analytical estimates rather than guarantees.

---

# 5. ⚖️ Fairness Analysis

CashFlow-Lens includes fairness analysis using **Fairlearn**.

Current analysis includes:

- Equalized Odds
- Disparate Impact
- Group-level performance comparison
- Threshold-based mitigation experiments

The purpose is to make model behaviour more auditable across configured groups instead of evaluating the model only through overall predictive performance.

---

# 6. 🌪️ Financial Stress Testing

The project includes configurable financial stress scenarios.

Examples:

```text
📉 Sales Drop
📈 Input Cost Increase
📈 Interest Rate Increase
📈 Operating Expense Increase
```

The workflow is:

```text
Baseline
   │
   ▼
Apply Financial Shock
   │
   ▼
Recalculate Features
   │
   ▼
Re-evaluate Risk
   │
   ▼
Compare with Baseline
```

This allows analysis of potential **risk migration and financial sensitivity** under adverse conditions.

---

# 7. 👤 Human Decision Support

CashFlow-Lens is designed to provide contextual signals rather than autonomous lending decisions.

Example signals include:

```text
⚠ Manual Review Recommended
⚠ Additional Evidence Required
✓ Continue Monitoring
⚠ Evidence Mismatch Detected
```

The intended workflow is:

```text
System Analysis
      ↓
Evidence Review
      ↓
Risk Interpretation
      ↓
Human Underwriter
      ↓
Final Decision
```

> 🤝 **The system supports the decision. It does not make the lending decision.**

---

# 🏗️ System Architecture

```text
================================================================================
🏢 1. FINANCIAL DATA LAYER
================================================================================

                Synthetic MSME Financial Data

     [ Cash Flow ] [ Invoices ] [ Payments ] [ GST ] [ Loans ]
             │          │            │          │        │
             └──────────┴────────────┴──────────┴────────┘
                                │
                                ▼

================================================================================
🧹 2. DATA & FEATURE ENGINEERING
================================================================================

        Validation → Preprocessing → Feature Engineering
                                │
                                ▼

================================================================================
🤖 3. CREDIT RISK ENGINE
================================================================================

       ┌────────────────┐
       │ Logistic       │
       │ Regression     │
       └────────────────┘

       ┌────────────────┐
       │ Random Forest  │
       └────────────────┘

       ┌────────────────┐
       │ LightGBM       │
       └────────────────┘

                 │
                 ▼
          Risk Assessment
                 │
                 ▼

================================================================================
🧠 4. INTELLIGENCE LAYERS
================================================================================

 ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
 │ 🔍 Evidence      │ │ 🔗 Provenance    │ │ 🧠 Explainability│
 │    Integrity     │ │                  │ │                  │
 │ Cross-source     │ │ Claim → Source   │ │ SHAP             │
 │ consistency      │ │ → Record         │ │ Local reasons    │
 └──────────────────┘ └──────────────────┘ └──────────────────┘

 ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
 │ 📈 Forecasting   │ │ ⚖️ Fairness      │ │ 🌪️ Stress Test   │
 │                  │ │                  │ │                  │
 │ 30/60/90 day     │ │ Equalized Odds   │ │ Sales shock      │
 │ projections      │ │ Disparate Impact │ │ Cost shock       │
 └──────────────────┘ └──────────────────┘ └──────────────────┘
                 │
                 ▼

================================================================================
🔌 5. APPLICATION & API LAYER
================================================================================

                         FastAPI

 [ Risk ] [ Borrower ] [ Cash Flow ] [ Evidence ]
 [ Provenance ] [ Fairness ] [ Stress Testing ]
                 │
                 ▼

================================================================================
🖥️ 6. PRESENTATION LAYER
================================================================================

                  React + Vite + Tailwind CSS

 [ Dashboard ] [ Borrower ] [ Risk Analytics ]
 [ Cash Flow ] [ Evidence ] [ Provenance ]
 [ Fairness ] [ Stress Testing ]
                 │
                 ▼

================================================================================
👤 HUMAN UNDERWRITER
================================================================================

 Review → Investigate → Compare Evidence
       → Assess Risk → Consider Scenarios
       → Make Final Decision
================================================================================
```

---

# 🔍 Layer-by-Layer Architecture

| Layer | Responsibility |
|---|---|
| 🏢 **Financial Data** | Generates and provides structured MSME financial information for controlled analysis and demonstrations. |
| 🧹 **Data & Features** | Validates, preprocesses, and transforms financial information into model-ready features. |
| 🤖 **Credit Risk Engine** | Trains/evaluates multiple models and produces borrower-level risk estimates. |
| 🔍 **Evidence Integrity** | Represents cross-source financial consistency checks and mismatch scenarios. |
| 🔗 **Provenance** | Provides structures for tracing financial information toward supporting records. |
| 🧠 **Explainability** | Uses SHAP to interpret model predictions. |
| 📈 **Forecasting** | Produces 30/60/90-day cash-flow projections. |
| ⚖️ **Fairness** | Evaluates model behaviour across configured groups. |
| 🌪️ **Stress Testing** | Evaluates risk under configurable financial shocks. |
| 🔌 **API** | Exposes analytical capabilities through FastAPI. |
| 🖥️ **Frontend** | Presents analytical outputs through a React/Vite dashboard. |
| 👤 **Human Decision Support** | Keeps final lending decisions with the human underwriter. |

---

# 🧪 Synthetic Financial Data

The current project operates using **synthetic MSME financial data**.

It does not require real:

- Bank account information
- UPI transaction data
- GST records
- Government financial records
- Customer financial records

The synthetic generator supports controlled scenarios such as:

```text
Unsettled invoices
GST / revenue mismatches
Duplicate invoices
Vendor identity mismatches
Inventory inconsistencies
Unusual inflow patterns
Transaction anomalies
```

This allows the analytical pipeline to be tested without exposing real financial information.

> **Synthetic scenarios are for engineering validation and demonstration. They should not be interpreted as evidence about real-world fraud prevalence or real lending populations.**

---

# 📊 Model Evaluation

The evaluation pipeline uses multiple complementary metrics:

```text
                 MODEL EVALUATION
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     Predictive     Probability    Error
     Performance     Quality      Analysis
          │            │            │
       ROC-AUC      Brier Score   Precision
       PR-AUC                     Recall
       KS                          F1
                                  Confusion Matrix
```

This provides a broader view of model behaviour, particularly for credit-risk classification where class imbalance and probability quality can be important.

---

# 📁 Repository Structure

```text
cashflow-lens/
│
├── backend/
│   ├── api/
│   │   └── main.py
│   │
│   ├── src/
│   │   ├── model/
│   │   ├── explainability/
│   │   ├── evidence/
│   │   ├── provenance/
│   │   ├── forecasting/
│   │   ├── fairness/
│   │   └── stress_testing/
│   │
│   ├── scripts/
│   ├── tests/
│   ├── data/
│   └── artifacts/
│
├── frontend/
│   └── src/
│       ├── components/
│       ├── pages/
│       └── services/
│
└── README.md
```

---

# 🛠️ Technology Stack

### Backend

- Python
- FastAPI
- Pandas
- NumPy
- Scikit-learn

### Machine Learning

- LightGBM
- Random Forest
- Logistic Regression
- SHAP
- Fairlearn

### Frontend

- React
- Vite
- Tailwind CSS

### Testing

- Pytest

---

# 🚀 Getting Started

## 1. Clone

```bash
git clone https://github.com/ManikandanPrabhu13/cashflow-lens.git
cd cashflow-lens
```

## 2. Backend

```bash
cd backend

python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate demonstration data and artifacts:

```bash
python scripts/setup_demo.py
```

Start the API:

```bash
uvicorn api.main:app --reload
```

---

## 3. Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local Vite development URL shown in the terminal.

---

# 🛡️ Responsible AI & Safety

> **A credit prediction is not a lending decision.**

CashFlow-Lens follows these principles:

- 🔍 **Evidence before conclusions** — inconsistencies trigger investigation rather than automatic fraud labels.
- 🧠 **Explainability over black boxes** — predictions can be accompanied by contributing factors.
- 🤝 **Human oversight** — final lending decisions remain with human decision-makers.
- 📊 **Multiple signals** — risk is examined through multiple financial indicators.
- 🌪️ **Scenario awareness** — financial resilience can be examined under configurable stress.
- ⚖️ **Model accountability** — predictive performance and group-level behaviour are evaluated.
- 🔐 **Privacy-aware development** — the current prototype uses synthetic financial data.

---

# 🧪 Testing

The backend contains a **Pytest-based testing foundation** for validating components of the analytical system.

The modular architecture allows data processing, analytical components, and API behaviour to be tested independently.

---

# 🗺️ Current Development Status

### ✅ Implemented / Present

- [x] Synthetic MSME financial-data generation
- [x] Financial feature engineering
- [x] Logistic Regression
- [x] Random Forest
- [x] LightGBM
- [x] Multi-metric model evaluation
- [x] SHAP-based explainability
- [x] Vendor-related financial data and mismatch scenarios
- [x] Evidence-integrity framework structure
- [x] Provenance-oriented structures
- [x] 30 / 60 / 90-day cash-flow forecasting
- [x] Fairness analysis with Fairlearn
- [x] Threshold-based mitigation experiments
- [x] Configurable stress-testing framework
- [x] FastAPI backend
- [x] React/Vite dashboard
- [x] Backend testing foundation

### 🚧 Integration Stage

- [ ] Fully dynamic evidence reconciliation
- [ ] Fully borrower-specific provenance responses
- [ ] Complete API integration of every analytical module
- [ ] Expanded frontend/backend integration testing
- [ ] Improved demo reproducibility
- [ ] Enhanced analytical visualizations

---

# ⚠️ Current Limitations

CashFlow-Lens is currently a **prototype / research-oriented analytical system**, not a production lending platform.

- Demonstration data is synthetic.
- Synthetic model performance does not establish real-world lending performance.
- Forecasts are projections, not guarantees.
- Fairness metrics depend on the selected population, labels, groups, and methodology.
- Evidence mismatches require contextual investigation.
- Some evidence/API responses remain demonstration-oriented rather than fully dynamic.
- The system does not autonomously approve or reject loans.
- Production deployment would require additional security, governance, monitoring, regulatory validation, and domain-specific testing.

---

# 💡 Project Vision

> **Credit assessment should provide more than a number.**

CashFlow-Lens brings together:

```text
             RISK
               │
               ▼
         EXPLANATION
               │
               ▼
            EVIDENCE
               │
               ▼
          PROVENANCE
               │
               ▼
           FORECAST
               │
               ▼
           FAIRNESS
               │
               ▼
        STRESS TESTING
               │
               ▼
      HUMAN DECISION SUPPORT
```

The long-term vision is to explore how **explainable machine learning, alternative financial evidence, provenance, forecasting, fairness analysis, and financial stress testing** can work together to create richer and more transparent credit intelligence for MSMEs.

---

<div align="center">

# 💰 CashFlow-Lens

### **PREDICT • VERIFY • TRACE • EXPLAIN • FORECAST • AUDIT • STRESS TEST**

*Explainable credit intelligence for a more evidence-aware MSME lending workflow.*

</div>
