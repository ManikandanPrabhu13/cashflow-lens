<div align="center">

# 💰 CashFlow-Lens

### Explainable & Evidence-Aware Credit Risk Analytics for MSMEs

*From financial evidence to explainable risk — without turning lending into a black box.*

[![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB?style=for-the-badge&logo=react&logoColor=black)](#)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](#)
[![ML](https://img.shields.io/badge/ML-LightGBM%20%7C%20Random%20Forest-FF6F00?style=for-the-badge&logo=scikit-learn&logoColor=white)](#)
[![Explainability](https://img.shields.io/badge/Explainability-SHAP-8A2BE2?style=for-the-badge)](#)
[![Fairness](https://img.shields.io/badge/Fairness-Fairlearn-00A86B?style=for-the-badge)](#)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange?style=for-the-badge)](#)

</div>

---

## ✨ What is CashFlow-Lens?

CashFlow-Lens is a **comprehensive, modular credit risk analytics platform** designed for Micro, Small, and Medium Enterprises (MSMEs).

It moves beyond traditional **"black-box" credit scoring** by integrating:

- alternative financial data
- evidence integrity analysis
- provenance-oriented financial tracing
- explainable machine learning
- cash-flow forecasting
- fairness auditing
- scenario-based stress testing

The goal is to provide an underwriter with **more context than a single credit score**.

> **PREDICT → VERIFY → TRACE → EXPLAIN → FORECAST → AUDIT → STRESS TEST → HUMAN DECISION SUPPORT**

> ⚠️ **Important:** This system does **NOT** make autonomous lending decisions such as auto-approving or auto-rejecting borrowers. It provides rich, contextual decision support for human underwriters.

---

# 🎯 Why CashFlow-Lens?

MSMEs can have valuable financial signals even when conventional credit information is limited or fragmented.

A credit assessment may need to consider relationships between:

```text
       Cash Flow
           │
           ├──────────────┐
           │              │
       Invoices          GST
           │              │
           └──────┬───────┘
                  │
            Bank Settlement
                  │
             Loan History
                  │
                  ▼
          Credit Risk Context
```

CashFlow-Lens therefore looks beyond a single score and asks:

> **What is the risk?**

> **Why does the model see this risk?**

> **What evidence supports the financial claim?**

> **How might the borrower's cash position evolve?**

> **How does the risk change under stress?**

> **Does the model behave differently across groups?**

---

# ✨ Key Features

### 1. 🧠 Model Pipeline & Explainability

Utilizes **LightGBM / Random Forest / Logistic Regression** with SHAP integration to provide:

- Global feature importance
- Local borrower-level explanations
- Reason codes
- Probability-based risk estimates
- Model evaluation metrics

The evaluation pipeline includes:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1
- Brier Score
- KS Statistic
- Confusion Matrix

---

### 2. 🔍 Evidence Integrity

CashFlow-Lens cross-checks alternative financial data streams to identify inconsistencies.

Examples include:

```text
Invoice ↔ GST
Invoice ↔ Bank Settlement
Purchase ↔ Inventory
Vendor ↔ Transaction
Revenue ↔ Cash Settlement
```

The system is designed to **flag mismatches without prematurely declaring "fraud."**

An inconsistency becomes a signal for investigation rather than an automatic accusation.

---

### 3. 🔗 Data Provenance

CashFlow-Lens traces the lineage of claimed financial metrics back toward their underlying records.

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

### 4. 📈 Cash-Flow Forecasting

Projects:

- **30-day**
- **60-day**
- **90-day**

cash-flow and liquidity conditions using **exponentially weighted historical averages**.

The forecasting layer provides indicators such as:

- Expected inflow
- Expected outflow
- Net cash flow
- Projected balance
- Liquidity pressure
- Debt-service capacity indicators

---

### 5. ⚖️ Fairness & Mitigation

CashFlow-Lens audits model behavior across configured groups using **Fairlearn**.

Current analysis includes:

- Equalized Odds
- Disparate Impact
- Group-level performance comparisons
- Threshold-based mitigation experiments

The goal is to make model behavior **auditable**, rather than treating overall model performance as the only measure of quality.

---

### 6. 🌪️ Stress Testing

Underwriters can apply configurable financial shocks such as:

```text
📉 Sales Drop
📈 Input Cost Increase
📈 Interest Rate Increase
📈 Operating Expense Increase
```

The system then evaluates the resulting stressed state to observe potential **risk migration and sensitivity**.

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

---

### 7. 👤 Human Decision Support

CashFlow-Lens provides contextual signals such as:

```text
⚠ Manual Review Recommended
⚠ Additional Evidence Required
✓ Continue Monitoring
⚠ Evidence Mismatch Detected
```

These signals are intended to help an underwriter determine **what requires further investigation**.

> 🤝 **The system supports the decision. It does not make the lending decision.**

---

# 🏗️ Architecture

CashFlow-Lens is organized as a modular pipeline from financial evidence to human decision support.

```text
==========================================================================================
🏢 1. FINANCIAL DATA LAYER
   Synthetic MSME Financial Data

   [ Cash Flow ]   [ Invoices ]   [ Payments ]   [ GST ]   [ Loans ]
         │               │              │           │          │
         └───────────────┴──────────────┴───────────┴──────────┘
                                  │
                                  ▼
==========================================================================================
🧹 2. DATA & FEATURE ENGINEERING
   Python + Pandas + NumPy + Scikit-learn

   [ Validation ] → [ Preprocessing ] → [ Feature Engineering ]
                                  │
                                  ▼
==========================================================================================
🤖 3. CREDIT RISK ENGINE

   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │ Logistic         │   │ Random Forest    │   │ LightGBM         │
   │ Regression       │   │                  │   │                  │
   └──────────────────┘   └──────────────────┘   └──────────────────┘
                                  │
                                  ▼
                           Risk Assessment
                                  │
                  ┌───────────────┼────────────────┐
                  ▼               ▼                ▼
==========================================================================================
🧠 4. INTELLIGENCE LAYERS

   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │ 🔍 EVIDENCE      │  │ 🔗 PROVENANCE    │  │ 🧠 EXPLAINABILITY│
   │    INTEGRITY     │  │                  │  │                  │
   │                  │  │ Claim → Source   │  │ SHAP             │
   │ Cross-source     │  │ → Record         │  │ Local reasons    │
   │ consistency      │  │                  │  │                  │
   └──────────────────┘  └──────────────────┘  └──────────────────┘

   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │ 📈 FORECASTING   │  │ ⚖️ FAIRNESS      │  │ 🌪️ STRESS TEST   │
   │                  │  │                  │  │                  │
   │ 30 / 60 / 90 day │  │ Equalized Odds  │  │ Sales shock      │
   │ liquidity        │  │ Disparate       │  │ Cost shock       │
   │ projections      │  │ Impact          │  │ Interest shock   │
   └──────────────────┘  └──────────────────┘  └──────────────────┘
                                  │
                                  ▼
==========================================================================================
🔌 5. APPLICATION & API LAYER
   FastAPI

   [ Risk ] [ Borrower ] [ Cash Flow ] [ Evidence ]
   [ Provenance ] [ Fairness ] [ Stress Testing ]
                                  │
                                  ▼
==========================================================================================
🖥️ 6. PRESENTATION LAYER
   React + Vite + Tailwind CSS

   [ Dashboard ] [ Borrower ] [ Risk Analytics ]
   [ Cash Flow ] [ Evidence ] [ Provenance ]
   [ Fairness ] [ Stress Testing ]
                                  │
                                  ▼
==========================================================================================
👤 HUMAN UNDERWRITER

   Review → Investigate → Compare Evidence → Assess Risk
   → Consider Scenarios → Make Final Decision
==========================================================================================
```

---

## 🔍 Layer by Layer

| Layer | What it does |
|---|---|
| 🏢 **1. Financial Data** | Provides structured MSME financial information for analysis and controlled demonstrations. |
| 🧹 **2. Data & Features** | Validates, preprocesses, and transforms financial information into model-ready features. |
| 🤖 **3. Credit Risk Engine** | Trains/evaluates multiple models and produces borrower-level risk estimates. |
| 🔍 **4. Intelligence Layers** | Adds evidence integrity, provenance, explainability, forecasting, fairness, and stress analysis. |
| 🔌 **5. API & Application** | FastAPI services expose the analytical capabilities to the frontend. |
| 🖥️ **6. Presentation** | React dashboard presents the combined financial and risk intelligence. |
| 👤 **Human Decision Support** | Keeps final lending decisions with the human underwriter. |

---

# 📊 Model Evaluation

CashFlow-Lens evaluates models using multiple complementary metrics rather than relying on accuracy alone.

```text
                 MODEL EVALUATION
                        │
       ┌────────────────┼────────────────┐
       ▼                ▼                ▼
   Predictive       Probability       Error
   Performance      Quality           Analysis
       │                │                │
   ROC-AUC          Brier Score      Precision
   PR-AUC                            Recall
   KS                                F1
                                     Confusion Matrix
```

This is particularly relevant to credit-risk modeling, where class imbalance and probability quality can matter as much as raw classification accuracy.

---

# 🧪 Synthetic Data & Privacy

The current demonstration environment uses **synthetic MSME financial data**.

No real:

- 🏦 Bank account information
- 💳 UPI transaction data
- 🧾 GST records
- 🏛️ Government financial records
- 👤 Customer financial records

are accessed by the project.

The synthetic-data generator supports controlled scenarios including:

- Unsettled invoices
- GST/revenue mismatches
- Duplicate invoices
- Vendor identity mismatches
- Inventory inconsistencies
- Unusual inflow patterns
- Transaction anomalies

> **Synthetic scenarios are used for engineering validation and demonstration. They should not be interpreted as evidence about real-world fraud prevalence or real lending populations.**

---

# 📁 Repository Architecture

```text
cashflow-lens/
│
├── backend/                  # FastAPI & Python ML Pipeline
│   ├── api/                  # REST API Endpoints
│   ├── src/                  # Core ML, Evidence, and Forecasting Modules
│   │   ├── model/
│   │   ├── explainability/
│   │   ├── evidence/
│   │   ├── provenance/
│   │   ├── forecasting/
│   │   ├── fairness/
│   │   └── stress_testing/
│   │
│   ├── scripts/              # Demo initialization scripts
│   ├── tests/                # Pytest suite
│   ├── data/                 # Raw & processed data (gitignored)
│   └── artifacts/            # Models, explainers & metrics (gitignored)
│
├── frontend/                 # React, Vite & Tailwind Dashboard
│   ├── src/
│   │   ├── components/       # Reusable UI widgets
│   │   ├── pages/            # Core dashboard views
│   │   └── services/         # API integration layer
│   └── ...
│
└── README.md
```

---

# 🛠️ Technology Stack

### Backend

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](#)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](#)
[![Scikit Learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](#)

### Machine Learning

[![LightGBM](https://img.shields.io/badge/LightGBM-2F6B3C?style=flat-square)](#)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-8A2BE2?style=flat-square)](#)
[![Fairlearn](https://img.shields.io/badge/Fairlearn-Fairness-00A86B?style=flat-square)](#)

### Frontend

[![React](https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=react&logoColor=black)](#)
[![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat-square&logo=vite&logoColor=white)](#)
[![Tailwind](https://img.shields.io/badge/Tailwind_CSS-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white)](#)

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

Generate the demonstration dataset and artifacts:

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

Open the local Vite development URL displayed in the terminal.

---

# 🛡️ Responsible AI & Safety Principles

> **A credit prediction is not a lending decision.**

CashFlow-Lens is built around several principles:

- 🔍 **Evidence before conclusions** — inconsistencies trigger investigation rather than automatic fraud labels.
- 🧠 **Explainability over black boxes** — predictions are accompanied by contributing factors.
- 🤝 **Human oversight** — final lending decisions remain with human decision-makers.
- 📊 **Multiple signals** — risk is examined through multiple financial indicators.
- 🌪️ **Scenario awareness** — financial resilience is examined under configurable stress conditions.
- ⚖️ **Model accountability** — predictive performance and group-level behavior are both evaluated.
- 🔐 **Privacy-aware development** — the current prototype uses synthetic financial data.

---

# 🧪 Testing

The backend includes a **Pytest-based test suite** covering core components of the analytical system.

Testing is structured around the modular architecture so that data processing, analytical components, and API behavior can be validated independently.

---

# 🗺️ Roadmap

### ✅ Implemented

- [x] Synthetic MSME financial-data generation
- [x] Financial feature engineering
- [x] Logistic Regression baseline
- [x] Random Forest
- [x] LightGBM
- [x] Multi-metric model evaluation
- [x] SHAP-based explainability
- [x] Evidence-integrity analysis
- [x] Provenance-oriented data structures
- [x] 30 / 60 / 90-day cash-flow forecasting
- [x] Fairness analysis with Fairlearn
- [x] Threshold-based mitigation experiments
- [x] Configurable stress-testing framework
- [x] FastAPI backend
- [x] React/Vite dashboard
- [x] Backend testing foundation

### 🚧 In Progress

- [ ] Complete end-to-end integration of analytical modules with API responses
- [ ] Fully dynamic borrower-specific evidence/provenance views
- [ ] Expanded backend/frontend integration testing
- [ ] Improved demo reproducibility
- [ ] Enhanced analytical visualizations

### 🔮 Future

- [ ] Additional temporal forecasting approaches
- [ ] Expanded provenance visualization
- [ ] Model monitoring
- [ ] Additional fairness diagnostics
- [ ] Authorized real-world data connectors
- [ ] Production-oriented security and deployment architecture

---

# ⚠️ Current Limitations

CashFlow-Lens is a **prototype / research-oriented system**, not a production lending platform.

- Demonstration data is synthetic.
- Synthetic model performance does not establish real-world lending performance.
- Forecasts are projections, not guarantees.
- Fairness metrics depend on the selected population, labels, groups, and methodology.
- Evidence mismatches require contextual investigation.
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

The long-term vision is to explore how **explainable machine learning, alternative financial evidence, provenance, forecasting, fairness analysis, and scenario testing** can work together to create richer and more transparent credit intelligence for MSMEs.

---

<div align="center">

## 💰 CashFlow-Lens

### **PREDICT • VERIFY • TRACE • EXPLAIN • FORECAST • AUDIT • STRESS TEST**

*Explainable credit intelligence for a more evidence-aware MSME lending workflow.*

</div>
