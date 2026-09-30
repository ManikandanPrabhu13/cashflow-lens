# CashFlow-Lens

### Explainable & Evidence-Aware Credit Risk Analytics for MSMEs

CashFlow-Lens is a comprehensive, modular credit risk analytics platform designed for Micro, Small, and Medium Enterprises (MSMEs). It moves beyond traditional "black-box" credit scoring by integrating alternative data, cryptographic-style evidence provenance, and deep explainability.

**Core Philosophy: PREDICT → VERIFY → TRACE → EXPLAIN → FORECAST → AUDIT → STRESS TEST → HUMAN DECISION SUPPORT**

*Note: This system does NOT make autonomous lending decisions (e.g., auto-approve or auto-reject). It provides rich, contextual decision support for human underwriters.*

## Key Features

1. **Model Pipeline & Explainability**: Utilizes LightGBM/Random Forest with SHAP integration to provide global feature importance and local, borrower-level reason codes.
2. **Evidence Integrity**: Cross-checks alternative data streams (e.g., Invoice ↔ GST ↔ Bank Settlement) to flag mismatches without prematurely declaring "fraud."
3. **Data Provenance**: Traces the lineage of claimed financial metrics back to their foundational documents.
4. **Cash-Flow Forecasting**: Projects 30, 60, and 90-day liquidity and debt-service capacity using exponentially weighted historical averages.
5. **Fairness & Mitigation**: Audits models across protected groups (e.g., gender, minority status) using Fairlearn, computing Equalized Odds and Disparate Impact, and applies threshold adjustments for mitigation.
6. **Stress Testing**: Allows underwriters to apply configurable macroeconomic shocks (sales drops, input cost increases, interest rate hikes) to observe risk migration in real-time.
7. **Human Decision Support**: Recommends actions like "Manual Review Recommended", "Continue Monitoring", or "Additional Evidence Required" based on comprehensive risk segmentation.

## Repository Architecture

```text
cashflow-lens/
├── backend/                  # FastAPI & Python ML Pipeline
│   ├── api/                  # REST API Endpoints
│   ├── src/                  # Core ML, Evidence, and Forecasting Modules
│   ├── scripts/              # Demo initialization script
│   ├── tests/                # Pytest suite
│   ├── data/                 # Raw and processed data (gitignored)
│   └── artifacts/            # Trained models, explainers, metrics (gitignored)
└── frontend/                 # React, Vite, Tailwind Dashboard
    ├── src/
    │   ├── components/       # Reusable UI widgets
    │   ├── pages/            # Core dashboard views
    │   └── services/         # API integration layer
    └── ...