import { 
  RiskAnalysis, 
  EvidenceIntegrity, 
  ProvenanceTrace, 
  Forecast, 
  FairnessReport, 
  StressTestResponse 
} from '../types/api';

export const mockBorrowers = [
  { borrower_id: "DEMO-001" },
  { borrower_id: "DEMO-002" },
  { borrower_id: "DEMO-003" }
];

export const mockRiskAnalysis: RiskAnalysis = {
  borrower_id: "DEMO-001",
  pd: 0.18,
  risk_segment: "Medium Risk",
  decision_support: "Continue Monitoring",
  reason_codes: [
    "Increased risk due to recent_late_payments (value: 2.00)",
    "Decreased risk due to strong_cash_reserves (value: 25000.00)",
    "Increased risk due to volatile_revenue (value: 0.45)"
  ],
  top_contributions: [
    { feature: "cash_balance", contribution: -0.12, value: 25000 },
    { feature: "late_payment_count", contribution: 0.08, value: 2 },
    { feature: "revenue_volatility", contribution: 0.05, value: 0.45 },
    { feature: "industry_risk", contribution: -0.03, value: 1.2 },
    { feature: "debt_to_income", contribution: 0.02, value: 0.35 }
  ]
};

export const mockEvidence: EvidenceIntegrity = {
  borrower_id: "DEMO-001",
  status: "Needs Review",
  checks: [
    { relationship: "Invoice ↔ Bank Settlement", status: "Verified", difference: 0 },
    { relationship: "GST ↔ Invoice", status: "Mismatch", difference: 1250 },
    { relationship: "Purchase ↔ Inventory", status: "Needs Review", difference: null },
    { relationship: "Vendor ↔ MSME", status: "Verified", difference: 0 }
  ],
  alerts: ["GST mismatch detected in recent quarter"]
};

export const mockProvenance: ProvenanceTrace = {
  borrower_id: "DEMO-001",
  trace: [
    { source: "Claimed Revenue", target: "Invoice", match: true, amount_diff: 0 },
    { source: "Invoice", target: "Bank Settlement", match: true, amount_diff: 0 },
    { source: "Bank Settlement", target: "GST Record", match: false, amount_diff: 1250 }
  ]
};

export const mockForecast: Forecast = {
  "30d": {
    expected_inflow: 12500,
    expected_outflow: 8000,
    expected_net_cash_flow: 4500,
    projected_cash_balance: 19500,
    debt_service_capacity_dscr: 1.5,
    liquidity_pressure: "Low"
  },
  "60d": {
    expected_inflow: 24800,
    expected_outflow: 16500,
    expected_net_cash_flow: 8300,
    projected_cash_balance: 23300,
    debt_service_capacity_dscr: 1.4,
    liquidity_pressure: "Medium"
  },
  "90d": {
    expected_inflow: 36000,
    expected_outflow: 25000,
    expected_net_cash_flow: 11000,
    projected_cash_balance: 26000,
    debt_service_capacity_dscr: 1.3,
    liquidity_pressure: "Medium"
  }
};

export const mockFairness: FairnessReport = {
  baseline: {
    fairness: {
      group_metrics: {
        "Female-Owned": { tpr: 0.85, fpr: 0.15 },
        "Male-Owned": { tpr: 0.88, fpr: 0.12 }
      },
      disparities: {
        equal_opportunity_difference: 0.03,
        equalized_odds_difference: 0.03
      }
    }
  },
  mitigated: {
    fairness: {
      group_metrics: {
        "Female-Owned": { tpr: 0.86, fpr: 0.13 },
        "Male-Owned": { tpr: 0.87, fpr: 0.13 }
      },
      disparities: {
        equal_opportunity_difference: 0.01,
        equalized_odds_difference: 0.01
      }
    }
  },
  improvement: {
    equal_opportunity_diff_change: 0.02,
    equalized_odds_diff_change: 0.02
  }
};

export const mockStressTestResponse: StressTestResponse = {
  scenario: {
    sales_shock: -0.20,
    cost_shock: 0.10,
    interest_shock: 0.02,
    opex_shock: 0.0
  },
  baseline: {
    metrics: { monthly_revenue: 15000, monthly_expenses: 10000, net_cash_flow: 5000 },
    pd: 0.18,
    risk_segment: "Medium Risk"
  },
  stressed: {
    metrics: { monthly_revenue: 12000, monthly_expenses: 11000, net_cash_flow: 1000 },
    pd: 0.38,
    risk_segment: "High Risk"
  },
  impact: {
    pd_change: 0.20,
    risk_migration: "Medium Risk -> High Risk",
    revenue_impact: -3000,
    net_cash_flow_impact: -4000
  }
};