export interface BorrowerInfo {
  borrower_id: string;
}

export interface BorrowerDetails {
  [key: string]: any;
}

export interface RiskAnalysis {
  borrower_id: string;
  pd: number;
  risk_segment: "Low Risk" | "Medium Risk" | "High Risk";
  decision_support: string;
  reason_codes: string[];
  top_contributions: {
    feature: string;
    contribution: number;
    value: number;
  }[];
}

export interface EvidenceCheck {
  relationship: string;
  status: "Verified" | "Mismatch" | "Needs Review" | "No Evidence";
  difference: number | null;
}

export interface EvidenceIntegrity {
  borrower_id: string;
  status: string;
  checks: EvidenceCheck[];
  alerts: string[];
}

export interface ProvenanceTraceNode {
  source: string;
  target: string;
  match: boolean;
  amount_diff: number;
}

export interface ProvenanceTrace {
  borrower_id: string;
  trace: ProvenanceTraceNode[];
}

export interface ForecastPeriod {
  expected_inflow: number;
  expected_outflow: number;
  expected_net_cash_flow: number;
  projected_cash_balance: number;
  debt_service_capacity_dscr: number;
  liquidity_pressure: "Low" | "Medium" | "High";
}

export interface Forecast {
  "30d": ForecastPeriod;
  "60d": ForecastPeriod;
  "90d": ForecastPeriod;
}

export interface FairnessReport {
  baseline: {
    fairness: {
      group_metrics: Record<string, any>;
      disparities: Record<string, number>;
    };
  };
  mitigated: {
    fairness: {
      group_metrics: Record<string, any>;
      disparities: Record<string, number>;
    };
  };
  improvement: Record<string, number>;
}

export interface StressTestRequest {
  borrower_id: string;
  sales_shock: number;
  cost_shock: number;
  interest_shock: number;
  opex_shock: number;
}

export interface StressTestResponse {
  scenario: Record<string, number>;
  baseline: {
    metrics: Record<string, number>;
    pd: number;
    risk_segment: string;
  };
  stressed: {
    metrics: Record<string, number>;
    pd: number;
    risk_segment: string;
  };
  impact: {
    pd_change: number;
    risk_migration: string;
    revenue_impact: number;
    net_cash_flow_impact: number;
  };
}