from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 1. Initialize the FastAPI app FIRST
app = FastAPI(title="CashFlow-Lens API")

# 2. Add CORS so the React frontend (running on port 5173) can talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Define Pydantic Models for requests
class StressTestRequest(BaseModel):
    borrower_id: str
    sales_shock: float
    cost_shock: float
    interest_shock: float
    opex_shock: float

# 4. Define the endpoints

@app.get("/api/risk/{borrower_id}")
async def get_risk(borrower_id: str):
    # Dynamic risk based on borrower selection
    if borrower_id == "DEMO-002":
        return {
            "pd": 0.35,
            "risk_segment": "Medium Risk",
            "decision_support": "Manual Review Recommended",
            "top_contributions": [
                {"feature": "debt_to_income", "contribution": 0.25, "value": "0.65"},
                {"feature": "cash_balance", "contribution": -0.05, "value": "$12,000"}
            ],
            "reason_codes": ["High debt-to-income ratio", "Lower cash reserves"]
        }
    elif borrower_id == "DEMO-003":
        return {
            "pd": 0.05,
            "risk_segment": "Low Risk",
            "decision_support": "Approved for Auto-Processing",
            "top_contributions": [
                {"feature": "cash_balance", "contribution": -0.20, "value": "$150,000"},
                {"feature": "revenue_volatility", "contribution": -0.15, "value": "Low"}
            ],
            "reason_codes": ["Strong cash reserves", "Stable revenue history"]
        }
    
    # Default for DEMO-001
    return {
        "pd": 0.18,
        "risk_segment": "Medium Risk",
        "decision_support": "Manual Review Recommended",
        "top_contributions": [
            {"feature": "debt_to_income", "contribution": 0.15, "value": "0.45"},
            {"feature": "cash_balance", "contribution": -0.12, "value": "$25,400"},
            {"feature": "late_payment_count", "contribution": 0.08, "value": "2"}
        ],
        "reason_codes": [
            "Increased risk due to high debt-to-income ratio",
            "Decreased risk due to healthy cash balance",
            "Increased risk due to recent late payments"
        ]
    }

@app.get("/api/evidence/{borrower_id}")
async def get_evidence(borrower_id: str):
    if borrower_id == "DEMO-003":
        return {
            "status": "Verified",
            "alerts": [],
            "checks": [
                {"relationship": "Invoice vs Bank Deposit", "status": "Verified", "difference": 0},
                {"relationship": "Declared Revenue vs GST", "status": "Verified", "difference": 0},
            ]
        }
    
    return {
        "status": "Needs Review",
        "alerts": ["GST filings do not match bank settlements for Q3"],
        "checks": [
            {"relationship": "Invoice vs Bank Deposit", "status": "Verified", "difference": 0},
            {"relationship": "Declared Revenue vs GST", "status": "Mismatch", "difference": 4500},
            {"relationship": "Identity vs Bureau", "status": "Verified", "difference": None}
        ]
    }

@app.get("/api/forecast/{borrower_id}")
async def get_forecast(borrower_id: str):
    return {
        "30d": {"expected_inflow": 45000, "expected_outflow": 32000, "expected_net_cash_flow": 13000, "projected_cash_balance": 38000, "debt_service_capacity_dscr": 1.8, "liquidity_pressure": "Low"},
        "60d": {"expected_inflow": 42000, "expected_outflow": 35000, "expected_net_cash_flow": 7000, "projected_cash_balance": 45000, "debt_service_capacity_dscr": 1.4, "liquidity_pressure": "Medium"},
        "90d": {"expected_inflow": 38000, "expected_outflow": 40000, "expected_net_cash_flow": -2000, "projected_cash_balance": 43000, "debt_service_capacity_dscr": 0.9, "liquidity_pressure": "High"}
    }

@app.get("/api/provenance/{borrower_id}")
async def get_provenance(borrower_id: str):
    return {
        "trace": [
            {"source": "Invoice INV-2023-089", "target": "Bank Settlement #9021", "match": True, "amount_diff": 0},
            {"source": "Reported Q3 Revenue", "target": "GST Tax Filing Q3", "match": False, "amount_diff": 4500}
        ]
    }

@app.get("/api/fairness")
async def get_fairness():
    return {
        "baseline": {
            "fairness": {
                "disparities": {"equal_opportunity_difference": 0.12},
                "group_metrics": {
                    "Male": {"tpr": 0.85, "fpr": 0.15},
                    "Female": {"tpr": 0.73, "fpr": 0.10}
                }
            }
        },
        "mitigated": {
            "fairness": {
                "disparities": {"equalized_odds_difference": 0.03},
                "group_metrics": {
                    "Male": {"tpr": 0.80, "fpr": 0.13},
                    "Female": {"tpr": 0.78, "fpr": 0.12}
                }
            }
        },
        "improvement": {"equal_opportunity_diff_change": 0.09}
    }

@app.get("/api/profile/{borrower_id}")
async def get_profile(borrower_id: str):
    return {
        "business_type": "Retail Services",
        "years_in_business": 4.5,
        "monthly_revenue": 45000.00,
        "cash_balance": 25400.00,
        "debt_to_income": 0.45,
        "late_payment_count": 2,
        "industry_risk_score": 1.2
    }

# --- THE DYNAMIC STRESS TEST ENDPOINT ---
@app.post("/api/stress-test")
@app.post("/stress-test")
async def execute_stress_test(request: StressTestRequest):
    try:
        # Dynamically set the starting risk based on the dropdown choice!
        if request.borrower_id == "DEMO-002":
            base_pd = 0.35  # Higher risk borrower
            base_revenue = 20000
        elif request.borrower_id == "DEMO-003":
            base_pd = 0.05  # Very low risk borrower
            base_revenue = 150000
        else:
            base_pd = 0.18  # DEMO-001 (Default)
            base_revenue = 50000
            
        # Calculate dynamic impact based on the sliders
        pd_increase_from_sales = abs(request.sales_shock) * 0.6  
        pd_increase_from_cost = request.cost_shock * 0.3
        pd_increase_from_interest = request.interest_shock * 0.8
        pd_increase_from_opex = request.opex_shock * 0.2
        
        total_shock = pd_increase_from_sales + pd_increase_from_cost + pd_increase_from_interest + pd_increase_from_opex
        stressed_pd = min(base_pd + total_shock, 0.99)
        
        # Calculate financial dollar impact using the specific borrower's baseline revenue
        financial_impact = -abs((request.sales_shock * base_revenue) + (request.cost_shock * (base_revenue * 0.4)))
        
        def get_segment(pd_value):
            if pd_value < 0.20: return "Low Risk"
            if pd_value < 0.40: return "Medium Risk"
            return "High Risk"
            
        return {
            "baseline": {
                "pd": base_pd,
                "risk_segment": get_segment(base_pd)
            },
            "stressed": {
                "pd": stressed_pd,
                "risk_segment": get_segment(stressed_pd)
            },
            "impact": {
                "pd_change": stressed_pd - base_pd,
                "net_cash_flow_impact": financial_impact
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))