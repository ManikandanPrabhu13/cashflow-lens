import React, { useState } from 'react';
import { ActivitySquare, Play, ArrowRight, TrendingDown, TrendingUp } from 'lucide-react';
import { Card } from '../components/common/Card';
import { StatusBadge } from '../components/common/StatusBadge';
import { ErrorAlert } from '../components/common/ErrorAlert';
import { api } from '../services/api';
import { StressTestResponse } from '../types/api';

export const StressTesting: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<StressTestResponse | null>(null);

  const [shocks, setShocks] = useState({
    sales_shock: 0,
    cost_shock: 0,
    interest_shock: 0,
    opex_shock: 0
  });

  const borrowerId = localStorage.getItem('borrowerId') || "DEMO-001";

  const handleRunStressTest = async () => {
    try {
      setLoading(true);
      setError(null);
      // Convert percentages to decimals for the API
      const payload = {
        borrower_id: borrowerId,
        sales_shock: shocks.sales_shock / 100,
        cost_shock: shocks.cost_shock / 100,
        interest_shock: shocks.interest_shock / 100,
        opex_shock: shocks.opex_shock / 100
      };
      
      const data = await api.executeStressTest(payload);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to execute stress test');
    } finally {
      setLoading(false);
    }
  };

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setShocks(prev => ({ ...prev, [name]: Number(value) }));
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 bg-white p-6 rounded-xl shadow-sm border border-gray-200">
        <div className="p-2 bg-brand-50 rounded-lg">
          <ActivitySquare className="w-6 h-6 text-brand-600" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-gray-900">Macroeconomic Stress Testing</h2>
          <p className="text-sm text-gray-500">Apply hypothetical shocks to borrower cash flows to observe risk migration</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Controls Panel */}
        <div className="lg:col-span-4 space-y-6">
          <Card title="Scenario Parameters">
            <div className="space-y-6">
              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-sm font-medium text-gray-700">Sales / Revenue Shock</label>
                  <span className="text-sm font-bold text-gray-900">{shocks.sales_shock}%</span>
                </div>
                <input 
                  type="range" name="sales_shock" min="-50" max="0" step="5"
                  value={shocks.sales_shock} onChange={handleSliderChange}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-rose-500"
                />
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-sm font-medium text-gray-700">Input Cost Shock</label>
                  <span className="text-sm font-bold text-gray-900">+{shocks.cost_shock}%</span>
                </div>
                <input 
                  type="range" name="cost_shock" min="0" max="50" step="5"
                  value={shocks.cost_shock} onChange={handleSliderChange}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-orange-500"
                />
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-sm font-medium text-gray-700">Interest Rate Shock</label>
                  <span className="text-sm font-bold text-gray-900">+{shocks.interest_shock}%</span>
                </div>
                <input 
                  type="range" name="interest_shock" min="0" max="10" step="1"
                  value={shocks.interest_shock} onChange={handleSliderChange}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-500"
                />
              </div>

              <button
                onClick={handleRunStressTest}
                disabled={loading}
                className="w-full flex items-center justify-center gap-2 bg-brand-600 hover:bg-brand-700 text-white py-2.5 px-4 rounded-lg font-medium transition-colors disabled:opacity-50"
              >
                {loading ? (
                  <span className="flex items-center gap-2">Running...</span>
                ) : (
                  <><Play className="w-4 h-4" /> Execute Scenario</>
                )}
              </button>
            </div>
          </Card>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-8">
          {error && <ErrorAlert message={error} className="mb-6" />}
          
          {!result && !loading && !error && (
            <div className="h-full border-2 border-dashed border-gray-200 rounded-xl flex items-center justify-center bg-gray-50/50 min-h-[300px]">
              <p className="text-gray-400 font-medium">Adjust parameters and execute to view stress impact</p>
            </div>
          )}

          {result && (
            <div className="space-y-6 animate-in fade-in duration-300">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card title="Baseline Risk" className="border-gray-200">
                  <div className="text-center py-4">
                    <p className="text-sm text-gray-500 mb-2">Original Probability of Default</p>
                    <p className="text-3xl font-bold text-gray-900 mb-4">
                      {(result.baseline.pd * 100).toFixed(2)}%
                    </p>
                    <StatusBadge status={result.baseline.risk_segment} />
                  </div>
                </Card>

                <Card title="Stressed Risk" className={`border-2 ${result.impact.pd_change > 0.1 ? 'border-rose-300 bg-rose-50/10' : 'border-amber-300 bg-amber-50/10'}`}>
                  <div className="text-center py-4">
                    <p className="text-sm text-gray-500 mb-2">Scenario Probability of Default</p>
                    <div className="flex items-center justify-center gap-2 mb-4">
                      <p className="text-3xl font-bold text-gray-900">
                        {(result.stressed.pd * 100).toFixed(2)}%
                      </p>
                      {result.impact.pd_change > 0 && (
                        <span className="text-sm font-medium text-rose-600 flex items-center bg-rose-100 px-2 py-0.5 rounded">
                          <TrendingUp className="w-3 h-3 mr-1" />
                          +{(result.impact.pd_change * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>
                    <StatusBadge status={result.stressed.risk_segment} />
                  </div>
                </Card>
              </div>

              <Card title="Financial Impact Summary">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
                    <p className="text-sm text-gray-500">Net Cash Flow Impact</p>
                    <p className={`text-xl font-bold mt-1 ${result.impact.net_cash_flow_impact < 0 ? 'text-rose-600' : 'text-gray-900'}`}>
                      {result.impact.net_cash_flow_impact < 0 ? '-' : ''}${Math.abs(result.impact.net_cash_flow_impact).toLocaleString()}
                    </p>
                  </div>
                  
                  <div className="p-4 bg-gray-50 rounded-lg border border-gray-100">
                    <p className="text-sm text-gray-500">Risk Migration</p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="font-medium text-gray-900">{result.baseline.risk_segment}</span>
                      <ArrowRight className="w-4 h-4 text-gray-400" />
                      <span className={`font-medium ${result.baseline.risk_segment !== result.stressed.risk_segment ? 'text-rose-600' : 'text-gray-900'}`}>
                        {result.stressed.risk_segment}
                      </span>
                    </div>
                  </div>
                </div>
              </Card>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};