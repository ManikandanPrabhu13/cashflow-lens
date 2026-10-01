import React, { useEffect, useState } from 'react';
import { ShieldAlert, TrendingUp, Activity, FileCheck } from 'lucide-react';
import { KPICard } from '../components/kpi/KPICard';
import { Card } from '../components/common/Card';
import { StatusBadge } from '../components/common/StatusBadge';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorAlert } from '../components/common/ErrorAlert';
import { api } from '../services/api';
import { RiskAnalysis, EvidenceIntegrity, Forecast } from '../types/api';

export const Dashboard: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<{
    risk: RiskAnalysis | null;
    evidence: EvidenceIntegrity | null;
    forecast: Forecast | null;
  }>({ risk: null, evidence: null, forecast: null });

  // Defaulting to DEMO-001 for the dashboard view
 const borrowerId = localStorage.getItem('borrowerId') || "DEMO-001";

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const [risk, evidence, forecast] = await Promise.all([
        api.getRiskAnalysis(borrowerId),
        api.getEvidence(borrowerId),
        api.getForecast(borrowerId)
      ]);
      
      setData({ risk, evidence, forecast });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [borrowerId]);

  if (loading) return <LoadingSpinner text="Compiling Dashboard Analytics..." />;
  if (error) return <ErrorAlert message={error} onRetry={fetchData} />;
  if (!data.risk || !data.forecast || !data.evidence) return null;

  return (
    <div className="space-y-6">
      {/* Top KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="flex flex-col justify-center">
          <div className="flex justify-between items-start mb-2">
            <h3 className="text-sm font-medium text-gray-500">Predicted PD</h3>
            <ShieldAlert className="w-5 h-5 text-brand-600" />
          </div>
          <div className="text-3xl font-bold text-gray-900 mb-2">
            {(data.risk.pd * 100).toFixed(1)}%
          </div>
          <div>
            <StatusBadge status={data.risk.risk_segment} />
          </div>
        </Card>

        <KPICard 
          title="Monthly Net Cash Flow"
          value={`$${data.forecast['30d'].expected_net_cash_flow.toLocaleString()}`}
          icon={TrendingUp}
          trend={{ value: 12.5, isPositive: true, label: "vs last month" }}
        />

        <KPICard 
          title="Liquidity Pressure"
          value={data.forecast['30d'].liquidity_pressure}
          icon={Activity}
          subtitle={`DSCR: ${data.forecast['30d'].debt_service_capacity_dscr.toFixed(2)}x`}
          valueColor={data.forecast['30d'].liquidity_pressure === 'High' ? 'text-rose-600' : 'text-emerald-600'}
        />

        <Card className="flex flex-col justify-center">
          <div className="flex justify-between items-start mb-2">
            <h3 className="text-sm font-medium text-gray-500">Evidence Status</h3>
            <FileCheck className="w-5 h-5 text-brand-600" />
          </div>
          <div className="text-xl font-bold text-gray-900 mb-2">
            {data.evidence.checks.filter(c => c.status === 'Verified').length} / {data.evidence.checks.length} Verified
          </div>
          <div>
            <StatusBadge status={data.evidence.status} />
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Decision Support Summary */}
        <Card title="Underwriter Decision Support">
          <div className="mb-6">
            <p className="text-sm text-gray-500 mb-1">System Recommendation</p>
            <div className="text-lg font-semibold flex items-center gap-3">
              <StatusBadge status={data.risk.decision_support} />
            </div>
          </div>
          
          <div>
            <p className="text-sm text-gray-500 mb-3">Key Risk Drivers (Local SHAP)</p>
            <ul className="space-y-3">
              {data.risk.reason_codes.map((code, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm text-gray-700 bg-gray-50 p-3 rounded-lg border border-gray-100">
                  <span className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${code.includes('Increased') ? 'bg-rose-500' : 'bg-emerald-500'}`} />
                  {code}
                </li>
              ))}
            </ul>
          </div>
        </Card>

        {/* 90-Day Outlook */}
        <Card title="90-Day Liquidity Outlook">
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="text-xs text-gray-500 uppercase bg-gray-50">
                <tr>
                  <th className="px-4 py-3 rounded-tl-lg">Metric</th>
                  <th className="px-4 py-3 text-right">30 Days</th>
                  <th className="px-4 py-3 text-right">60 Days</th>
                  <th className="px-4 py-3 text-right rounded-tr-lg">90 Days</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-gray-100">
                  <td className="px-4 py-4 font-medium text-gray-900">Projected Inflow</td>
                  <td className="px-4 py-4 text-right text-emerald-600">${data.forecast['30d'].expected_inflow.toLocaleString()}</td>
                  <td className="px-4 py-4 text-right text-emerald-600">${data.forecast['60d'].expected_inflow.toLocaleString()}</td>
                  <td className="px-4 py-4 text-right text-emerald-600">${data.forecast['90d'].expected_inflow.toLocaleString()}</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="px-4 py-4 font-medium text-gray-900">Projected Outflow</td>
                  <td className="px-4 py-4 text-right text-rose-600">${data.forecast['30d'].expected_outflow.toLocaleString()}</td>
                  <td className="px-4 py-4 text-right text-rose-600">${data.forecast['60d'].expected_outflow.toLocaleString()}</td>
                  <td className="px-4 py-4 text-right text-rose-600">${data.forecast['90d'].expected_outflow.toLocaleString()}</td>
                </tr>
                <tr className="bg-gray-50/50 font-semibold">
                  <td className="px-4 py-4 text-gray-900">Net Cash Balance</td>
                  <td className="px-4 py-4 text-right">${data.forecast['30d'].projected_cash_balance.toLocaleString()}</td>
                  <td className="px-4 py-4 text-right">${data.forecast['60d'].projected_cash_balance.toLocaleString()}</td>
                  <td className="px-4 py-4 text-right">${data.forecast['90d'].projected_cash_balance.toLocaleString()}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
};