import React, { useEffect, useState } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell 
} from 'recharts';
import { Card } from '../components/common/Card';
import { StatusBadge } from '../components/common/StatusBadge';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorAlert } from '../components/common/ErrorAlert';
import { api } from '../services/api';
import { RiskAnalysis } from '../types/api';

export const RiskAnalytics: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [risk, setRisk] = useState<RiskAnalysis | null>(null);

  const borrowerId = "DEMO-001";

  const fetchRisk = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getRiskAnalysis(borrowerId);
      setRisk(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load risk data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRisk();
  }, [borrowerId]);

  if (loading) return <LoadingSpinner text="Loading Risk Analytics..." />;
  if (error) return <ErrorAlert message={error} onRetry={fetchRisk} />;
  if (!risk) return null;

  // Format data for Recharts
  const chartData = risk.top_contributions.map(item => ({
    name: item.feature.replace(/_/g, ' '),
    value: item.contribution,
    actualValue: item.value
  }));

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center bg-white p-6 rounded-xl shadow-sm border border-gray-200">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">{(risk.pd * 100).toFixed(2)}%</h2>
          <p className="text-sm text-gray-500 font-medium mt-1">Probability of Default (PD)</p>
        </div>
        <div className="text-right">
          <StatusBadge status={risk.risk_segment} />
          <p className="text-sm text-gray-500 font-medium mt-2">Model: LightGBM Primary</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Local Explainability (SHAP Values)">
          <p className="text-sm text-gray-500 mb-6">
            Displays the impact of specific features on this borrower's risk score. 
            Positive values increase risk, negative values decrease risk.
          </p>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                layout="vertical"
                data={chartData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" width={120} tick={{ fontSize: 12 }} />
                <Tooltip 
                  formatter={(value: number, name: string, props: any) => [
                    `${value.toFixed(4)} (Actual: ${props.payload.actualValue})`, 
                    'Impact'
                  ]}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#ef4444' : '#10b981'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Generated Reason Codes">
          <div className="space-y-4">
            {risk.reason_codes.map((code, idx) => {
              const isPositive = code.includes('Decreased');
              return (
                <div key={idx} className="p-4 rounded-lg border border-gray-100 bg-gray-50 flex gap-4 items-center">
                  <div className={`w-2 h-12 rounded-full ${isPositive ? 'bg-emerald-500' : 'bg-rose-500'}`} />
                  <div>
                    <h4 className={`font-semibold text-sm ${isPositive ? 'text-emerald-700' : 'text-rose-700'}`}>
                      {isPositive ? 'Risk Mitigator' : 'Risk Driver'}
                    </h4>
                    <p className="text-gray-700 text-sm mt-1">{code}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      </div>
    </div>
  );
};