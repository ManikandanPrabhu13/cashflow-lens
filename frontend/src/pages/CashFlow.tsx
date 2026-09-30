import React, { useEffect, useState } from 'react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import { Card } from '../components/common/Card';
import { StatusBadge } from '../components/common/StatusBadge';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorAlert } from '../components/common/ErrorAlert';
import { api } from '../services/api';
import { Forecast } from '../types/api';

export const CashFlow: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [forecast, setForecast] = useState<Forecast | null>(null);

  const borrowerId = "DEMO-001";

  const fetchForecast = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getForecast(borrowerId);
      setForecast(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load cash flow data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchForecast();
  }, [borrowerId]);

  if (loading) return <LoadingSpinner text="Generating Forecast Models..." />;
  if (error) return <ErrorAlert message={error} onRetry={fetchForecast} />;
  if (!forecast) return null;

  // Map forecast data to Recharts format
  const chartData = [
    { name: '30 Days', Inflow: forecast['30d'].expected_inflow, Outflow: forecast['30d'].expected_outflow, Balance: forecast['30d'].projected_cash_balance },
    { name: '60 Days', Inflow: forecast['60d'].expected_inflow, Outflow: forecast['60d'].expected_outflow, Balance: forecast['60d'].projected_cash_balance },
    { name: '90 Days', Inflow: forecast['90d'].expected_inflow, Outflow: forecast['90d'].expected_outflow, Balance: forecast['90d'].projected_cash_balance },
  ];

  return (
    <div className="space-y-6">
      <Card title="Cash Flow & Balance Projection (90 Days)">
        <div className="h-[400px] w-full mt-4">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="colorInflow" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="colorOutflow" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <XAxis dataKey="name" />
              <YAxis />
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <Tooltip formatter={(value: number) => `$${value.toLocaleString()}`} />
              <Legend />
              <Area type="monotone" dataKey="Inflow" stroke="#10b981" fillOpacity={1} fill="url(#colorInflow)" />
              <Area type="monotone" dataKey="Outflow" stroke="#ef4444" fillOpacity={1} fill="url(#colorOutflow)" />
              <Area type="step" dataKey="Balance" stroke="#3b82f6" fill="none" strokeWidth={3} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {['30d', '60d', '90d'].map((period) => {
          const data = forecast[period as keyof Forecast];
          return (
            <Card key={period} title={`${period.toUpperCase()} Outlook`}>
              <div className="space-y-4">
                <div className="flex justify-between items-center border-b border-gray-100 pb-2">
                  <span className="text-gray-500 text-sm">Net Cash Flow</span>
                  <span className="font-semibold text-gray-900">${data.expected_net_cash_flow.toLocaleString()}</span>
                </div>
                <div className="flex justify-between items-center border-b border-gray-100 pb-2">
                  <span className="text-gray-500 text-sm">Debt Capacity (DSCR)</span>
                  <span className="font-semibold text-gray-900">{data.debt_service_capacity_dscr.toFixed(2)}x</span>
                </div>
                <div className="flex justify-between items-center pt-2">
                  <span className="text-gray-500 text-sm">Liquidity Pressure</span>
                  <StatusBadge status={data.liquidity_pressure} />
                </div>
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
};