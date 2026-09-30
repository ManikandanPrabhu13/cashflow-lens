import React, { useEffect, useState } from 'react';
import { Scale, Users, TrendingDown } from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import { Card } from '../components/common/Card';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorAlert } from '../components/common/ErrorAlert';
import { api } from '../services/api';
import { FairnessReport } from '../types/api';

export const Fairness: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<FairnessReport | null>(null);

  const fetchFairness = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getFairness();
      setReport(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load fairness report');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFairness();
  }, []);

  if (loading) return <LoadingSpinner text="Running Fairness Audit..." />;
  if (error) return <ErrorAlert message={error} onRetry={fetchFairness} />;
  if (!report) return null;

  const baselineGroups = report.baseline.fairness.group_metrics;
  const mitigatedGroups = report.mitigated.fairness.group_metrics;
  
  // Transform data for the comparison chart
  const chartData = Object.keys(baselineGroups).map(group => ({
    name: group,
    'Baseline TPR': Number((baselineGroups[group].tpr * 100).toFixed(1)),
    'Mitigated TPR': Number((mitigatedGroups[group].tpr * 100).toFixed(1)),
    'Baseline FPR': Number((baselineGroups[group].fpr * 100).toFixed(1)),
    'Mitigated FPR': Number((mitigatedGroups[group].fpr * 100).toFixed(1)),
  }));

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 bg-white p-6 rounded-xl shadow-sm border border-gray-200">
        <div className="p-2 bg-brand-50 rounded-lg">
          <Scale className="w-6 h-6 text-brand-600" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-gray-900">Model Fairness & Mitigation</h2>
          <p className="text-sm text-gray-500">Evaluating equalized odds and demographic parity across protected attributes</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <div className="flex items-center gap-2 mb-1">
            <Users className="w-4 h-4 text-gray-400" />
            <h3 className="text-sm font-medium text-gray-500">Equal Opportunity Diff</h3>
          </div>
          <div className="text-2xl font-bold text-gray-900">
            {(report.baseline.fairness.disparities.equal_opportunity_difference * 100).toFixed(2)}%
          </div>
          <p className="text-xs text-gray-500 mt-1">Baseline disparity in True Positive Rates</p>
        </Card>

        <Card>
          <div className="flex items-center gap-2 mb-1">
            <TrendingDown className="w-4 h-4 text-emerald-500" />
            <h3 className="text-sm font-medium text-gray-500">Mitigation Improvement</h3>
          </div>
          <div className="text-2xl font-bold text-emerald-600">
            -{(report.improvement.equal_opportunity_diff_change * 100).toFixed(2)}%
          </div>
          <p className="text-xs text-gray-500 mt-1">Reduction in True Positive Rate disparity</p>
        </Card>

        <Card>
          <div className="flex items-center gap-2 mb-1">
            <Scale className="w-4 h-4 text-gray-400" />
            <h3 className="text-sm font-medium text-gray-500">Equalized Odds Diff</h3>
          </div>
          <div className="text-2xl font-bold text-gray-900">
            {(report.mitigated.fairness.disparities.equalized_odds_difference * 100).toFixed(2)}%
          </div>
          <p className="text-xs text-gray-500 mt-1">Post-mitigation maximum disparity</p>
        </Card>
      </div>

      <Card title="True Positive Rate (TPR) - Baseline vs Mitigated">
        <div className="h-[350px] w-full mt-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" />
              <YAxis unit="%" />
              <Tooltip formatter={(value: number) => `${value}%`} />
              <Legend />
              <Bar dataKey="Baseline TPR" fill="#94a3b8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Mitigated TPR" fill="#14b8a6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  );
};