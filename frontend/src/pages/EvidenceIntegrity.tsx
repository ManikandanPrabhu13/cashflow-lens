import React, { useEffect, useState } from 'react';
import { ShieldCheck, ShieldAlert, AlertTriangle, FileX, CheckCircle2 } from 'lucide-react';
import { Card } from '../components/common/Card';
import { StatusBadge } from '../components/common/StatusBadge';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorAlert } from '../components/common/ErrorAlert';
import { api } from '../services/api';
import { EvidenceIntegrity as EvidenceType } from '../types/api';

export const EvidenceIntegrity: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [evidence, setEvidence] = useState<EvidenceType | null>(null);

  const borrowerId = "DEMO-001";

  const fetchEvidence = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getEvidence(borrowerId);
      setEvidence(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load evidence data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEvidence();
  }, [borrowerId]);

  if (loading) return <LoadingSpinner text="Reconciling Documents..." />;
  if (error) return <ErrorAlert message={error} onRetry={fetchEvidence} />;
  if (!evidence) return null;

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'Verified': return <CheckCircle2 className="w-5 h-5 text-emerald-500" />;
      case 'Mismatch': return <ShieldAlert className="w-5 h-5 text-rose-500" />;
      case 'Needs Review': return <AlertTriangle className="w-5 h-5 text-orange-500" />;
      default: return <FileX className="w-5 h-5 text-gray-400" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Overview Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center bg-white p-6 rounded-xl shadow-sm border border-gray-200 gap-4">
        <div className="flex items-center gap-4">
          <div className={`p-3 rounded-xl ${evidence.status === 'Verified' ? 'bg-emerald-100' : 'bg-orange-100'}`}>
            {evidence.status === 'Verified' ? (
              <ShieldCheck className="w-8 h-8 text-emerald-600" />
            ) : (
              <ShieldAlert className="w-8 h-8 text-orange-600" />
            )}
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-900">Document Reconciliation</h2>
            <p className="text-sm text-gray-500 mt-1">Cross-referencing alternative data sources</p>
          </div>
        </div>
        <div className="text-right">
          <p className="text-sm text-gray-500 mb-2">Overall Integrity Status</p>
          <StatusBadge status={evidence.status} />
        </div>
      </div>

      {/* Alerts Section */}
      {evidence.alerts && evidence.alerts.length > 0 && (
        <div className="bg-orange-50 border border-orange-200 rounded-xl p-4 flex gap-3">
          <AlertTriangle className="w-5 h-5 text-orange-500 shrink-0 mt-0.5" />
          <div>
            <h4 className="text-sm font-semibold text-orange-800">Active Integrity Alerts</h4>
            <ul className="mt-2 space-y-1">
              {evidence.alerts.map((alert, idx) => (
                <li key={idx} className="text-sm text-orange-700 list-disc list-inside">
                  {alert}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Detail Checks */}
      <Card title="Relationship Verification Checks">
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs text-gray-500 uppercase bg-gray-50 border-y border-gray-100">
              <tr>
                <th className="px-6 py-4 font-medium">Data Relationship</th>
                <th className="px-6 py-4 font-medium">Match Status</th>
                <th className="px-6 py-4 font-medium text-right">Discrepancy Amount</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {evidence.checks.map((check, idx) => (
                <tr key={idx} className="hover:bg-gray-50/50 transition-colors">
                  <td className="px-6 py-4 font-medium text-gray-900">
                    {check.relationship}
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(check.status)}
                      <StatusBadge status={check.status} />
                    </div>
                  </td>
                  <td className="px-6 py-4 text-right text-gray-600">
                    {check.difference !== null && check.difference > 0 ? (
                      <span className="text-rose-600 font-medium">
                        ${check.difference.toLocaleString()}
                      </span>
                    ) : check.difference === 0 ? (
                      <span className="text-emerald-600 font-medium">$0.00</span>
                    ) : (
                      <span className="text-gray-400 italic">N/A</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
};