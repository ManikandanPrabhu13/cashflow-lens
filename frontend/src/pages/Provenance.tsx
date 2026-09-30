import React, { useEffect, useState } from 'react';
import { GitMerge, ArrowRight, Database, FileText, CheckCircle2, XCircle } from 'lucide-react';
import { Card } from '../components/common/Card';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorAlert } from '../components/common/ErrorAlert';
import { api } from '../services/api';
import { ProvenanceTrace } from '../types/api';

export const Provenance: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [provenance, setProvenance] = useState<ProvenanceTrace | null>(null);

  const borrowerId = "DEMO-001";

  const fetchProvenance = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getProvenance(borrowerId);
      setProvenance(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load provenance trace');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProvenance();
  }, [borrowerId]);

  if (loading) return <LoadingSpinner text="Tracing Data Lineage..." />;
  if (error) return <ErrorAlert message={error} onRetry={fetchProvenance} />;
  if (!provenance || provenance.trace.length === 0) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 bg-white p-6 rounded-xl shadow-sm border border-gray-200">
        <div className="p-2 bg-brand-50 rounded-lg">
          <GitMerge className="w-6 h-6 text-brand-600" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-gray-900">Data Lineage Trace</h2>
          <p className="text-sm text-gray-500">Cryptographic-style mapping of financial claims to raw data sources</p>
        </div>
      </div>

      <Card>
        <div className="py-8 px-4 md:px-12">
          <div className="relative border-l-2 border-gray-200 ml-4 md:ml-6 space-y-12">
            {provenance.trace.map((node, idx) => (
              <div key={idx} className="relative">
                {/* Node Source */}
                <div className="absolute -left-[25px] flex items-center justify-center w-12 h-12 rounded-full bg-white border-2 border-brand-500 shadow-sm">
                  <FileText className="w-5 h-5 text-brand-600" />
                </div>
                
                <div className="pl-12 flex flex-col md:flex-row md:items-center gap-4 md:gap-8">
                  {/* Source to Target Logic */}
                  <div className="flex-1 bg-gray-50 border border-gray-200 p-4 rounded-lg">
                    <p className="text-xs text-gray-500 font-semibold uppercase tracking-wider mb-1">Source Record</p>
                    <p className="text-base font-bold text-gray-900">{node.source}</p>
                  </div>

                  <div className="flex items-center justify-center text-gray-400 hidden md:flex">
                    <ArrowRight className="w-6 h-6" />
                  </div>

                  <div className="flex-1 bg-gray-50 border border-gray-200 p-4 rounded-lg">
                    <p className="text-xs text-gray-500 font-semibold uppercase tracking-wider mb-1">Target Validation</p>
                    <p className="text-base font-bold text-gray-900 flex items-center gap-2">
                      <Database className="w-4 h-4 text-gray-400" />
                      {node.target}
                    </p>
                  </div>

                  {/* Outcome Tag */}
                  <div className="flex shrink-0 items-center">
                    {node.match ? (
                      <div className="flex items-center gap-2 text-emerald-700 bg-emerald-50 px-4 py-2 rounded-full border border-emerald-200">
                        <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                        <span className="font-semibold text-sm">Matched</span>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 text-rose-700 bg-rose-50 px-4 py-2 rounded-full border border-rose-200">
                        <XCircle className="w-5 h-5 text-rose-500" />
                        <span className="font-semibold text-sm">
                          Mismatch (${node.amount_diff.toLocaleString()})
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}

            {/* Final validation endpoint */}
            <div className="relative mt-12">
              <div className="absolute -left-[25px] flex items-center justify-center w-12 h-12 rounded-full bg-gray-100 border-2 border-gray-300">
                <Database className="w-5 h-5 text-gray-500" />
              </div>
              <div className="pl-12 pt-3">
                <p className="text-sm font-semibold text-gray-500">End of Provenance Chain</p>
              </div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};