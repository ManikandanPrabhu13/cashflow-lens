import React, { useEffect, useState } from 'react';
import { User } from 'lucide-react';
import { Card } from '../components/common/Card';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorAlert } from '../components/common/ErrorAlert';
import { api } from '../services/api';

export const BorrowerProfile: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [details, setDetails] = useState<Record<string, any> | null>(null);

  const borrowerId = "DEMO-001";

  const fetchDetails = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getBorrowerDetails(borrowerId);
      setDetails(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load borrower profile');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDetails();
  }, [borrowerId]);

  if (loading) return <LoadingSpinner text="Loading Borrower Profile..." />;
  if (error) return <ErrorAlert message={error} onRetry={fetchDetails} />;
  if (!details) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 bg-white p-6 rounded-xl shadow-sm border border-gray-200">
        <div className="p-2 bg-brand-50 rounded-lg">
          <User className="w-6 h-6 text-brand-600" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-gray-900">Borrower Profile: {borrowerId}</h2>
          <p className="text-sm text-gray-500">Raw operational and financial metadata</p>
        </div>
      </div>

      <Card title="Model Input Features (Raw Data)">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {Object.entries(details).map(([key, value]) => (
            <div key={key} className="bg-gray-50 p-4 rounded-lg border border-gray-100 break-words">
              <p className="text-xs text-gray-500 font-semibold uppercase tracking-wider mb-1">
                {key.replace(/_/g, ' ')}
              </p>
              <p className="text-base font-bold text-gray-900">
                {typeof value === 'number' && !Number.isInteger(value) 
                  ? value.toFixed(2) 
                  : String(value)}
              </p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};