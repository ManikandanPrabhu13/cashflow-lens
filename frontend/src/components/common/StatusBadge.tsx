import React from 'react';

interface StatusBadgeProps {
  status: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status }) => {
  const getBadgeStyles = (status: string) => {
    const s = status.toLowerCase();
    
    // Risk Segments
    if (s.includes('low risk')) return 'bg-emerald-100 text-emerald-800 border-emerald-200';
    if (s.includes('medium risk')) return 'bg-amber-100 text-amber-800 border-amber-200';
    if (s.includes('high risk')) return 'bg-rose-100 text-rose-800 border-rose-200';
    
    // Evidence / General Statuses
    if (s === 'verified') return 'bg-blue-100 text-blue-800 border-blue-200';
    if (s === 'mismatch') return 'bg-red-100 text-red-800 border-red-200';
    if (s === 'needs review') return 'bg-orange-100 text-orange-800 border-orange-200';
    if (s === 'no evidence') return 'bg-gray-100 text-gray-600 border-gray-200';
    
    // Decision Support
    if (s.includes('continue monitoring')) return 'bg-teal-100 text-teal-800 border-teal-200';
    if (s.includes('manual review') || s.includes('additional evidence')) return 'bg-purple-100 text-purple-800 border-purple-200';
    
    // Default
    return 'bg-gray-100 text-gray-800 border-gray-200';
  };

  return (
    <span className={`px-2.5 py-1 rounded-full text-xs font-medium border ${getBadgeStyles(status)}`}>
      {status}
    </span>
  );
};