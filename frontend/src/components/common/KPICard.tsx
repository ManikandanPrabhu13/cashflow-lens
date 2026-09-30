import React from 'react';
import { LucideIcon } from 'lucide-react';

interface KPICardProps {
  title: string;
  value: string | number;
  icon: LucideIcon;
  subtitle?: string;
  trend?: {
    value: number;
    isPositive: boolean;
    label: string;
  };
  valueColor?: string;
}

export const KPICard: React.FC<KPICardProps> = ({ 
  title, 
  value, 
  icon: Icon, 
  subtitle,
  trend,
  valueColor = "text-gray-900"
}) => {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex flex-col h-full">
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-sm font-medium text-gray-500">{title}</h3>
        <div className="p-2 bg-brand-50 rounded-lg">
          <Icon className="w-5 h-5 text-brand-600" />
        </div>
      </div>
      
      <div className="mt-auto">
        <div className={`text-2xl font-bold ${valueColor}`}>{value}</div>
        
        {trend && (
          <div className="flex items-center mt-2 text-sm">
            <span className={`font-medium ${trend.isPositive ? 'text-emerald-600' : 'text-rose-600'}`}>
              {trend.isPositive ? '+' : '-'}{Math.abs(trend.value)}%
            </span>
            <span className="text-gray-400 ml-2">{trend.label}</span>
          </div>
        )}
        
        {subtitle && !trend && (
          <div className="text-sm text-gray-500 mt-2">{subtitle}</div>
        )}
      </div>
    </div>
  );
};