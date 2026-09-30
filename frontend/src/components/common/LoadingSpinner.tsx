import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingSpinnerProps {
  text?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ text = "Loading data..." }) => {
  return (
    <div className="flex flex-col items-center justify-center p-12 w-full h-full min-h-[200px]">
      <Loader2 className="w-8 h-8 text-brand-600 animate-spin mb-4" />
      <p className="text-gray-500 font-medium">{text}</p>
    </div>
  );
};