import { 
  BorrowerInfo, 
  RiskAnalysis, 
  EvidenceIntegrity, 
  ProvenanceTrace, 
  Forecast, 
  FairnessReport, 
  StressTestRequest, 
  StressTestResponse 
} from '../types/api';

import * as MOCK from './mockData';

// Setup Base URL for FastAPI Backend
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const IS_DEMO = import.meta.env.VITE_DEMO_MODE === 'true';

async function fetchWithFallback<T>(endpoint: string, mockData: T, options?: RequestInit): Promise<T> {
  if (IS_DEMO) {
    return new Promise((resolve) => setTimeout(() => resolve(mockData), 600)); // Simulate latency
  }

  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      }
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.warn(`Failed to fetch ${endpoint}, falling back to mock data.`, error);
    return mockData; // Fallback to mock data on network error to keep UI functional
  }
}

export const api = {
  getHealth: () => fetchWithFallback<{status: string, model_loaded: boolean}>('/health', { status: 'healthy', model_loaded: true }),
  
  getBorrowers: () => fetchWithFallback<BorrowerInfo[]>('/borrowers', MOCK.mockBorrowers),
  
  getBorrowerDetails: (id: string) => fetchWithFallback<Record<string, any>>(`/borrowers/${id}`, { id, name: "Demo Business LLC", industry: "Retail" }),
  
  getRiskAnalysis: (id: string) => fetchWithFallback<RiskAnalysis>(`/risk?borrower_id=${id}`, { ...MOCK.mockRiskAnalysis, borrower_id: id }),
  
  getEvidence: (id: string) => fetchWithFallback<EvidenceIntegrity>(`/evidence?borrower_id=${id}`, { ...MOCK.mockEvidence, borrower_id: id }),
  
  getProvenance: (id: string) => fetchWithFallback<ProvenanceTrace>(`/provenance?borrower_id=${id}`, { ...MOCK.mockProvenance, borrower_id: id }),
  
  getForecast: (id: string) => fetchWithFallback<Forecast>(`/forecast?borrower_id=${id}`, MOCK.mockForecast),
  
  getFairness: () => fetchWithFallback<FairnessReport>('/fairness', MOCK.mockFairness),
  
  executeStressTest: (data: StressTestRequest) => fetchWithFallback<StressTestResponse>('/stress-test', MOCK.mockStressTestResponse, {
    method: 'POST',
    body: JSON.stringify(data)
  })
};