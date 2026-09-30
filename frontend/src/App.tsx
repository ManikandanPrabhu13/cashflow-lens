import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { MainLayout } from './layouts/MainLayout';
import { Dashboard } from './pages/Dashboard';
import { RiskAnalytics } from './pages/RiskAnalytics';
import { CashFlow } from './pages/CashFlow';
import { EvidenceIntegrity } from './pages/EvidenceIntegrity';
import { Provenance } from './pages/Provenance';
import { Fairness } from './pages/Fairness';
import { StressTesting } from './pages/StressTesting';
import { BorrowerProfile } from './pages/BorrowerProfile';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="risk" element={<RiskAnalytics />} />
          <Route path="cashflow" element={<CashFlow />} />
          <Route path="evidence" element={<EvidenceIntegrity />} />
          <Route path="provenance" element={<Provenance />} />
          <Route path="fairness" element={<Fairness />} />
          <Route path="stress-test" element={<StressTesting />} />
          <Route path="profile" element={<BorrowerProfile />} />
          
          {/* Catch-all redirect */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;