import React from 'react';
import { Link, useLocation, Outlet } from 'react-router-dom';
import { 
  LayoutDashboard, 
  TrendingDown, 
  Wallet, 
  FileCheck, 
  GitMerge, 
  Scale, 
  ActivitySquare
} from 'lucide-react';

const IS_DEMO = import.meta.env.VITE_DEMO_MODE === 'true';

const navItems = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/risk', label: 'Risk Analytics', icon: TrendingDown },
  { path: '/cashflow', label: 'Cash Flow', icon: Wallet },
  { path: '/evidence', label: 'Evidence Integrity', icon: FileCheck },
  { path: '/provenance', label: 'Provenance', icon: GitMerge },
  { path: '/fairness', label: 'Fairness Audit', icon: Scale },
  { path: '/stress-test', label: 'Stress Testing', icon: ActivitySquare },
];

export const MainLayout: React.FC = () => {
  const location = useLocation();

  return (
    <div className="flex h-screen bg-gray-50 font-sans">
      {/* Sidebar */}
      <aside className="w-64 bg-slate-900 text-white flex flex-col shrink-0">
        <div className="p-6">
          <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            <span className="w-8 h-8 rounded bg-brand-500 flex items-center justify-center text-white">
              C
            </span>
            CashFlow-Lens
          </h1>
          <p className="text-slate-400 text-xs mt-2 uppercase tracking-wider font-semibold">
            MSME Risk Analytics
          </p>
        </div>

        <nav className="flex-1 px-4 space-y-1 mt-4 overflow-y-auto">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            const Icon = item.icon;
            
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors text-sm font-medium ${
                  isActive 
                    ? 'bg-brand-600 text-white' 
                    : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                }`}
              >
                <Icon className="w-4 h-4" />
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="p-4 border-t border-slate-800">
          <div className="bg-slate-800 rounded-lg p-3 text-xs text-slate-300">
            System Status: <span className="text-emerald-400 font-medium float-right">Online</span>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Top Header */}
        <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-8 shrink-0 shadow-sm z-10">
          <div className="flex items-center">
            <h2 className="text-lg font-semibold text-gray-800">
              {navItems.find(item => item.path === location.pathname)?.label || 'Dashboard'}
            </h2>
          </div>
          
          <div className="flex items-center gap-4">
            {IS_DEMO && (
              <div className="flex items-center gap-2 px-3 py-1 bg-amber-100 text-amber-800 rounded-full text-xs font-bold uppercase tracking-wide border border-amber-200">
                <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse"></span>
                Demo Data
              </div>
            )}
            
            <select className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-brand-500 focus:border-brand-500 block w-full p-2.5 font-medium">
              <option value="DEMO-001">Borrower: DEMO-001</option>
              <option value="DEMO-002">Borrower: DEMO-002</option>
              <option value="DEMO-003">Borrower: DEMO-003</option>
            </select>
          </div>
        </header>

        {/* Scrollable Page Content */}
        <div className="flex-1 overflow-auto p-8 bg-gray-50">
          <Outlet />
        </div>
      </main>
    </div>
  );
};