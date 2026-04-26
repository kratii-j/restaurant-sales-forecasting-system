import { LayoutDashboard, BarChart3, AlertTriangle, Store, Activity } from 'lucide-react'

type Page = 'dashboard' | 'forecasting' | 'risk' | 'restaurants' | 'models'

interface SidebarProps {
  active: Page
  onNavigate: (page: Page) => void
}

const navItems: { id: Page; label: string; icon: React.ReactNode }[] = [
  { id: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard size={16} /> },
  { id: 'forecasting', label: 'Forecasting', icon: <BarChart3 size={16} /> },
  { id: 'risk', label: 'Risk Analysis', icon: <AlertTriangle size={16} /> },
  { id: 'restaurants', label: 'Restaurants', icon: <Store size={16} /> },
  { id: 'models', label: 'Models', icon: <Activity size={16} /> },
]

export default function Sidebar({ active, onNavigate }: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <span className="logo-text">DineCast</span>
        <span className="logo-sub">forecasting</span>
      </div>
      <nav className="sidebar-nav">
        {navItems.map(item => (
          <button
            key={item.id}
            className={`nav-item ${active === item.id ? 'active' : ''}`}
            onClick={() => onNavigate(item.id)}
          >
            {item.icon}
            <span>{item.label}</span>
          </button>
        ))}
      </nav>
      <div className="sidebar-footer">
        <span className="version">v1.0.0</span>
      </div>
    </aside>
  )
}
