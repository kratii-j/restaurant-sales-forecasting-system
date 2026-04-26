import { useEffect, useState } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'
import { api } from '../api'
import KpiCard from '../components/KpiCard'

const TT = { background: '#0a0a0a', border: '1px solid #1a1a1a', fontFamily: 'JetBrains Mono', fontSize: 12 }

export default function Dashboard() {
  const [summary, setSummary] = useState<any>(null)
  const [trends, setTrends] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([api.dashboard.summary(), api.dashboard.trends(90)])
      .then(([s, t]) => { setSummary(s); setTrends(t) })
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="page-loading">loading...</div>

  return (
    <div className="page">
      <h1 className="page-title">Dashboard</h1>

      <div className="kpi-grid">
        <KpiCard label="Total Restaurants" value={summary?.total_restaurants?.toLocaleString() ?? '—'} />
        <KpiCard label="Avg Daily Orders" value={summary?.avg_daily_orders?.toFixed(1) ?? '—'} />
        <KpiCard label="Avg Daily Revenue" value={`₹${summary?.avg_daily_revenue?.toFixed(0) ?? '—'}`} />
        <KpiCard label="High Risk" value={summary?.high_risk_count ?? 0} accent="var(--danger)" />
      </div>

      <div className="chart-section">
        <h2 className="section-title">Orders Trend (90d)</h2>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={trends}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
              <XAxis dataKey="date" stroke="#666" fontSize={11} tickFormatter={v => v.slice(5)} />
              <YAxis stroke="#666" fontSize={11} />
              <Tooltip contentStyle={TT} labelStyle={{ color: '#999' }} />
              <Area type="monotone" dataKey="avg_orders" stroke="#10b981" fill="#10b981" fillOpacity={0.1} strokeWidth={1.5} name="Avg Orders" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="chart-section">
        <h2 className="section-title">Revenue Trend (90d)</h2>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={trends}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
              <XAxis dataKey="date" stroke="#666" fontSize={11} tickFormatter={v => v.slice(5)} />
              <YAxis stroke="#666" fontSize={11} />
              <Tooltip contentStyle={TT} labelStyle={{ color: '#999' }} />
              <Area type="monotone" dataKey="avg_revenue" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.1} strokeWidth={1.5} name="Avg Revenue" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="stats-row">
        <div className="stat-card">
          <span className="stat-label">Low Risk</span>
          <span className="stat-value accent">{summary?.low_risk_count}</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Medium Risk</span>
          <span className="stat-value warning">{summary?.medium_risk_count}</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">High Risk</span>
          <span className="stat-value danger">{summary?.high_risk_count}</span>
        </div>
      </div>
    </div>
  )
}
