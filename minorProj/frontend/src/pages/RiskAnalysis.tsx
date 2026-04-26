import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import { api } from '../api'
import KpiCard from '../components/KpiCard'
import DataTable from '../components/DataTable'
import StatusBadge from '../components/StatusBadge'

const COLORS: Record<string, string> = { Low: '#10b981', Medium: '#f59e0b', High: '#ef4444' }
const TT = { background: '#0a0a0a', border: '1px solid #1a1a1a', fontFamily: 'JetBrains Mono', fontSize: 12 }

export default function RiskAnalysis() {
  const [summary, setSummary] = useState<any>(null)
  const [restaurants, setRestaurants] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('')
  const [page, setPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)

  useEffect(() => { api.risk.summary().then(setSummary) }, [])

  useEffect(() => {
    setLoading(true)
    api.risk.restaurants({ level: filter || undefined, page, per_page: 20 })
      .then(res => { setRestaurants(res.items); setTotalPages(res.total_pages) })
      .finally(() => setLoading(false))
  }, [filter, page])

  const columns = [
    { key: 'restaurant_id', label: 'ID' },
    { key: 'mean_actual', label: 'Avg Orders', align: 'right' as const },
    { key: 'mean_predicted', label: 'Avg Predicted', align: 'right' as const },
    { key: 'rmse', label: 'RMSE', align: 'right' as const },
    { key: 'mae', label: 'MAE', align: 'right' as const },
    { key: 'risk_score', label: 'Risk Score', align: 'right' as const },
    { key: 'risk_level', label: 'Risk Level', align: 'center' as const, render: (v: string) => <StatusBadge level={v} /> },
  ]

  return (
    <div className="page">
      <h1 className="page-title">Risk Analysis</h1>

      {summary && (
        <>
          <div className="kpi-grid">
            <KpiCard label="Low Risk" value={summary.low} accent="var(--accent)" />
            <KpiCard label="Medium Risk" value={summary.medium} accent="var(--warning)" />
            <KpiCard label="High Risk" value={summary.high} accent="var(--danger)" />
          </div>

          <div className="chart-section">
            <h2 className="section-title">Risk Distribution</h2>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={summary.distribution}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
                  <XAxis dataKey="level" stroke="#666" fontSize={12} />
                  <YAxis stroke="#666" fontSize={11} />
                  <Tooltip contentStyle={TT} />
                  <Bar dataKey="count" name="Restaurants">
                    {summary.distribution.map((e: any, i: number) => (
                      <Cell key={i} fill={COLORS[e.level] || '#666'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

      <div className="table-section">
        <div className="table-header">
          <h2 className="section-title">Restaurants by Risk</h2>
          <select value={filter} onChange={e => { setFilter(e.target.value); setPage(1) }}>
            <option value="">All Levels</option>
            <option value="Low">Low</option>
            <option value="Medium">Medium</option>
            <option value="High">High</option>
          </select>
        </div>
        {loading ? <div className="page-loading">loading...</div> : <DataTable columns={columns} data={restaurants} />}
        <div className="pagination">
          <button disabled={page <= 1} onClick={() => setPage(p => p - 1)}>← Prev</button>
          <span className="page-info">{page} / {totalPages}</span>
          <button disabled={page >= totalPages} onClick={() => setPage(p => p + 1)}>Next →</button>
        </div>
      </div>
    </div>
  )
}
