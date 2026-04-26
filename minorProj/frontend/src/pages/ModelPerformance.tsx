import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import { api } from '../api'
import DataTable from '../components/DataTable'
import StatusBadge from '../components/StatusBadge'

const TT = { background: '#0a0a0a', border: '1px solid #1a1a1a', fontFamily: 'JetBrains Mono', fontSize: 12 }

export default function ModelPerformance() {
  const [comparison, setComparison] = useState<any[]>([])
  const [restaurants, setRestaurants] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)

  useEffect(() => {
    api.models.comparison().then(r => setComparison(r.items))
  }, [])

  useEffect(() => {
    api.models.restaurantPerformance({ page, per_page: 20 })
      .then(res => { setRestaurants(res.items); setTotalPages(res.total_pages) })
      .finally(() => setLoading(false))
  }, [page])

  // Chart data: only models with RMSE
  const chartData = comparison.filter(m => m.rmse != null).map(m => ({
    model: m.model.length > 18 ? m.model.slice(0, 18) + '…' : m.model,
    RMSE: m.rmse,
    MAE: m.mae,
  }))

  // Find best model
  const bestRmse = Math.min(...comparison.filter(m => m.rmse != null).map(m => m.rmse))

  const compColumns = [
    {
      key: 'model', label: 'Model',
      render: (v: string, row: any) => (
        <span className={row.rmse === bestRmse ? 'model-best' : ''}>{v} {row.rmse === bestRmse ? '★' : ''}</span>
      ),
    },
    { key: 'source', label: 'Source' },
    { key: 'rmse', label: 'RMSE', align: 'right' as const },
    { key: 'mae', label: 'MAE', align: 'right' as const },
    { key: 'r2', label: 'R²', align: 'right' as const },
  ]

  const restColumns = [
    { key: 'restaurant_id', label: 'ID' },
    { key: 'n_samples', label: 'Samples', align: 'right' as const },
    { key: 'rmse', label: 'RMSE', align: 'right' as const },
    { key: 'mae', label: 'MAE', align: 'right' as const },
    { key: 'mean_actual', label: 'Avg Actual', align: 'right' as const },
    { key: 'mean_predicted', label: 'Avg Predicted', align: 'right' as const },
    { key: 'risk_level', label: 'Risk', align: 'center' as const, render: (v: string) => <StatusBadge level={v} /> },
  ]

  return (
    <div className="page">
      <h1 className="page-title">Model Performance</h1>

      {chartData.length > 0 && (
        <div className="chart-section">
          <h2 className="section-title">Model Comparison</h2>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={chartData} layout="vertical" margin={{ left: 80 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
                <XAxis type="number" stroke="#666" fontSize={11} />
                <YAxis dataKey="model" type="category" stroke="#666" fontSize={11} width={120} />
                <Tooltip contentStyle={TT} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="RMSE" fill="#ef4444" barSize={14} />
                <Bar dataKey="MAE" fill="#f59e0b" barSize={14} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div className="table-section">
        <h2 className="section-title">All Models</h2>
        <DataTable columns={compColumns} data={comparison} />
      </div>

      <div className="table-section">
        <h2 className="section-title">Per-Restaurant Accuracy</h2>
        {loading ? <div className="page-loading">loading...</div> : <DataTable columns={restColumns} data={restaurants} />}
        <div className="pagination">
          <button disabled={page <= 1} onClick={() => setPage(p => p - 1)}>← Prev</button>
          <span className="page-info">{page} / {totalPages}</span>
          <button disabled={page >= totalPages} onClick={() => setPage(p => p + 1)}>Next →</button>
        </div>
      </div>
    </div>
  )
}
