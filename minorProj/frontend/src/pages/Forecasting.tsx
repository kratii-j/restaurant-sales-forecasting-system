import { useEffect, useState } from 'react'
import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'
import { api } from '../api'
import StatusBadge from '../components/StatusBadge'

const TT = { background: '#0a0a0a', border: '1px solid #1a1a1a', fontFamily: 'JetBrains Mono', fontSize: 12 }

export default function Forecasting() {
  const [restaurants, setRestaurants] = useState<any[]>([])
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [forecast, setForecast] = useState<any[]>([])
  const [prediction, setPrediction] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [form, setForm] = useState({
    date: new Date().toISOString().split('T')[0],
    discount: 0,
    promotion_flag: false,
  })

  useEffect(() => {
    api.restaurants.list({ per_page: 100, sort_by: 'votes', sort_order: 'desc' })
      .then(res => {
        setRestaurants(res.items)
        if (res.items.length) setSelectedId(res.items[0].restaurant_id)
      })
  }, [])

  useEffect(() => {
    if (!selectedId) return
    setLoading(true)
    api.restaurants.forecast(selectedId).then(setForecast).finally(() => setLoading(false))
  }, [selectedId])

  const handlePredict = async () => {
    if (!selectedId) return
    setLoading(true)
    try {
      const r = await api.predict({
        restaurant_id: selectedId,
        date: form.date,
        discount: form.discount,
        promotion_flag: form.promotion_flag,
      })
      setPrediction(r)
    } catch (e) { console.error(e) }
    setLoading(false)
  }

  return (
    <div className="page">
      <h1 className="page-title">Forecasting</h1>

      <div className="form-section">
        <div className="form-row">
          <label className="form-label">
            Restaurant
            <select value={selectedId ?? ''} onChange={e => setSelectedId(Number(e.target.value))}>
              {restaurants.map(r => (
                <option key={r.restaurant_id} value={r.restaurant_id}>
                  {r.restaurant_name} ({r.city})
                </option>
              ))}
            </select>
          </label>
          <label className="form-label">
            Date
            <input type="date" value={form.date} onChange={e => setForm(f => ({ ...f, date: e.target.value }))} />
          </label>
          <label className="form-label">
            Discount %
            <input type="number" min={0} max={100} value={form.discount} onChange={e => setForm(f => ({ ...f, discount: Number(e.target.value) }))} />
          </label>
          <label className="checkbox-label">
            <input type="checkbox" checked={form.promotion_flag} onChange={e => setForm(f => ({ ...f, promotion_flag: e.target.checked }))} />
            Promotion Active
          </label>
          <button className="primary" onClick={handlePredict} disabled={loading}>
            {loading ? 'Predicting...' : 'Predict'}
          </button>
        </div>
      </div>

      {prediction && (
        <div className="prediction-result">
          <div className="pred-grid">
            <div className="pred-card">
              <span className="pred-label">Predicted Orders</span>
              <span className="pred-value">{prediction.predicted_orders}</span>
              <span className="pred-ci">[{prediction.lower_bound_orders} — {prediction.upper_bound_orders}]</span>
            </div>
            <div className="pred-card">
              <span className="pred-label">Predicted Revenue</span>
              <span className="pred-value">₹{prediction.predicted_revenue?.toFixed(0)}</span>
              <span className="pred-ci">[₹{prediction.lower_bound_revenue?.toFixed(0)} — ₹{prediction.upper_bound_revenue?.toFixed(0)}]</span>
            </div>
            <div className="pred-card">
              <span className="pred-label">Risk</span>
              <StatusBadge level={prediction.risk_level} />
              <span className="pred-ci">Score: {prediction.risk_score}</span>
            </div>
          </div>
        </div>
      )}

      {forecast.length > 0 && (
        <div className="chart-section">
          <h2 className="section-title">Actual vs Predicted Orders</h2>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart data={forecast}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
                <XAxis dataKey="date" stroke="#666" fontSize={11} tickFormatter={v => v.slice(5)} />
                <YAxis stroke="#666" fontSize={11} />
                <Tooltip contentStyle={TT} />
                <Line type="monotone" dataKey="actual_orders" stroke="#ffffff" strokeWidth={1.5} dot={false} name="Actual" />
                <Line type="monotone" dataKey="predicted_orders" stroke="#10b981" strokeWidth={1.5} dot={false} name="Predicted" strokeDasharray="4 4" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  )
}
