import { useEffect, useState } from 'react'
import { api } from '../api'
import { MapPin, Star, Truck, BookOpen } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'

const TT = { background: '#0a0a0a', border: '1px solid #1a1a1a', fontFamily: 'JetBrains Mono', fontSize: 12 }

export default function RestaurantExplorer() {
  const [data, setData] = useState<any>({ items: [], total: 0, total_pages: 1, cities: [] })
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [city, setCity] = useState('')
  const [expandedId, setExpandedId] = useState<number | null>(null)
  const [detail, setDetail] = useState<any>(null)

  useEffect(() => {
    setLoading(true)
    api.restaurants.list({ page, search: search || undefined, city: city || undefined, per_page: 20 })
      .then(setData)
      .finally(() => setLoading(false))
  }, [page, search, city])

  const handleExpand = async (id: number) => {
    if (expandedId === id) { setExpandedId(null); setDetail(null); return }
    setExpandedId(id)
    const d = await api.restaurants.get(id)
    setDetail(d)
  }

  return (
    <div className="page">
      <h1 className="page-title">Restaurant Explorer</h1>

      <div className="explorer-controls">
        <input
          placeholder="Search restaurants..."
          value={search}
          onChange={e => { setSearch(e.target.value); setPage(1) }}
        />
        <select value={city} onChange={e => { setCity(e.target.value); setPage(1) }}>
          <option value="">All Cities</option>
          {data.cities?.map((c: string) => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>

      {loading ? (
        <div className="page-loading">loading...</div>
      ) : (
        <>
          <div className="restaurant-grid">
            {data.items.map((r: any) => (
              <div
                key={r.restaurant_id}
                className={`restaurant-card ${expandedId === r.restaurant_id ? 'expanded' : ''}`}
                onClick={() => handleExpand(r.restaurant_id)}
              >
                <div className="rc-name">{r.restaurant_name}</div>
                <div className="rc-city"><MapPin size={12} /> {r.city}</div>
                <div className="rc-cuisines">{r.cuisines}</div>
                <div className="rc-meta">
                  <span className="rc-rating"><Star size={12} /> {r.aggregate_rating}</span>
                  <span>{r.votes} votes</span>
                  <span>₹{r.average_cost_for_two} for two</span>
                </div>
                <div style={{ marginTop: 8, display: 'flex', gap: 4 }}>
                  {r.has_online_delivery && <span className="rc-tag"><Truck size={10} /> Delivery</span>}
                  {r.has_table_booking && <span className="rc-tag"><BookOpen size={10} /> Booking</span>}
                </div>

                {expandedId === r.restaurant_id && detail && (
                  <div className="rc-detail">
                    <div style={{ fontSize: 12, color: '#999', marginBottom: 4 }}>
                      {detail.locality} · {detail.currency} · Price Range: {detail.price_range}
                    </div>
                    <h3 className="section-title" style={{ marginTop: 12 }}>Recent Orders (30d)</h3>
                    <ResponsiveContainer width="100%" height={160}>
                      <BarChart data={detail.recent_orders}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
                        <XAxis dataKey="date" stroke="#666" fontSize={10} tickFormatter={(v: string) => v.slice(8)} />
                        <YAxis stroke="#666" fontSize={10} />
                        <Tooltip contentStyle={TT} />
                        <Bar dataKey="total_orders" fill="#10b981" name="Orders" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="pagination">
            <button disabled={page <= 1} onClick={() => setPage(p => p - 1)}>← Prev</button>
            <span className="page-info">{page} / {data.total_pages} ({data.total} restaurants)</span>
            <button disabled={page >= data.total_pages} onClick={() => setPage(p => p + 1)}>Next →</button>
          </div>
        </>
      )}
    </div>
  )
}
