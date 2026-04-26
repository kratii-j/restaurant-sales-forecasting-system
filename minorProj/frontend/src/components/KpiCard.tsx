interface KpiCardProps {
  label: string
  value: string | number
  delta?: string
  trend?: 'up' | 'down' | 'neutral'
  accent?: string
}

export default function KpiCard({ label, value, delta, trend, accent }: KpiCardProps) {
  return (
    <div className="kpi-card" style={accent ? { borderColor: accent } : undefined}>
      <span className="kpi-label">{label}</span>
      <span className="kpi-value">{value}</span>
      {delta && (
        <span className={`kpi-delta ${trend || 'neutral'}`}>
          {trend === 'up' ? '↑' : trend === 'down' ? '↓' : '—'} {delta}
        </span>
      )}
    </div>
  )
}
